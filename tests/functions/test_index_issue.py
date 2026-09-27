"""Tests for the indexing run.

`inngest.experimental.mocked` executes a function's steps without a server, so
these exercise the real orchestration: which steps run, in what order, and what
each one is handed. Only the network and the embedding model are faked — both
have their own tests, and the model is far too slow to run a few hundred times
here.

The mocked harness is not the server, so these do not replace a real run on the
dev server. They do catch the things that are painful to notice there.
"""

from collections.abc import Callable, Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import inngest
import pytest
from inngest.experimental import mocked

from gh_issue.functions import index_issue as mod
from gh_issue.github import api
from gh_issue.github.types import Comment, Issue
from gh_issue.index import storage
from gh_issue.index.chunking import Chunk, EmbeddedChunk
from gh_issue.index.embed import BATCH as EMBED_BATCH
from gh_issue.index.embed import DIM
from gh_issue.index.records.failures import Failures
from gh_issue.index.store import Store

REPO = "owner/name"
SYNCED_AT = "2026-09-25T09:00:00Z"
CLIENT = mocked.Inngest(app_id="test")

ISSUE: Issue = {
    "number": 412,
    "title": "App freezes when I scroll fast",
    "body": "Happens on long lists.",
    "state": "open",
    "labels": [{"name": "bug"}],
    "user": {"login": "reporter"},
    "created_at": "2026-01-01T00:00:00Z",
    "comments": 2,
}


def comment(comment_id: int, body: str) -> Comment:
    return {
        "id": comment_id,
        "body": body,
        "user": {"login": "maintainer"},
        "created_at": "2026-01-02T00:00:00Z",
    }


COMMENTS: list[Comment] = [comment(1, "first reply"), comment(2, "second reply")]


@dataclass
class Rig:
    """A real store, a fake GitHub and a fake model, plus what got called."""

    store: Store
    comments: list[Comment]
    comment_fetches: list[int] = field(default_factory=list)
    embed_batches: list[int] = field(default_factory=list)
    # Runs inside the comment fetch, to change the world partway through a run.
    during_fetch: Callable[[], None] = lambda: None

    @property
    def embedded(self) -> int:
        return sum(self.embed_batches)

    def set_comments(self, comments: list[Comment]) -> None:
        self.comments[:] = comments

    def run(self, issue: Issue, *, deleted: bool = False) -> Any:
        data: dict[str, Any] = {"repo": REPO, "issue": issue}
        if deleted:
            data["deleted"] = True
        result = mocked.trigger(
            mod.index_issue, inngest.Event(name="repo/issue.changed", data=data), CLIENT
        )
        assert result.status is mocked.Status.COMPLETED, result.error
        return result.output


def remove_repo() -> None:
    """What `gh-issue remove` does to the records the guard reads."""
    storage.cursors().delete(REPO)
    storage.backfills().delete(REPO)


@pytest.fixture
def rig(tmp_path: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    harness = Rig(store=Store(dim=DIM, path=str(tmp_path)), comments=list(COMMENTS))
    # Synced, as `gh-issue sync` leaves a repo before any of its issues arrive.
    storage.cursors().set(REPO, SYNCED_AT)

    def fake_list_comments(repo: str, number: int) -> list[Comment]:
        harness.comment_fetches.append(number)
        harness.during_fetch()
        return list(harness.comments)

    def fake_embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
        harness.embed_batches.append(len(chunks))
        return [
            {
                **chunk,
                "vector": [float(i + 1)] + [0.0] * (DIM - 1),
                "sparse": {"indices": [i + 1], "values": [1.0]},
            }
            for i, chunk in enumerate(chunks)
        ]

    monkeypatch.setattr(mod, "store", lambda: harness.store)
    monkeypatch.setattr(api, "list_comments", fake_list_comments)
    monkeypatch.setattr(mod, "embed_chunks", fake_embed_chunks)
    yield harness
    harness.store.close()


def test_indexes_the_body_and_every_comment(rig: Rig) -> None:
    output = rig.run(ISSUE)

    assert output == {
        "repo": REPO,
        "issue": 412,
        "embedded": 3,
        "deleted": 0,
        "refreshed": False,
    }
    assert sorted(rig.store.fingerprints_for_issue(REPO, 412)) == [
        "owner/name#412:body",
        "owner/name#412:comment:1",
        "owner/name#412:comment:2",
    ]


def test_an_issue_with_no_comments_never_calls_github(rig: Rig) -> None:
    """The listing already gave us the count, so the call would be pure waste."""
    output = rig.run({**ISSUE, "comments": 0})

    assert rig.comment_fetches == []
    assert output["embedded"] == 1
    assert list(rig.store.fingerprints_for_issue(REPO, 412)) == ["owner/name#412:body"]


def test_running_twice_embeds_nothing_the_second_time(rig: Rig) -> None:
    rig.run(ISSUE)
    output = rig.run(ISSUE)

    assert output == {
        "repo": REPO,
        "issue": 412,
        "embedded": 0,
        "deleted": 0,
        "refreshed": False,
    }
    assert rig.embedded == 3
    assert len(rig.store.fingerprints_for_issue(REPO, 412)) == 3


def stored_meta(rig: Rig) -> set[tuple[str, str, tuple[str, ...]]]:
    """Every distinct copy of the issue metadata across the thread's chunks."""
    return {
        (chunk["title"], chunk["state"], tuple(chunk["labels"]))
        for chunk in rig.store.chunks_for_issue(REPO, 412)
    }


def test_closing_an_issue_updates_every_chunk_without_re_embedding(rig: Rig) -> None:
    """Closing changes no text, so no hash moves — without the refresh, every
    chunk would say `open` forever and a `state: closed` search would miss it."""
    rig.run(ISSUE)
    before = rig.embedded

    output = rig.run({**ISSUE, "state": "closed", "labels": [{"name": "wontfix"}]})

    assert output["refreshed"] is True
    assert rig.embedded == before
    assert stored_meta(rig) == {("App freezes when I scroll fast", "closed", ("wontfix",))}


def test_retitling_re_embeds_the_body_and_refreshes_the_comments(rig: Rig) -> None:
    """The title is part of the body's text, so the body is embedded again. The
    comments only carry a copy of it, which the refresh rewrites."""
    rig.run(ISSUE)
    before = rig.embedded

    output = rig.run({**ISSUE, "title": "Scrolling long lists freezes the app"})

    assert output["embedded"] == 1
    assert rig.embedded - before == 1
    assert output["refreshed"] is True
    assert stored_meta(rig) == {("Scrolling long lists freezes the app", "open", ("bug",))}


def test_an_edited_comment_re_embeds_only_itself(rig: Rig) -> None:
    rig.run(ISSUE)
    before = rig.embedded

    rig.set_comments([COMMENTS[0], comment(2, "actually it was a debounce bug")])
    output = rig.run(ISSUE)

    assert rig.embedded - before == 1
    assert output["embedded"] == 1
    assert output["deleted"] == 0
    assert output["refreshed"] is False
    assert len(rig.store.fingerprints_for_issue(REPO, 412)) == 3


def test_a_removed_comment_is_cleaned_up(rig: Rig) -> None:
    """The orphan case: the index has to shrink, not only grow."""
    rig.run(ISSUE)

    rig.set_comments([COMMENTS[0]])
    output = rig.run({**ISSUE, "comments": 1})

    assert output["deleted"] == 1
    assert sorted(rig.store.fingerprints_for_issue(REPO, 412)) == [
        "owner/name#412:body",
        "owner/name#412:comment:1",
    ]


def test_a_deleted_issue_purges_the_whole_thread(rig: Rig) -> None:
    rig.run(ISSUE)
    before = rig.embedded

    output = rig.run(ISSUE, deleted=True)

    assert output == {"repo": REPO, "issue": 412, "purged": True}
    assert rig.store.fingerprints_for_issue(REPO, 412) == {}
    assert rig.comment_fetches == [412]  # the purge path fetches nothing
    assert rig.embedded == before


def test_embedding_is_batched_against_the_step_output_cap(rig: Rig) -> None:
    """A 250-comment thread must not try to return 251 vectors from one step."""
    rig.set_comments([comment(i, f"reply number {i}") for i in range(250)])

    output = rig.run({**ISSUE, "comments": 250})

    assert rig.embed_batches == [EMBED_BATCH, 251 - EMBED_BATCH]
    assert output["embedded"] == 251
    assert len(rig.store.fingerprints_for_issue(REPO, 412)) == 251


# A repo removed while its issues are still queued -------------------------------


def test_an_issue_of_a_removed_repo_is_skipped_without_calling_github(rig: Rig) -> None:
    """`gh-issue remove` can't reach events already queued in Inngest, so each
    run checks the repo is still synced before doing anything."""
    remove_repo()

    output = rig.run(ISSUE)

    assert output == {"repo": REPO, "issue": 412, "skipped": "not synced"}
    assert rig.comment_fetches == []
    assert rig.embedded == 0
    assert rig.store.fingerprints_for_issue(REPO, 412) == {}


def test_a_repo_removed_mid_run_gets_no_more_writes(rig: Rig) -> None:
    """The check is plain Python, not a step, so it runs again on every replay:
    a removal takes effect at the next step instead of after the whole run."""
    rig.during_fetch = remove_repo

    output = rig.run(ISSUE)

    assert output["skipped"] == "not synced"
    assert rig.comment_fetches == [412]
    assert rig.embedded == 0
    assert rig.store.fingerprints_for_issue(REPO, 412) == {}


def test_one_run_per_issue_and_the_newest_event_wins() -> None:
    """Two runs on the same issue at once undo each other's work: the one that
    finishes last deletes a comment the other just wrote, or puts back a state
    the other just replaced. Both were reproduced with this function.

    Inngest's singleton, keyed per issue in cancel mode, lets only one run per
    issue exist and cancels it when a newer event arrives. The mocked harness
    does not enforce singleton, so this pins the configuration; the behaviour
    was measured on the dev server — a newer event cancels a run mid-flight and
    a run waiting to retry, and from 0.2 s apart the newest event always wins.
    """
    singleton = mod.index_issue.get_config("http://unused").main.singleton

    assert singleton is not None
    assert singleton.mode == "cancel"
    assert singleton.key == 'event.data.repo + "#" + string(event.data.issue.number)'


# When an issue fails every retry ---------------------------------------------


def failed_run_event() -> inngest.Event:
    """`inngest/function.failed` as the dev server delivered it to a Python
    on_failure handler, trimmed to the fields that matter here."""
    return inngest.Event(
        name="inngest/function.failed",
        data={
            "error": {"error": "NonRetriableError", "message": "GitHub said 502"},
            "event": {
                "name": "repo/issue.changed",
                "data": {"repo": REPO, "issue": ISSUE},
            },
            "function_id": "gh-issue-index-issue",
            "run_id": "01M3BWHK2R1YWKPAHSP2A8MGEB",
        },
    )


def test_index_issue_has_a_failure_handler() -> None:
    assert mod.index_issue.get_config("http://unused").on_failure is not None


def run_failure_handler() -> None:
    """The mocked harness only calls a function's main handler, so the failure
    handler is wrapped as a function of its own here and handed the event the
    dev server sends it."""
    handler = CLIENT.create_function(
        fn_id="failure-handler", trigger=inngest.TriggerEvent(event="inngest/function.failed")
    )(mod.on_index_failure)
    result = mocked.trigger(handler, failed_run_event(), CLIENT)
    assert result.status is mocked.Status.COMPLETED, result.error


@pytest.mark.usefixtures("data_dir")
def test_a_run_that_fails_every_retry_leaves_a_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The marker names the issue, not a copy of it: `sync` reads the issue
    from GitHub again before retrying, so a retry never carries an old copy."""
    storage.cursors().set(REPO, SYNCED_AT)
    failures = Failures(str(tmp_path))
    ticks: list[str] = []

    def fake_now() -> str:
        ticks.append(f"2026-09-25T10:00:{len(ticks):02d}Z")
        return ticks[-1]

    monkeypatch.setattr(mod, "failures", lambda: failures)
    monkeypatch.setattr(mod, "now", fake_now)

    run_failure_handler()

    assert failures.for_repo(REPO) == [
        {
            "repo": REPO,
            "number": 412,
            "error": "GitHub said 502",
            "run_id": "01M3BWHK2R1YWKPAHSP2A8MGEB",
            "at": "2026-09-25T10:00:00Z",
        }
    ]
    assert ticks == ["2026-09-25T10:00:00Z"]


@pytest.mark.usefixtures("data_dir")
def test_a_failure_in_a_removed_repo_leaves_no_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Nothing would ever clear it: `sync` only retries repos that are synced."""
    failures = Failures(str(tmp_path))
    monkeypatch.setattr(mod, "failures", lambda: failures)

    run_failure_handler()

    assert failures.for_repo(REPO) == []
