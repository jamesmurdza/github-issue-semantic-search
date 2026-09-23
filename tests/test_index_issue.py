"""Tests for the indexing run.

`inngest.experimental.mocked` executes a function's steps without a server, so
these exercise the real orchestration: which steps run, in what order, and what
each one is handed. Only the network and the embedding model are faked — both
have their own tests, and the model is far too slow to run a few hundred times
here.

The mocked harness is not the server, so these do not replace the end-to-end run
in `gh-issue index`. They do catch the things that are painful to notice there.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import inngest
import pytest
from inngest.experimental import mocked

from gh_issue.chunking import Chunk, EmbeddedChunk
from gh_issue.embed import BATCH as EMBED_BATCH
from gh_issue.embed import DIM
from gh_issue.functions import index_issue as mod
from gh_issue.github import api
from gh_issue.github.types import Comment, Issue
from gh_issue.index.store import Store

REPO = "owner/name"
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


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    harness = Rig(store=Store(dim=DIM, path=str(tmp_path)), comments=list(COMMENTS))

    def fake_list_comments(repo: str, number: int) -> list[Comment]:
        harness.comment_fetches.append(number)
        return list(harness.comments)

    def fake_embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
        harness.embed_batches.append(len(chunks))
        return [
            {**chunk, "vector": [float(i + 1)] + [0.0] * (DIM - 1)}
            for i, chunk in enumerate(chunks)
        ]

    monkeypatch.setattr(mod, "store", lambda: harness.store)
    monkeypatch.setattr(api, "list_comments", fake_list_comments)
    monkeypatch.setattr(mod, "embed_chunks", fake_embed_chunks)
    yield harness
    harness.store.close()


def test_indexes_the_body_and_every_comment(rig: Rig) -> None:
    output = rig.run(ISSUE)

    assert output == {"repo": REPO, "issue": 412, "embedded": 3, "deleted": 0}
    assert sorted(rig.store.hashes_for_issue(REPO, 412)) == [
        "owner/name#412:body",
        "owner/name#412:comment:1",
        "owner/name#412:comment:2",
    ]


def test_an_issue_with_no_comments_never_calls_github(rig: Rig) -> None:
    """The listing already gave us the count, so the call would be pure waste."""
    output = rig.run({**ISSUE, "comments": 0})

    assert rig.comment_fetches == []
    assert output["embedded"] == 1
    assert list(rig.store.hashes_for_issue(REPO, 412)) == ["owner/name#412:body"]


def test_running_twice_embeds_nothing_the_second_time(rig: Rig) -> None:
    rig.run(ISSUE)
    output = rig.run(ISSUE)

    assert output == {"repo": REPO, "issue": 412, "embedded": 0, "deleted": 0}
    assert rig.embedded == 3
    assert len(rig.store.hashes_for_issue(REPO, 412)) == 3


def test_an_edited_comment_re_embeds_only_itself(rig: Rig) -> None:
    rig.run(ISSUE)
    before = rig.embedded

    rig.set_comments([COMMENTS[0], comment(2, "actually it was a debounce bug")])
    output = rig.run(ISSUE)

    assert rig.embedded - before == 1
    assert output["embedded"] == 1
    assert output["deleted"] == 0
    assert len(rig.store.hashes_for_issue(REPO, 412)) == 3


def test_a_removed_comment_is_cleaned_up(rig: Rig) -> None:
    """The orphan case: the index has to shrink, not only grow."""
    rig.run(ISSUE)

    rig.set_comments([COMMENTS[0]])
    output = rig.run({**ISSUE, "comments": 1})

    assert output["deleted"] == 1
    assert sorted(rig.store.hashes_for_issue(REPO, 412)) == [
        "owner/name#412:body",
        "owner/name#412:comment:1",
    ]


def test_a_deleted_issue_purges_the_whole_thread(rig: Rig) -> None:
    rig.run(ISSUE)
    before = rig.embedded

    output = rig.run(ISSUE, deleted=True)

    assert output == {"repo": REPO, "issue": 412, "purged": True}
    assert rig.store.hashes_for_issue(REPO, 412) == {}
    assert rig.comment_fetches == [412]  # the purge path fetches nothing
    assert rig.embedded == before


def test_embedding_is_batched_against_the_step_output_cap(rig: Rig) -> None:
    """A 250-comment thread must not try to return 251 vectors from one step."""
    rig.set_comments([comment(i, f"reply number {i}") for i in range(250)])

    output = rig.run({**ISSUE, "comments": 250})

    assert rig.embed_batches == [EMBED_BATCH, 251 - EMBED_BATCH]
    assert output["embedded"] == 251
    assert len(rig.store.hashes_for_issue(REPO, 412)) == 251
