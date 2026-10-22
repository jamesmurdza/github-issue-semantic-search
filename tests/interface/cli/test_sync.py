"""`gh-issue sync`: what it decides to queue, given what is recorded.

`sync` does no indexing itself. It reads the local records — backfill progress,
the sweep cursor, failure markers — and the index's size, and sends Inngest the
events that bring the repo up to date. So every test here is about which events
go out, with which ids, and what is recorded afterwards.
"""

import socket
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import inngest
import pytest

from gh_issue.functions.events import (
    EVENT_BACKFILL_PAGE,
    EVENT_ISSUE_CHANGED,
    EVENT_SWEEP_REQUESTED,
)
from gh_issue.github import api
from gh_issue.github.types import Issue
from gh_issue.index.records.backfills import BackfillProgress, Backfills
from gh_issue.index.records.cursors import Cursors
from gh_issue.index.records.failures import Failures
from gh_issue.index.store import RepoStats
from gh_issue.interface.cli import sync as mod

REPO = "owner/name"
NOW = "2026-09-25T10:00:00Z"


def issue(number: int, *, state: str = "open") -> Issue:
    return {
        "number": number,
        "title": f"issue {number}",
        "body": "body",
        "state": state,
        "labels": [],
        "user": {"login": "someone"},
        "created_at": "2026-01-01T00:00:00Z",
        "comments": 0,
    }


def progress(chain: str, next_cursor: str | None, *, finished: bool) -> BackfillProgress:
    return {
        "repo": REPO,
        "chain": chain,
        "next": next_cursor,
        "finished": finished,
        "at": "2026-09-25T09:30:00Z",
    }


@dataclass
class FakeStore:
    chunks: dict[str, int]

    def stats(self) -> dict[str, RepoStats]:
        return {repo: {"chunks": n, "issues": n} for repo, n in self.chunks.items() if n}


@dataclass
class Rig:
    backfills: Backfills
    cursors: Cursors
    failures: Failures
    store: FakeStore = field(default_factory=lambda: FakeStore({}))
    github_repos: set[str] = field(default_factory=lambda: {REPO, "a/b", "c/d"})
    github_issues: dict[int, Issue] = field(default_factory=dict)
    sent: list[inngest.Event] = field(default_factory=list)
    said: list[str] = field(default_factory=list)
    worker_up: bool = True

    def sync(self, repo: str = REPO) -> list[str]:
        return mod.sync_repo(repo)

    def run(self, *repos: str) -> None:
        mod.run(list(repos), say=self.said.append)

    def chain(self) -> str:
        recorded = self.backfills.get(REPO)
        assert recorded is not None, "no backfill recorded"
        return recorded["chain"]

    def named(self, name: str) -> list[inngest.Event]:
        return [event for event in self.sent if event.name == name]

    def finished_and_indexed(self, repo: str = REPO) -> None:
        self.backfills.set({**progress("c9", None, finished=True), "repo": repo})
        self.cursors.set(repo, "2026-09-25T09:00:00Z")
        self.store.chunks[repo] = 500


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    harness = Rig(
        backfills=Backfills(str(tmp_path / "backfills")),
        cursors=Cursors(str(tmp_path / "cursors")),
        failures=Failures(str(tmp_path / "failed")),
    )
    chains = iter(["chain-1", "chain-2", "chain-3"])

    monkeypatch.setattr(mod, "backfills", lambda: harness.backfills)
    monkeypatch.setattr(mod, "cursors", lambda: harness.cursors)
    monkeypatch.setattr(mod, "failures", lambda: harness.failures)
    monkeypatch.setattr(mod, "store", lambda: harness.store)
    monkeypatch.setattr(mod, "send", harness.sent.extend)
    monkeypatch.setattr(mod, "now", lambda: NOW)
    monkeypatch.setattr(mod, "new_chain", lambda: next(chains))
    monkeypatch.setattr(mod, "worker_running", lambda: harness.worker_up)
    monkeypatch.setattr(mod, "preflight", lambda production: [])
    monkeypatch.setattr(api, "repo_exists", lambda repo: repo in harness.github_repos)
    monkeypatch.setattr(api, "get_issue", lambda repo, n: harness.github_issues.get(n))
    yield harness


# Starting --------------------------------------------------------------------------


def test_a_new_repo_starts_a_backfill(rig: Rig) -> None:
    rig.sync()

    (start,) = rig.sent
    assert start.name == EVENT_BACKFILL_PAGE
    assert start.id == "backfill:chain-1:start"
    assert start.data == {"repo": REPO, "chain": "chain-1"}


def test_starting_records_progress_and_hands_the_sweep_its_starting_point(rig: Rig) -> None:
    """Recorded before anything is sent, so a second `sync` sees a backfill in
    progress. The sweep cursor is the moment the backfill started: anything
    changed after that is the sweep's to catch, whichever page it was on."""
    rig.sync()

    assert rig.backfills.get(REPO) == {
        "repo": REPO,
        "chain": "chain-1",
        "next": None,
        "finished": False,
        "at": NOW,
    }
    assert rig.cursors.get(REPO) == NOW


def test_a_repo_github_does_not_know_is_refused_before_anything_happens(rig: Rig) -> None:
    with pytest.raises(mod.RepoNotFound, match="owner/typo"):
        rig.sync("owner/typo")

    assert rig.sent == []
    assert rig.backfills.get("owner/typo") is None
    assert rig.cursors.get("owner/typo") is None


def test_syncing_again_before_anything_is_indexed_does_not_start_a_second_backfill(
    rig: Rig,
) -> None:
    """Qdrant is still empty a minute into a backfill. Deciding on that alone
    would start a second full backfill; the record says one is under way."""
    rig.sync()
    rig.sent.clear()

    rig.sync()

    assert [event.id for event in rig.named(EVENT_BACKFILL_PAGE)] == ["backfill:chain-1:start"]
    assert rig.chain() == "chain-1"


# Bringing up to date ---------------------------------------------------------------


def test_an_unfinished_backfill_resumes_from_its_last_recorded_page(rig: Rig) -> None:
    """With the id the chain itself would use for that page, so a chain that is
    still alive has already sent it and Inngest drops this one."""
    rig.backfills.set(progress("c9", "CURSOR-7", finished=False))
    rig.cursors.set(REPO, "2026-09-25T09:00:00Z")
    rig.store.chunks[REPO] = 500

    rig.sync()

    (resume,) = rig.named(EVENT_BACKFILL_PAGE)
    assert resume.id == "backfill:c9:CURSOR-7"
    assert resume.data == {"repo": REPO, "chain": "c9", "cursor": "CURSOR-7"}
    (sweep,) = rig.named(EVENT_SWEEP_REQUESTED)
    assert sweep.data == {"repo": REPO}


def test_a_finished_backfill_only_sweeps(rig: Rig) -> None:
    rig.finished_and_indexed()

    rig.sync()

    assert [event.name for event in rig.sent] == [EVENT_SWEEP_REQUESTED]


def test_a_finished_backfill_with_nothing_in_the_index_starts_over(rig: Rig) -> None:
    """The index was wiped — a new Qdrant volume, say. Only a finished record
    makes an empty index mean this."""
    rig.backfills.set(progress("c9", None, finished=True))

    rig.sync()

    (start,) = rig.sent
    assert start.id == "backfill:chain-1:start"
    assert rig.chain() == "chain-1"


# Retrying failures -----------------------------------------------------------------


def add_failure(rig: Rig, number: int) -> None:
    rig.failures.add({"repo": REPO, "number": number, "error": "boom", "run_id": "r", "at": NOW})


def test_failed_issues_are_read_fresh_from_github_and_sent_again(rig: Rig) -> None:
    rig.finished_and_indexed()
    add_failure(rig, 412)
    add_failure(rig, 418)
    rig.github_issues = {412: issue(412, state="closed"), 418: issue(418)}

    lines = rig.sync()

    assert [event.data for event in rig.named(EVENT_ISSUE_CHANGED)] == [
        {"repo": REPO, "issue": issue(412, state="closed")},
        {"repo": REPO, "issue": issue(418)},
    ]
    assert rig.failures.for_repo(REPO) == []
    assert any("#412" in line and "#418" in line for line in lines)


def test_a_failed_issue_github_no_longer_has_is_dropped(rig: Rig) -> None:
    rig.finished_and_indexed()
    add_failure(rig, 412)

    lines = rig.sync()

    assert rig.named(EVENT_ISSUE_CHANGED) == []
    assert rig.failures.for_repo(REPO) == []
    assert any("#412" in line for line in lines)


def test_markers_stay_when_sending_fails(rig: Rig, monkeypatch: pytest.MonkeyPatch) -> None:
    rig.finished_and_indexed()
    add_failure(rig, 412)
    rig.github_issues = {412: issue(412)}

    def unreachable(events: Any) -> None:
        raise ConnectionError("dev server is down")

    monkeypatch.setattr(mod, "send", unreachable)

    with pytest.raises(ConnectionError):
        rig.sync()
    assert [f["number"] for f in rig.failures.for_repo(REPO)] == [412]


def test_sending_to_a_dev_server_that_is_not_there_names_it_and_its_setting(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        origin = f"http://127.0.0.1:{probe.getsockname()[1]}"
    absent = inngest.Inngest(
        app_id="probe", is_production=False, api_base_url=origin, event_api_base_url=origin
    )
    monkeypatch.setattr(mod, "client", absent)

    with pytest.raises(mod.CannotSend) as raised:
        mod.send([inngest.Event(name="probe")])

    assert str(raised.value).startswith(
        f"Couldn't send events to the Inngest dev server at {origin} (inngest.dev):"
    )


def test_starting_over_clears_old_failure_markers(rig: Rig) -> None:
    """A full backfill indexes those issues anyway."""
    add_failure(rig, 412)

    rig.sync()

    assert rig.failures.for_repo(REPO) == []


# The command -----------------------------------------------------------------------


def test_with_no_repo_named_it_syncs_every_repo_synced_before(rig: Rig) -> None:
    rig.finished_and_indexed("a/b")
    rig.finished_and_indexed("c/d")

    rig.run()

    assert sorted(str(event.data["repo"]) for event in rig.named(EVENT_SWEEP_REQUESTED)) == [
        "a/b",
        "c/d",
    ]


def test_with_nothing_synced_yet_it_says_how_to_start(rig: Rig) -> None:
    with pytest.raises(SystemExit, match="gh-issue sync owner/name"):
        rig.run()


def test_an_unknown_repo_is_reported_and_the_rest_still_sync(rig: Rig) -> None:
    rig.finished_and_indexed()

    with pytest.raises(SystemExit):
        rig.run("owner/typo", REPO)

    assert any("owner/typo" in line and "not found" in line for line in rig.said)
    assert len(rig.named(EVENT_SWEEP_REQUESTED)) == 1


def test_it_warns_when_no_worker_is_running(rig: Rig) -> None:
    """The events still go out and wait in Inngest; nothing runs them yet."""
    rig.finished_and_indexed()
    rig.worker_up = False

    rig.run(REPO)

    assert rig.sent
    assert "gh-issue worker" in rig.said[-1]


def test_no_warning_when_a_worker_is_up(rig: Rig) -> None:
    rig.finished_and_indexed()

    rig.run(REPO)

    assert not any("gh-issue worker" in line for line in rig.said)


def test_it_refuses_to_run_when_preflight_finds_problems(
    rig: Rig, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mod, "preflight", lambda production: ["GITHUB_TOKEN is not set."])
    with pytest.raises(SystemExit, match="GITHUB_TOKEN"):
        rig.run(REPO)
    assert rig.sent == []
