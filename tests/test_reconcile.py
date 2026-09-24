"""Tests for the hourly sweep.

`reconcile` is the cron: it sends one `repo/sweep.requested` per tracked repo.
`sweep_repo` lists what changed since that repo's cursor, fans it out, and moves
the cursor. The mocked harness replays the function once per step, the same way
the server does, which is what makes the clock test below meaningful.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import inngest
import pytest
from inngest._internal import server_lib
from inngest.experimental import mocked

from gh_issue.functions import reconcile as mod
from gh_issue.functions.index_issue import IssueChanged
from gh_issue.functions.reconcile import SweepRequested
from gh_issue.github import api
from gh_issue.github.api import IssuePage
from gh_issue.github.types import Issue
from gh_issue.index.cursors import Cursors
from gh_issue.settings import EVENT_ISSUE_CHANGED, EVENT_SWEEP_REQUESTED

REPO = "owner/name"
LAST_SWEEP = "2026-09-24T09:00:00Z"


class RecordingClient(mocked.Inngest):
    def __init__(self) -> None:
        super().__init__(app_id="test")
        self.sent: list[server_lib.Event] = []

    def send_sync(
        self,
        events: server_lib.Event | list[server_lib.Event],
        *,
        skip_middleware: bool = False,
    ) -> list[str]:
        batch = events if isinstance(events, list) else [events]
        self.sent.extend(batch)
        return super().send_sync(batch, skip_middleware=skip_middleware)


def issue(number: int, *, pull_request: bool = False) -> Issue:
    made: Issue = {
        "number": number,
        "title": f"issue {number}",
        "body": "body",
        "state": "open",
        "labels": [],
        "user": {"login": "someone"},
        "created_at": "2026-01-01T00:00:00Z",
        "comments": 0,
    }
    if pull_request:
        made["pull_request"] = {"url": "https://api.github.com/..."}
    return made


@dataclass
class Rig:
    cursors: Cursors
    pages: dict[str | None, IssuePage] = field(default_factory=dict)
    requested: list[tuple[str, str | None, str | None]] = field(default_factory=list)
    ticks: list[str] = field(default_factory=list)
    client: RecordingClient = field(default_factory=RecordingClient)

    def sweep(self, repo: str = REPO) -> Any:
        return self._trigger(
            mod.sweep_repo, inngest.Event(name=EVENT_SWEEP_REQUESTED, data={"repo": repo})
        )

    def reconcile(self) -> Any:
        return self._trigger(mod.reconcile, inngest.Event(name="inngest/scheduled.timer"))

    def _trigger(self, fn: inngest.Function[Any], event: inngest.Event) -> Any:
        result = mocked.trigger(fn, event, self.client)
        assert result.status is mocked.Status.COMPLETED, result.error
        return result.output

    def sent(self, name: str) -> list[server_lib.Event]:
        return [event for event in self.client.sent if event.name == name]

    def swept(self) -> list[str]:
        """Repos the cron asked to sweep."""
        return sorted(
            cast(SweepRequested, event.data)["repo"] for event in self.sent(EVENT_SWEEP_REQUESTED)
        )

    def fanned_out(self) -> list[int]:
        return [
            cast(IssueChanged, event.data)["issue"]["number"]
            for event in self.sent(EVENT_ISSUE_CHANGED)
        ]


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    harness = Rig(cursors=Cursors(str(tmp_path)))

    def fake_list_issues(
        repo: str, cursor: str | None = None, since: str | None = None
    ) -> IssuePage:
        harness.requested.append((repo, cursor, since))
        return harness.pages[cursor]

    def fake_now() -> str:
        """A clock that moves forward every time anyone looks at it."""
        tick = f"2026-09-24T10:00:{len(harness.ticks):02d}Z"
        harness.ticks.append(tick)
        return tick

    monkeypatch.setattr(mod, "cursors", lambda: harness.cursors)
    monkeypatch.setattr(api, "list_issues", fake_list_issues)
    monkeypatch.setattr(mod, "_now", fake_now)
    yield harness


# reconcile ---------------------------------------------------------------


def test_reconcile_requests_one_sweep_per_tracked_repo(rig: Rig) -> None:
    rig.cursors.set("a/b", LAST_SWEEP)
    rig.cursors.set("c/d", LAST_SWEEP)

    output = rig.reconcile()

    assert rig.swept() == ["a/b", "c/d"]
    assert output == {"repos": 2}


def test_reconcile_with_nothing_tracked_sends_nothing(rig: Rig) -> None:
    assert rig.reconcile() == {"repos": 0}
    assert rig.client.sent == []


# sweep_repo --------------------------------------------------------------


def test_sweep_lists_what_changed_since_the_cursor(rig: Rig) -> None:
    rig.cursors.set(REPO, LAST_SWEEP)
    rig.pages[None] = {"issues": [issue(1), issue(2, pull_request=True), issue(3)], "next": None}

    output = rig.sweep()

    assert rig.requested == [(REPO, None, LAST_SWEEP)]
    assert rig.fanned_out() == [1, 3]
    assert output["changed"] == 2


def test_sweep_pages_through_a_long_gap(rig: Rig) -> None:
    """After downtime there can be more than one page of changes. The cursor
    has not moved while the worker was down, so the gap is all still there."""
    rig.cursors.set(REPO, LAST_SWEEP)
    rig.pages[None] = {"issues": [issue(1), issue(2)], "next": "PAGE-2"}
    rig.pages["PAGE-2"] = {"issues": [issue(3)], "next": "PAGE-3"}
    rig.pages["PAGE-3"] = {"issues": [issue(4)], "next": None}

    output = rig.sweep()

    assert [cursor for _, cursor, _ in rig.requested] == [None, "PAGE-2", "PAGE-3"]
    assert all(since == LAST_SWEEP for _, _, since in rig.requested)
    assert rig.fanned_out() == [1, 2, 3, 4]
    assert output["changed"] == 4


def test_the_cursor_moves_to_when_the_sweep_started(rig: Rig) -> None:
    """Anything updated while the sweep is listing must be caught next time, so
    the cursor has to move to a moment *before* the listing, not after it.

    That moment is read inside a step. The function body replays once per step,
    so a clock read in plain Python would be read again on every replay, and the
    cursor would end up at whatever the last replay saw — after the listing.
    This clock ticks on every read, so that mistake would show up here."""
    rig.cursors.set(REPO, LAST_SWEEP)
    rig.pages[None] = {"issues": [issue(1)], "next": None}

    rig.sweep()

    assert rig.ticks, "the clock was never read"
    assert rig.cursors.get(REPO) == rig.ticks[0]
    assert len(rig.ticks) == 1


def test_a_quiet_hour_still_moves_the_cursor(rig: Rig) -> None:
    rig.cursors.set(REPO, LAST_SWEEP)
    rig.pages[None] = {"issues": [], "next": None}

    output = rig.sweep()

    assert rig.sent(EVENT_ISSUE_CHANGED) == []
    assert output["changed"] == 0
    assert rig.cursors.get(REPO) != LAST_SWEEP


def test_a_repo_without_a_cursor_is_left_alone(rig: Rig) -> None:
    """Listing with no `since` would return the whole repo. That is the
    backfill's job, not the sweep's."""
    output = rig.sweep("never/indexed")

    assert rig.requested == []
    assert rig.client.sent == []
    assert output == {"repo": "never/indexed", "skipped": "no cursor"}
