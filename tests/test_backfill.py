"""Tests for the backfill.

Each run handles one page and sends itself the next, so what matters is exactly
what each run sends: which issues fan out, and whether — and how — the chain
carries on. The mocked client discards sent events, so these record them.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, cast

import inngest
import pytest
from inngest._internal import server_lib
from inngest.experimental import mocked

from gh_issue.functions import backfill as mod
from gh_issue.functions.index_issue import IssueChanged
from gh_issue.github import api
from gh_issue.github.api import IssuePage
from gh_issue.github.types import Issue
from gh_issue.index.backfills import BackfillProgress, Backfills
from gh_issue.settings import EVENT_BACKFILL_PAGE, EVENT_ISSUE_CHANGED

REPO = "owner/name"


class RecordingClient(mocked.Inngest):
    """The mocked client, except it keeps what it was asked to send, and notes
    each send on a timeline it can share with the progress store."""

    def __init__(self, timeline: list[str]) -> None:
        super().__init__(app_id="test")
        self.sent: list[server_lib.Event] = []
        self.timeline = timeline

    def send_sync(
        self,
        events: server_lib.Event | list[server_lib.Event],
        *,
        skip_middleware: bool = False,
    ) -> list[str]:
        batch = events if isinstance(events, list) else [events]
        self.sent.extend(batch)
        if batch:
            self.timeline.append(f"send {batch[0].name}")
        return super().send_sync(batch, skip_middleware=skip_middleware)


class RecordingBackfills(Backfills):
    """The real progress store, noting each write on the shared timeline."""

    def __init__(self, directory: str, timeline: list[str]) -> None:
        super().__init__(directory)
        self.timeline = timeline

    def set(self, progress: BackfillProgress) -> None:
        self.timeline.append("record progress")
        super().set(progress)


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
    """A fake GitHub serving scripted pages keyed by the cursor that asks for
    them, a real progress store, and one timeline of writes and sends."""

    pages: dict[str | None, IssuePage]
    backfills: RecordingBackfills
    client: RecordingClient
    timeline: list[str]
    requested: list[tuple[str, str | None]] = field(default_factory=list)
    ticks: list[str] = field(default_factory=list)

    def run(self, data: dict[str, Any]) -> Any:
        result = mocked.trigger(
            mod.backfill_page, inngest.Event(name=EVENT_BACKFILL_PAGE, data=data), self.client
        )
        assert result.status is mocked.Status.COMPLETED, result.error
        return result.output

    def sent(self, name: str) -> list[server_lib.Event]:
        return [event for event in self.client.sent if event.name == name]

    def fanned_out(self) -> list[int]:
        """Issue numbers handed to index-issue, in the order they were sent."""
        return [
            cast(IssueChanged, event.data)["issue"]["number"]
            for event in self.sent(EVENT_ISSUE_CHANGED)
        ]

    def progress(self) -> BackfillProgress:
        recorded = self.backfills.get(REPO)
        assert recorded is not None, "no progress recorded"
        return recorded


@pytest.fixture
def rig(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    timeline: list[str] = []
    harness = Rig(
        pages={},
        backfills=RecordingBackfills(str(tmp_path), timeline),
        client=RecordingClient(timeline),
        timeline=timeline,
    )

    def fake_list_issues(repo: str, cursor: str | None = None) -> IssuePage:
        harness.requested.append((repo, cursor))
        return harness.pages[cursor]

    def fake_now() -> str:
        """A clock that moves forward every time anyone looks at it."""
        tick = f"2026-09-25T10:00:{len(harness.ticks):02d}Z"
        harness.ticks.append(tick)
        return tick

    monkeypatch.setattr(api, "list_issues", fake_list_issues)
    monkeypatch.setattr(mod, "backfills", lambda: harness.backfills)
    monkeypatch.setattr(mod, "now", fake_now)
    yield harness


def test_fans_out_one_event_per_issue_carrying_the_issue(rig: Rig) -> None:
    rig.pages[None] = {"issues": [issue(1), issue(2)], "next": None}

    rig.run({"repo": REPO})

    assert rig.fanned_out() == [1, 2]
    assert all(event.data["repo"] == REPO for event in rig.sent(EVENT_ISSUE_CHANGED))


def test_pull_requests_are_dropped(rig: Rig) -> None:
    """/issues returns pull requests too — 4,496 of them on inngest/inngest."""
    rig.pages[None] = {
        "issues": [issue(1), issue(2, pull_request=True), issue(3)],
        "next": None,
    }

    output = rig.run({"repo": REPO})

    assert rig.fanned_out() == [1, 3]
    assert output["issues"] == 2
    assert output["pull_requests"] == 1


def test_a_page_of_only_pull_requests_still_moves_on(rig: Rig) -> None:
    """An all-PR page must not end the backfill. On inngest/inngest the first
    page is 93 pull requests and 7 issues, so whole pages of them are ordinary."""
    rig.pages[None] = {"issues": [issue(1, pull_request=True)], "next": "CURSOR-2"}

    rig.run({"repo": REPO})

    assert rig.sent(EVENT_ISSUE_CHANGED) == []
    assert len(rig.sent(EVENT_BACKFILL_PAGE)) == 1


def test_sends_itself_the_next_cursor(rig: Rig) -> None:
    rig.pages[None] = {"issues": [issue(1)], "next": "CURSOR-2"}

    rig.run({"repo": REPO})

    (following,) = rig.sent(EVENT_BACKFILL_PAGE)
    assert following.data["repo"] == REPO
    assert following.data["cursor"] == "CURSOR-2"


def test_the_last_page_ends_the_chain(rig: Rig) -> None:
    rig.pages["CURSOR-9"] = {"issues": [issue(900)], "next": None}

    output = rig.run({"repo": REPO, "cursor": "CURSOR-9", "chain": "c1"})

    assert rig.sent(EVENT_BACKFILL_PAGE) == []
    assert output["next"] is None


def test_asks_github_for_the_page_the_event_names(rig: Rig) -> None:
    rig.pages["CURSOR-5"] = {"issues": [], "next": None}

    rig.run({"repo": REPO, "cursor": "CURSOR-5", "chain": "c1"})

    assert rig.requested == [(REPO, "CURSOR-5")]


def test_the_first_page_starts_a_chain_that_later_pages_carry(rig: Rig) -> None:
    rig.pages[None] = {"issues": [], "next": "CURSOR-2"}
    rig.run({"repo": REPO})
    (second,) = rig.sent(EVENT_BACKFILL_PAGE)
    chain = second.data["chain"]
    assert chain

    rig.pages["CURSOR-2"] = {"issues": [], "next": "CURSOR-3"}
    rig.run(dict(second.data))
    third = rig.sent(EVENT_BACKFILL_PAGE)[-1]
    assert third.data["chain"] == chain


def test_a_retried_send_cannot_fork_the_chain(rig: Rig) -> None:
    """Inngest drops a second event with the same id inside 24 hours, so a retry
    that re-sends "next page" is ignored rather than starting a second chain
    that would page — and index — the rest of the repo twice."""
    rig.pages["CURSOR-2"] = {"issues": [], "next": "CURSOR-3"}

    rig.run({"repo": REPO, "cursor": "CURSOR-2", "chain": "c1"})
    rig.run({"repo": REPO, "cursor": "CURSOR-2", "chain": "c1"})
    rig.run({"repo": REPO, "cursor": "CURSOR-2", "chain": "c2"})

    first, retried, other_backfill = (event.id for event in rig.sent(EVENT_BACKFILL_PAGE))
    assert first == retried
    assert first != other_backfill


def test_each_hand_off_carries_an_id_naming_the_chain_and_the_issue(rig: Rig) -> None:
    """If a page ever runs twice — a resume after a crash — its hand-offs repeat
    with the same ids, and Inngest drops them instead of indexing every issue on
    the page a second time."""
    rig.pages["CURSOR-2"] = {"issues": [issue(1), issue(2)], "next": None}

    rig.run({"repo": REPO, "cursor": "CURSOR-2", "chain": "c1"})
    rig.run({"repo": REPO, "cursor": "CURSOR-2", "chain": "c1"})
    rig.run({"repo": REPO, "cursor": "CURSOR-2", "chain": "c2"})

    assert [event.id for event in rig.sent(EVENT_ISSUE_CHANGED)] == [
        "issue:c1:1",
        "issue:c1:2",
        "issue:c1:1",
        "issue:c1:2",
        "issue:c2:1",
        "issue:c2:2",
    ]


# Progress, for `gh-issue sync` and `status` ------------------------------------


def test_each_page_records_where_the_backfill_got_to(rig: Rig) -> None:
    rig.pages["CURSOR-2"] = {"issues": [issue(1)], "next": "CURSOR-3"}

    rig.run({"repo": REPO, "cursor": "CURSOR-2", "chain": "c1"})

    assert rig.progress() == {
        "repo": REPO,
        "chain": "c1",
        "next": "CURSOR-3",
        "finished": False,
        "at": rig.ticks[0],
    }


def test_the_last_page_records_the_backfill_as_finished(rig: Rig) -> None:
    rig.pages["CURSOR-9"] = {"issues": [issue(900)], "next": None}

    rig.run({"repo": REPO, "cursor": "CURSOR-9", "chain": "c1"})

    assert rig.progress()["finished"] is True
    assert rig.progress()["next"] is None


def test_the_first_page_records_the_chain_it_started(rig: Rig) -> None:
    rig.pages[None] = {"issues": [], "next": "CURSOR-2"}

    rig.run({"repo": REPO})

    (following,) = rig.sent(EVENT_BACKFILL_PAGE)
    assert rig.progress()["chain"] == following.data["chain"]


def test_progress_is_recorded_before_the_next_page_is_sent(rig: Rig) -> None:
    """If the run dies between the two, the record names a page nobody has
    started, and `sync` resumes there. The other way round it could name a page
    that already ran, and resuming would run it again."""
    rig.pages["CURSOR-2"] = {"issues": [issue(1)], "next": "CURSOR-3"}

    rig.run({"repo": REPO, "cursor": "CURSOR-2", "chain": "c1"})

    assert rig.timeline == [
        f"send {EVENT_ISSUE_CHANGED}",
        "record progress",
        f"send {EVENT_BACKFILL_PAGE}",
    ]


def test_progress_is_written_once_and_stamped_once_despite_replays(rig: Rig) -> None:
    """The run replays once per step, so a write or a clock read outside a step
    would happen on every pass, and `at` would be whatever the last pass saw.
    This clock ticks on every read, so that mistake would show here."""
    rig.pages["CURSOR-2"] = {"issues": [issue(1)], "next": "CURSOR-3"}

    rig.run({"repo": REPO, "cursor": "CURSOR-2", "chain": "c1"})

    assert rig.timeline.count("record progress") == 1
    assert rig.ticks == [rig.progress()["at"]]
