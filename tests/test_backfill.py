"""Tests for the backfill.

Each run handles one page and sends itself the next, so what matters is exactly
what each run sends: which issues fan out, and whether — and how — the chain
carries on. The mocked client discards sent events, so these record them.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
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
from gh_issue.settings import EVENT_BACKFILL_PAGE, EVENT_ISSUE_CHANGED

REPO = "owner/name"


class RecordingClient(mocked.Inngest):
    """The mocked client, except it keeps what it was asked to send."""

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
    """A fake GitHub serving scripted pages, keyed by the cursor that asks for them."""

    pages: dict[str | None, IssuePage]
    requested: list[tuple[str, str | None]] = field(default_factory=list)
    client: RecordingClient = field(default_factory=RecordingClient)

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


@pytest.fixture
def rig(monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    harness = Rig(pages={})

    def fake_list_issues(repo: str, cursor: str | None = None) -> IssuePage:
        harness.requested.append((repo, cursor))
        return harness.pages[cursor]

    monkeypatch.setattr(api, "list_issues", fake_list_issues)
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
