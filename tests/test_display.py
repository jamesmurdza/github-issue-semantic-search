"""What `gh-issue search` and `gh-issue status` print."""

import datetime
from typing import Literal

import pytest

from gh_issue.config import Shown
from gh_issue.display import ago, format_config, format_hits, format_status
from gh_issue.heartbeat import WorkerStatus
from gh_issue.index.backfills import BackfillProgress
from gh_issue.index.store import SearchHit
from gh_issue.query import RepoStatus, Status

NOW = datetime.datetime(2026, 9, 25, 12, 0, 0, tzinfo=datetime.UTC)


@pytest.mark.parametrize(
    ("then", "said"),
    [
        ("2026-09-25T11:59:58Z", "just now"),
        ("2026-09-25T11:59:48Z", "12 s ago"),
        ("2026-09-25T11:57:00Z", "3 min ago"),
        ("2026-09-25T10:00:00Z", "2 h ago"),
        ("2026-09-21T12:00:00Z", "4 days ago"),
    ],
)
def test_ago(then: str, said: str) -> None:
    assert ago(then, NOW) == said


# search -------------------------------------------------------------------------


def hit(
    *,
    text: str = "It was an   IntersectionObserver\n\nrace condition.",
    issue_number: int = 412,
    kind: str = "comment",
) -> SearchHit:
    return {
        "id": "owner/name#412:comment:9",
        "text": text,
        "content_hash": "h",
        "repo": "owner/name",
        "issue_number": issue_number,
        "title": "App freezes when I scroll fast",
        "state": "closed",
        "labels": ["bug"],
        "kind": kind,
        "author": "maintainer",
        "created_at": "2026-01-02T08:30:00Z",
        "score": 0.8312,
    }


def test_a_hit_names_the_issue_says_who_wrote_what_and_links_it() -> None:
    """No score: the fused score only orders one search's results, and a number
    like 1.87 would read as something it is not. The order already says it."""
    lines = format_hits([hit()]).splitlines()

    assert lines == [
        "1. owner/name#412 (closed) App freezes when I scroll fast",
        "   comment by maintainer on 2026-01-02",
        "   It was an IntersectionObserver race condition.",
        "   https://github.com/owner/name/issues/412",
    ]


def test_a_long_post_is_cut_to_a_snippet() -> None:
    (snippet,) = [
        line for line in format_hits([hit(text="word " * 200)]).splitlines() if "word" in line
    ]
    assert len(snippet) <= 3 + 240
    assert snippet.endswith("...")


def test_hits_are_numbered_and_spaced() -> None:
    text = format_hits([hit(), hit(issue_number=7)])
    assert "\n\n2. owner/name#7 " in text


def test_a_hit_on_an_issue_body_does_not_repeat_the_title() -> None:
    """The body is stored as title and body together, because that is what gets
    embedded; the first line already shows the title."""
    body = hit(kind="body", text="App freezes when I scroll fast\n\nHappens on long lists.")
    assert format_hits([body]).splitlines()[2] == "   Happens on long lists."


def test_no_hits() -> None:
    assert format_hits([]) == "No matches."


# status -------------------------------------------------------------------------


def worker(
    *,
    running: bool = True,
    connected: bool = True,
    state: str | None = "ACTIVE",
    mode: str | None = "dev",
    last_seen: str | None = "2026-09-25T11:59:52Z",
) -> WorkerStatus:
    return {
        "running": running,
        "connected": connected,
        "state": state,
        "mode": mode,
        "last_seen": last_seen,
    }


def repo(
    name: str,
    *,
    issues: int = 313,
    chunks: int = 2104,
    caught_up_to: str | None = "2026-09-25T11:48:00Z",
    backfill: BackfillProgress | Literal["done"] | None = "done",
    failed: int = 0,
) -> RepoStatus:
    if backfill == "done":
        backfill = {
            "repo": name,
            "chain": "c1",
            "next": None,
            "finished": True,
            "at": "2026-09-25T09:00:00Z",
        }
    return {
        "repo": name,
        "issues": issues,
        "chunks": chunks,
        "caught_up_to": caught_up_to,
        "backfill": backfill,
        "failed": failed,
    }


def first_line(status: Status) -> str:
    return format_status(status, NOW).splitlines()[0]


def test_a_connected_worker() -> None:
    line = first_line({"worker": worker(), "repos": []})
    assert line == "Worker: connected to the local dev server (last heartbeat 8 s ago)"


def test_a_worker_that_lost_its_connection() -> None:
    line = first_line({"worker": worker(connected=False, state="RECONNECTING"), "repos": []})
    assert "reconnecting" in line
    assert "Inngest" in line


def test_a_worker_that_stopped_answering() -> None:
    line = first_line(
        {
            "worker": worker(running=False, connected=False, last_seen="2026-09-25T09:00:00Z"),
            "repos": [],
        }
    )
    assert "not running" in line
    assert "3 h ago" in line
    assert "gh-issue worker" in line


def test_no_worker_ever() -> None:
    never = worker(running=False, connected=False, state=None, mode=None, last_seen=None)
    line = first_line({"worker": never, "repos": []})
    assert line == "Worker: not running. Start one with: gh-issue worker"


def test_nothing_synced_says_how_to_start() -> None:
    text = format_status({"worker": worker(), "repos": []}, NOW)
    assert "gh-issue sync owner/name" in text


def test_a_row_per_repo_with_its_backfill_and_sweep() -> None:
    paging = repo(
        "some/repo",
        issues=1,
        chunks=1,
        caught_up_to=None,
        backfill={
            "repo": "some/repo",
            "chain": "c2",
            "next": "CURSOR-4",
            "finished": False,
            "at": "2026-09-25T11:57:00Z",
        },
    )
    rows = format_status({"worker": worker(), "repos": [repo("inngest/inngest"), paging]}, NOW)
    table = rows.splitlines()[2:5]

    assert table[0].split() == ["REPO", "ISSUES", "CHUNKS", "BACKFILL", "LAST", "SWEEP", "FAILED"]
    assert table[1].split() == ["inngest/inngest", "313", "2,104", "done", "12", "min", "ago", "0"]
    assert "paging, last page 3 min ago" in table[2]
    assert "never" in table[2]


def test_a_backfill_waiting_for_its_first_page() -> None:
    queued = repo(
        "some/repo",
        backfill={
            "repo": "some/repo",
            "chain": "c2",
            "next": None,
            "finished": False,
            "at": "2026-09-25T11:59:00Z",
        },
    )
    text = format_status({"worker": worker(), "repos": [queued]}, NOW)
    assert "queued 1 min ago" in text


def test_failures_are_counted_and_the_fix_named() -> None:
    text = format_status({"worker": worker(), "repos": [repo("a/b", failed=2)]}, NOW)
    assert text.splitlines()[-1] == "2 issue(s) failed every retry; `gh-issue sync` retries them."


# config ------------------------------------------------------------------------


def test_config_lists_each_setting_its_value_where_that_came_from_and_what_it_is() -> None:
    rows: list[Shown] = [
        {"key": "github.token", "value": "****1234", "source": "config file"},
        {"key": "inngest.event_key", "value": None, "source": None},
        {"key": "qdrant.url", "value": "http://q:6333", "source": "environment (QDRANT_URL)"},
    ]

    assert format_config("/home/me/.gh-issue/config.toml", rows) == (
        "Settings, from /home/me/.gh-issue/config.toml and the environment (which wins):\n"
        "\n"
        "  github.token        ****1234       config file\n"
        "      A GitHub token; for public repos, the public_repo scope is enough.\n"
        "  inngest.event_key   not set\n"
        "      Inngest Cloud's event key.\n"
        "  qdrant.url          http://q:6333  environment (QDRANT_URL)\n"
        "      Where Qdrant is.\n"
        "\n"
        "Change one: gh-issue config set <setting> [value]"
    )
