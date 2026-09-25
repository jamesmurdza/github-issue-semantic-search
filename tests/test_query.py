"""Tests for the read side the CLI and the MCP server share.

Real embedded Qdrant and the real embedding model: the point of `search` is that
it finds things by meaning, and only the real model can show that.
"""

from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest

from gh_issue import query
from gh_issue.chunking import chunks_for
from gh_issue.embed import DIM, embed_chunks
from gh_issue.github.types import Comment, Issue
from gh_issue.heartbeat import WorkerStatus
from gh_issue.index.backfills import BackfillProgress, Backfills
from gh_issue.index.cursors import Cursors
from gh_issue.index.failures import Failures
from gh_issue.index.store import Store

REPO = "owner/name"


def issue(number: int, title: str, body: str, *, state: str = "open") -> Issue:
    return {
        "number": number,
        "title": title,
        "body": body,
        "state": state,
        "labels": [{"name": "bug"}],
        "user": {"login": "reporter"},
        "created_at": "2026-01-01T00:00:00Z",
        "comments": 0,
    }


def comment(comment_id: int, body: str, created_at: str) -> Comment:
    return {
        "id": comment_id,
        "body": body,
        "user": {"login": "maintainer"},
        "created_at": created_at,
    }


FREEZE = issue(412, "App freezes when I scroll fast", "Happens on long lists.")
LOGIN = issue(7, "Login fails after the token expires", "Getting a 401 every morning.")
TYPO = issue(9, "Typo in the README", "It says 'teh' twice.", state="closed")


WORKER_UP: WorkerStatus = {
    "running": True,
    "connected": True,
    "state": "ACTIVE",
    "mode": "dev",
    "last_seen": "2026-09-25T10:06:00Z",
}


@dataclass
class Index:
    store: Store
    cursors: Cursors
    backfills: Backfills
    failures: Failures

    def add(self, item: Issue, comments: list[Comment] | None = None, repo: str = REPO) -> None:
        self.store.upsert(embed_chunks(chunks_for(repo, item, comments or [])))


@pytest.fixture
def index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Index]:
    opened = Index(
        store=Store(dim=DIM, path=str(tmp_path / "qdrant")),
        cursors=Cursors(str(tmp_path / "cursors")),
        backfills=Backfills(str(tmp_path / "backfills")),
        failures=Failures(str(tmp_path / "failed")),
    )
    monkeypatch.setattr(query, "store", lambda: opened.store)
    monkeypatch.setattr(query, "cursors", lambda: opened.cursors)
    monkeypatch.setattr(query, "backfills", lambda: opened.backfills)
    monkeypatch.setattr(query, "failures", lambda: opened.failures)
    monkeypatch.setattr(query, "worker_status", lambda: WORKER_UP)
    yield opened
    opened.store.close()


# search -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("words", "expected"),
    [
        ("the page locks up while I swipe through a big feed", 412),
        ("can't sign in, session keeps expiring", 7),
    ],
)
def test_search_finds_an_issue_by_meaning_not_wording(
    index: Index, words: str, expected: int
) -> None:
    """Each query shares no word, not even a stem, with the issue it should find."""
    for item in (FREEZE, LOGIN, TYPO):
        index.add(item)

    hits = query.search(words)

    assert hits[0]["issue_number"] == expected


def test_search_applies_filters(index: Index) -> None:
    for item in (FREEZE, LOGIN, TYPO):
        index.add(item)

    hits = query.search("the page locks up while I swipe through a big feed", {"state": "closed"})

    assert [hit["issue_number"] for hit in hits] == [9]


def test_search_respects_the_limit(index: Index) -> None:
    for item in (FREEZE, LOGIN, TYPO):
        index.add(item)

    assert len(query.search("anything at all", limit=2)) == 2


def test_search_of_an_empty_index_returns_nothing(index: Index) -> None:
    assert query.search("anything") == []


def test_an_empty_query_is_refused(index: Index) -> None:
    """An empty string still embeds to a vector, and the nearest chunks to it
    are noise. Better to say so than to return them."""
    with pytest.raises(ValueError, match="empty"):
        query.search("   ")


# get_issue --------------------------------------------------------------------


def test_get_issue_returns_the_thread_body_first_then_comments_in_order(index: Index) -> None:
    later = comment(2, "Fixed in 1.4.", "2026-01-03T00:00:00Z")
    earlier = comment(1, "IntersectionObserver race condition.", "2026-01-02T00:00:00Z")
    index.add({**FREEZE, "comments": 2}, [later, earlier])

    thread = query.get_issue(REPO, 412)

    assert thread is not None
    assert thread["title"] == "App freezes when I scroll fast"
    assert thread["state"] == "open"
    assert thread["labels"] == ["bug"]
    assert thread["url"] == "https://github.com/owner/name/issues/412"
    assert [(post["kind"], post["author"], post["text"]) for post in thread["posts"]] == [
        ("body", "reporter", "Happens on long lists."),
        ("comment", "maintainer", "IntersectionObserver race condition."),
        ("comment", "maintainer", "Fixed in 1.4."),
    ]


def test_get_issue_of_an_unindexed_issue_is_none(index: Index) -> None:
    assert query.get_issue(REPO, 412) is None


def test_get_issue_does_not_mix_up_repos(index: Index) -> None:
    index.add(FREEZE)
    index.add({**LOGIN, "number": 412}, repo="other/repo")

    thread = query.get_issue("other/repo", 412)

    assert thread is not None
    assert thread["title"] == "Login fails after the token expires"


# status -----------------------------------------------------------------------


def test_status_lists_every_indexed_or_tracked_repo(index: Index) -> None:
    """A repo can be tracked with nothing indexed yet (its backfill just
    started), or indexed with no sweep cursor. Both belong in the list."""
    index.add({**FREEZE, "comments": 1}, [comment(1, "same here", "2026-01-02T00:00:00Z")])
    index.add(LOGIN)
    index.cursors.set(REPO, "2026-09-25T10:00:00Z")
    index.cursors.set("queued/repo", "2026-09-25T10:05:00Z")
    index.add(TYPO, repo="untracked/repo")

    repos = query.status()["repos"]

    assert [(r["repo"], r["issues"], r["chunks"], r["caught_up_to"]) for r in repos] == [
        (REPO, 2, 3, "2026-09-25T10:00:00Z"),
        ("queued/repo", 0, 0, "2026-09-25T10:05:00Z"),
        ("untracked/repo", 1, 1, None),
    ]


def test_status_carries_each_repos_backfill_and_failures(index: Index) -> None:
    progress: BackfillProgress = {
        "repo": REPO,
        "chain": "c1",
        "next": "CURSOR-4",
        "finished": False,
        "at": "2026-09-25T10:04:00Z",
    }
    index.backfills.set(progress)
    for number in (412, 418):
        index.failures.add(
            {"repo": REPO, "number": number, "error": "boom", "run_id": "r", "at": "t"}
        )

    (repo,) = query.status()["repos"]

    assert repo["backfill"] == progress
    assert repo["failed"] == 2


def test_status_says_whether_a_worker_is_up(index: Index) -> None:
    assert query.status()["worker"] == WORKER_UP


def test_status_of_nothing_is_empty(index: Index) -> None:
    assert query.status()["repos"] == []


# make_filters -----------------------------------------------------------------


def test_filters_leave_out_whatever_was_not_given() -> None:
    assert query.make_filters() == {}
    assert query.make_filters(repos=[], labels=[]) == {}


def test_filters_carry_every_option_given() -> None:
    assert query.make_filters(
        repos=["a/b"],
        state="closed",
        labels=["bug", "p1"],
        author="ann",
        kind="comment",
        since="2026-01-31",
    ) == {
        "repos": ["a/b"],
        "state": "closed",
        "labels": ["bug", "p1"],
        "author": "ann",
        "kind": "comment",
        "since": "2026-01-31",
    }
