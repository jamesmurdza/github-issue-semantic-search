"""A small index for the web tests: a thread whose answer is in a comment, an
unrelated thread, a thread with a long body, and a thread in another repo that
must never show up."""

from collections.abc import Iterator
from pathlib import Path

import pytest
from starlette.testclient import TestClient

from gh_issue.github.types import Comment, Issue
from gh_issue.index import query, storage
from gh_issue.index.chunking import chunks_for
from gh_issue.index.embed import DIM, embed_chunks
from gh_issue.index.store import Store
from gh_issue.interface.web.app import make_app

REPO = "owner/name"


def issue(number: int, title: str, body: str, *, state: str = "open") -> Issue:
    return {
        "number": number,
        "title": title,
        "body": body,
        "state": state,
        "labels": [],
        "user": {"login": "reporter"},
        "created_at": "2026-01-01T00:00:00Z",
        "comments": 0,
    }


FREEZE = issue(412, "App freezes when I scroll fast", "Happens on long lists.")
ANSWER: Comment = {
    "id": 9981,
    "body": "Downgrading to 1.3.3 fixed it for me. The newer build cannot find the "
    "sqlite migrations folder when the server starts.",
    "user": {"login": "maintainer"},
    "created_at": "2026-01-02T00:00:00Z",
}
LOGIN = issue(
    7, "Login fails after the token expires", "Getting a 401 every morning.", state="closed"
)
CURRENCY = issue(
    55,
    "Invoices show the wrong currency",
    "The invoice total is shown in dollars instead of euros. " * 20,
)
ELSEWHERE = issue(
    99,
    "Server does not start: the sqlite migrations folder is missing",
    "After upgrading, the server fails to start.",
)


@pytest.fixture
def indexed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Store]:
    store = Store(dim=DIM, path=str(tmp_path / "qdrant"))
    store.upsert(embed_chunks(chunks_for(REPO, FREEZE, [ANSWER])))
    store.upsert(embed_chunks(chunks_for(REPO, LOGIN, [])))
    store.upsert(embed_chunks(chunks_for(REPO, CURRENCY, [])))
    store.upsert(embed_chunks(chunks_for("other/repo", ELSEWHERE, [])))
    monkeypatch.setattr(query, "store", lambda: store)
    monkeypatch.setattr(storage, "index_problem", lambda: None)
    yield store
    store.close()


@pytest.fixture
def client(indexed: Store) -> TestClient:
    return TestClient(make_app(REPO))
