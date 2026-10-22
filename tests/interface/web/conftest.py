"""A small index for the web tests: a thread whose answer is in a comment, an
unrelated thread, a thread with a long body, and two threads in another repo
that must never show up."""

from collections.abc import Callable, Iterator
from pathlib import Path

import httpx
import pytest
from starlette.testclient import TestClient

from gh_issue.github import api
from gh_issue.github.types import Comment, Issue
from gh_issue.index import query, storage
from gh_issue.index.chunking import chunks_for
from gh_issue.index.embed import DIM, embed_chunks
from gh_issue.index.store import Store
from gh_issue.interface.web.app import make_app

REPO = "owner/name"

Handler = Callable[[httpx.Request], httpx.Response]
GitHubAnswers = Callable[[Handler], list[httpx.Request]]


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
# Written the way issues are: a template's comment, a heading, bold, a quote,
# a link, and code both in a line and in a block.
WEBHOOK = issue(
    100,
    "Webhook signature check fails",
    "<!-- Thanks for filing an issue! -->\n"
    "## Describe the bug\n\n"
    "The **webhook** signature check fails with `InvalidSignature` after I rotate "
    "the signing key.\n\n"
    "> I followed the [rotation guide](https://example.com/rotate).\n\n"
    "```bash\ncurl -X POST localhost:8288/hook\n```\n",
)


def github_item(
    number: int, title: str, *, state: str = "open", repo: str = REPO
) -> dict[str, object]:
    """One issue as GitHub's search sends it, with the fields the server reads."""
    return {
        "number": number,
        "title": title,
        "state": state,
        "html_url": f"https://github.com/{repo}/issues/{number}",
        "user": {"login": "someone"},
        "created_at": "2026-03-03T09:30:00Z",
        "comments": 4,
    }


@pytest.fixture
def indexed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Store]:
    store = Store(dim=DIM, path=str(tmp_path / "qdrant"))
    store.upsert(embed_chunks(chunks_for(REPO, FREEZE, [ANSWER])))
    store.upsert(embed_chunks(chunks_for(REPO, LOGIN, [])))
    store.upsert(embed_chunks(chunks_for(REPO, CURRENCY, [])))
    store.upsert(embed_chunks(chunks_for("other/repo", ELSEWHERE, [])))
    store.upsert(embed_chunks(chunks_for("other/repo", WEBHOOK, [])))
    monkeypatch.setattr(query, "store", lambda: store)
    monkeypatch.setattr(storage, "index_problem", lambda: None)
    yield store
    store.close()


@pytest.fixture
def client(indexed: Store) -> TestClient:
    return TestClient(make_app())


@pytest.fixture
def github_answers(monkeypatch: pytest.MonkeyPatch) -> GitHubAnswers:
    """Give the server a token, and route its GitHub client through a fake
    transport. Call it with what GitHub should answer; it returns the requests
    the server went on to make."""

    def answer_with(handler: Handler) -> list[httpx.Request]:
        made: list[httpx.Request] = []

        def transport(request: httpx.Request) -> httpx.Response:
            made.append(request)
            return handler(request)

        monkeypatch.setenv("GITHUB_TOKEN", "a-token-for-tests")
        monkeypatch.setattr(
            api,
            "_client",
            lambda: httpx.Client(base_url=api.API, transport=httpx.MockTransport(transport)),
        )
        return made

    return answer_with
