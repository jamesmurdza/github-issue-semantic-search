"""The page's server, through Starlette's test client, over a real embedded
Qdrant and the real embedding model."""

import time
from collections.abc import Callable

import httpx
import pytest
from starlette.testclient import TestClient

from gh_issue.github import api
from gh_issue.index import storage

MIGRATIONS = (
    "after upgrading the server fails to start because the database migration files are missing"
)


def ours(client: TestClient, question: str) -> list[dict[str, object]]:
    response = client.get("/api/ours", params={"q": question})
    assert response.status_code == 200, response.text
    results: list[dict[str, object]] = response.json()["results"]
    return results


# The page ---------------------------------------------------------------------------


def test_the_page_names_the_repo_it_searches(client: TestClient) -> None:
    response = client.get("/")

    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert "Searching <strong>owner/name</strong>" in response.text
    assert "{{repo}}" not in response.text


# Our list ---------------------------------------------------------------------------


def test_a_question_finds_the_comment_that_answers_it(client: TestClient) -> None:
    """The wording differs from the comment's, and the answer is in neither the
    title nor the body. The link goes to the comment, not just the issue."""
    assert ours(client, MIGRATIONS)[0] == {
        "number": 412,
        "title": "App freezes when I scroll fast",
        "state": "open",
        "kind": "comment",
        "author": "maintainer",
        "date": "2026-01-02",
        "text": "Downgrading to 1.3.3 fixed it for me. The newer build cannot find the "
        "sqlite migrations folder when the server starts.",
        "url": "https://github.com/owner/name/issues/412#issuecomment-9981",
    }


def test_only_the_repo_the_server_was_started_with_is_searched(client: TestClient) -> None:
    """Issue 99, in another repo, says it in the title and would win otherwise."""
    assert 99 not in [result["number"] for result in ours(client, MIGRATIONS)]


def test_a_match_in_the_body_comes_without_the_title_and_links_to_the_issue(
    client: TestClient,
) -> None:
    """A body is stored as title and body together; the page shows the title
    on its own line already."""
    first = ours(client, "the page locks up when I swipe through a feed")[0]

    assert first["kind"] == "body"
    assert first["text"] == "Happens on long lists."
    assert first["url"] == "https://github.com/owner/name/issues/412"


def test_a_long_post_is_cut(client: TestClient) -> None:
    first = ours(client, "the invoice shows dollars, not euros")[0]

    assert first["number"] == 55
    text = str(first["text"])
    assert len(text) <= 280
    assert text.endswith("...")


# GitHub's list ----------------------------------------------------------------------


def github_answers(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> list[httpx.Request]:
    """Give the server a token, and route its GitHub client through a fake
    transport; returns the requests it made."""
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


def test_githubs_list_is_githubs_results_and_the_mode_it_ran(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    item = {
        "number": 433,
        "title": "Support dev server running on https",
        "state": "open",
        "html_url": "https://github.com/owner/name/issues/433",
    }
    made = github_answers(
        monkeypatch,
        lambda request: httpx.Response(200, json={"search_type": "hybrid", "items": [item]}),
    )

    response = client.get("/api/github", params={"q": MIGRATIONS})

    assert response.status_code == 200
    assert response.json() == {
        "mode": "hybrid",
        "results": [
            {
                "number": 433,
                "title": "Support dev server running on https",
                "state": "open",
                "url": "https://github.com/owner/name/issues/433",
            }
        ],
    }
    assert made[0].url.params["q"] == f"{MIGRATIONS} repo:owner/name is:issue"
    assert made[0].url.params["per_page"] == "5"


# What can go wrong, each as a sentence -------------------------------------------


@pytest.mark.parametrize("path", ["/api/ours", "/api/github"])
@pytest.mark.parametrize("query_string", ["", "?q=", "?q=%20%20"])
def test_an_empty_question_is_refused(client: TestClient, path: str, query_string: str) -> None:
    response = client.get(path + query_string)

    assert response.status_code == 400
    assert response.json() == {"error": "Type a question first."}


def test_our_list_says_what_is_wrong_with_the_index(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    problem = "Can't connect to Qdrant at http://localhost:6333 (qdrant.url)."
    monkeypatch.setattr(storage, "index_problem", lambda: problem)

    response = client.get("/api/ours", params={"q": "x"})

    assert response.status_code == 503
    assert response.json() == {"error": problem}


def test_githubs_list_without_a_token_says_how_to_set_one(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)

    response = client.get("/api/github", params={"q": "x"})

    assert response.status_code == 503
    assert response.json() == {
        "error": "github.token is not set. Set it with: gh-issue config set github.token"
    }


def test_githubs_list_says_how_long_to_wait_when_the_limit_is_spent(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    headers = {"x-ratelimit-remaining": "0", "x-ratelimit-reset": str(int(time.time()) + 40)}
    github_answers(monkeypatch, lambda request: httpx.Response(403, headers=headers))

    response = client.get("/api/github", params={"q": "x"})

    assert response.status_code == 429
    body = response.json()
    assert 38 <= body["retry_after"] <= 40
    assert body["error"] == (
        f"GitHub allows 10 of these searches a minute. Try again in {body['retry_after']} seconds."
    )


def test_githubs_list_says_when_github_failed(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    github_answers(monkeypatch, lambda request: httpx.Response(500))

    response = client.get("/api/github", params={"q": "x"})

    assert response.status_code == 502
    assert response.json() == {"error": "GitHub's search failed (HTTP 500)."}


def test_githubs_list_says_when_github_cannot_be_reached(
    client: TestClient, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreachable(request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("no route", request=request)

    github_answers(monkeypatch, unreachable)

    response = client.get("/api/github", params={"q": "x"})

    assert response.status_code == 502
    assert response.json() == {"error": "Can't reach GitHub."}
