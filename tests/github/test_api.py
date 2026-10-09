import time
from collections.abc import Callable

import httpx
import pytest

from gh_issue.github import api
from gh_issue.github.api import next_cursor_from_link, trim_comment


def test_a_missing_token_says_what_to_do(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without this, a run fails with the bare message 'GITHUB_TOKEN' — which is
    what the dev server showed when the worker ran without a token."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match=r"github.token is not set.*gh-issue config set"):
        api.list_comments("owner/name", 1)


def answering(
    monkeypatch: pytest.MonkeyPatch, handler: Callable[[httpx.Request], httpx.Response]
) -> list[str]:
    """Route the GitHub client through a fake transport; returns the paths asked for."""
    asked: list[str] = []

    def transport(request: httpx.Request) -> httpx.Response:
        asked.append(request.url.path)
        return handler(request)

    monkeypatch.setattr(
        api,
        "_client",
        lambda: httpx.Client(base_url=api.API, transport=httpx.MockTransport(transport)),
    )
    return asked


def test_get_issue_reads_one_issue_fresh(monkeypatch: pytest.MonkeyPatch) -> None:
    raw = {"number": 412, "title": "t", "state": "closed"}
    asked = answering(monkeypatch, lambda request: httpx.Response(200, json=raw))

    assert api.get_issue("owner/name", 412) == raw
    assert asked == ["/repos/owner/name/issues/412"]


@pytest.mark.parametrize(
    "status",
    [
        pytest.param(404, id="never-there"),
        pytest.param(410, id="deleted"),
        pytest.param(301, id="transferred-to-another-repo"),
    ],
)
def test_get_issue_of_an_issue_this_repo_no_longer_has_is_none(
    monkeypatch: pytest.MonkeyPatch, status: int
) -> None:
    answering(monkeypatch, lambda request: httpx.Response(status))
    assert api.get_issue("owner/name", 412) is None


def test_get_issue_still_raises_on_real_errors(monkeypatch: pytest.MonkeyPatch) -> None:
    """A 502 is GitHub having a bad minute, not an answer about the issue."""
    answering(monkeypatch, lambda request: httpx.Response(502))
    with pytest.raises(httpx.HTTPStatusError):
        api.get_issue("owner/name", 412)


def test_repo_exists(monkeypatch: pytest.MonkeyPatch) -> None:
    answering(
        monkeypatch,
        lambda request: httpx.Response(200 if request.url.path == "/repos/a/b" else 404),
    )
    assert api.repo_exists("a/b")
    assert not api.repo_exists("a/typo")


def test_next_cursor_from_link_header() -> None:
    link = (
        "<https://api.github.com/repositories/1/issues"
        '?per_page=100&after=Y3Vyc29yOnYyOpHOAA>; rel="next"'
    )
    assert next_cursor_from_link(link) == "Y3Vyc29yOnYyOpHOAA"


def test_next_cursor_picks_the_next_link_not_the_others() -> None:
    link = (
        '<https://api.github.com/x?after=PREV>; rel="prev", '
        '<https://api.github.com/x?after=NEXT>; rel="next", '
        '<https://api.github.com/x?after=LAST>; rel="last"'
    )
    assert next_cursor_from_link(link) == "NEXT"


def test_no_next_link_means_last_page() -> None:
    assert next_cursor_from_link('<https://api.github.com/x>; rel="prev"') is None
    assert next_cursor_from_link(None) is None
    assert next_cursor_from_link("") is None


def test_next_link_without_a_cursor_returns_none() -> None:
    assert next_cursor_from_link('<https://api.github.com/x?page=2>; rel="next"') is None


def test_trim_comment_keeps_only_what_chunking_needs() -> None:
    raw = {
        "id": 9981,
        "body": "IntersectionObserver race condition.",
        "user": {"login": "maintainer", "avatar_url": "https://...", "id": 7},
        "created_at": "2026-01-02T00:00:00Z",
        "updated_at": "2026-01-02T00:00:00Z",
        "html_url": "https://github.com/owner/name/issues/412#issuecomment-9981",
        "reactions": {"total_count": 3},
    }
    assert trim_comment(raw) == {
        "id": 9981,
        "body": "IntersectionObserver race condition.",
        "user": {"login": "maintainer"},
        "created_at": "2026-01-02T00:00:00Z",
    }


def test_trim_comment_turns_a_null_body_into_an_empty_string() -> None:
    raw = {
        "id": 1,
        "body": None,
        "user": {"login": "someone"},
        "created_at": "2026-01-02T00:00:00Z",
    }
    assert trim_comment(raw)["body"] == ""


# GitHub's own search, which the web page shows next to ours ----------------------


def test_search_issues_asks_githubs_hybrid_search_for_issues_in_one_repo(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    seen: list[httpx.Request] = []

    def answer(request: httpx.Request) -> httpx.Response:
        seen.append(request)
        item = {
            "number": 433,
            "title": "Support dev server running on https",
            "state": "open",
            "html_url": "https://github.com/owner/name/issues/433",
            "body": "a long body the page never shows",
            "user": {"login": "someone"},
        }
        return httpx.Response(
            200, json={"total_count": 1, "search_type": "hybrid", "items": [item]}
        )

    asked = answering(monkeypatch, answer)

    found = api.search_issues("owner/name", "cannot start the server", limit=5)

    assert asked == ["/search/issues"]
    assert dict(seen[0].url.params) == {
        "q": "cannot start the server repo:owner/name is:issue",
        "search_type": "hybrid",
        "per_page": "5",
    }
    assert found == {
        "mode": "hybrid",
        "issues": [
            {
                "number": 433,
                "title": "Support dev server running on https",
                "state": "open",
                "url": "https://github.com/owner/name/issues/433",
            }
        ],
    }


def test_search_issues_reports_the_mode_github_ran_not_the_one_asked_for(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If GitHub ever answers with plain keyword search, the page has to say so."""
    answering(
        monkeypatch,
        lambda request: httpx.Response(200, json={"search_type": "lexical", "items": []}),
    )
    found = api.search_issues("owner/name", "x")

    assert found["mode"] == "lexical"
    assert found["issues"] == []


def test_search_issues_turns_the_spent_limit_into_seconds_to_wait(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """What GitHub sent for the tenth search inside a minute, on 2026-10-09: a
    403, nothing remaining, and the time of the reset. No retry-after."""
    headers = {
        "x-ratelimit-remaining": "0",
        "x-ratelimit-reset": str(int(time.time()) + 40),
        "x-ratelimit-resource": "semantic_search",
    }
    answering(monkeypatch, lambda request: httpx.Response(403, headers=headers))

    with pytest.raises(api.SearchLimited) as limited:
        api.search_issues("owner/name", "x")
    assert 38 <= limited.value.retry_after <= 40


def test_search_issues_still_raises_on_a_403_that_is_not_the_limit(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A token that may not see the repo is also a 403, and waiting won't fix it."""
    answering(
        monkeypatch,
        lambda request: httpx.Response(403, headers={"x-ratelimit-remaining": "7"}),
    )
    with pytest.raises(httpx.HTTPStatusError):
        api.search_issues("owner/name", "x")
