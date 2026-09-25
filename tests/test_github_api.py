import pytest

from gh_issue.github import api
from gh_issue.github.api import next_cursor_from_link, trim_comment


def test_a_missing_token_says_what_to_do(monkeypatch: pytest.MonkeyPatch) -> None:
    """Without this, a run fails with the bare message 'GITHUB_TOKEN' — which is
    what the dev server showed when the worker ran without a .env."""
    monkeypatch.delenv("GITHUB_TOKEN", raising=False)
    with pytest.raises(RuntimeError, match=r"GITHUB_TOKEN is not set.*\.env"):
        api.list_comments("owner/name", 1)


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
