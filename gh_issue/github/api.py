"""Thin GitHub REST client.

Everything here is called from inside an Inngest step, so each function is one
network round trip with no retry logic of its own — Inngest owns the retries.
"""

import re
import time
from collections.abc import Mapping
from typing import Any, TypedDict
from urllib.parse import parse_qs, urlparse

import httpx

from gh_issue.env import github_token
from gh_issue.github.types import Comment, Issue

API = "https://api.github.com"
PER_PAGE = 100
# The mode GitHub's own issue search is asked for: semantic and keyword together.
SEARCH_MODE = "hybrid"

_NEXT_LINK = re.compile(r'<([^>]+)>;\s*rel="next"')


class IssuePage(TypedDict):
    """One page of issues plus the cursor that fetches the page after it."""

    issues: list[Issue]
    next: str | None


class FoundIssue(TypedDict):
    number: int
    title: str
    state: str
    url: str


class IssueSearch(TypedDict):
    # What GitHub says it ran, which need not be what was asked for.
    mode: str
    issues: list[FoundIssue]


class SearchLimited(Exception):
    """GitHub's allowance of semantic searches is spent; `retry_after` is the
    seconds until it is refilled."""

    def __init__(self, retry_after: int) -> None:
        super().__init__(f"GitHub's search limit is spent for {retry_after} more seconds")
        self.retry_after = retry_after


def next_cursor_from_link(link_header: str | None) -> str | None:
    """Pull the `after` cursor out of GitHub's Link header.

    The issues endpoint paginates with an opaque cursor rather than a page
    number, and it only appears in the header, never in the body.
    """
    if not link_header:
        return None
    match = _NEXT_LINK.search(link_header)
    if match is None:
        return None
    after = parse_qs(urlparse(match.group(1)).query).get("after")
    return after[0] if after else None


def trim_comment(comment: Mapping[str, Any]) -> Comment:
    """Keep only the fields chunking reads.

    A raw comment object is roughly 1.5 KB of JSON, most of it URLs we never
    look at. This is the return value of a step, and a step's return value is
    capped at 4 MiB, so the trim is what lets a very long thread through.
    """
    return {
        "id": comment["id"],
        "body": comment.get("body") or "",
        "user": {"login": comment["user"]["login"]},
        "created_at": comment["created_at"],
    }


MISSING_TOKEN = "github.token is not set. Set it with: gh-issue config set github.token"


def _client() -> httpx.Client:
    token = github_token()
    if not token:
        raise RuntimeError(MISSING_TOKEN)
    return httpx.Client(
        base_url=API,
        timeout=30.0,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
        },
    )


def list_issues(repo: str, cursor: str | None = None, since: str | None = None) -> IssuePage:
    """One page of issues, newest state included, plus the next cursor.

    `since` is what the hourly sweep uses: everything updated after that moment.
    Note that this endpoint also returns pull requests — they carry a
    `pull_request` field and the caller drops them.
    """
    params: dict[str, str | int] = {"per_page": PER_PAGE, "state": "all", "direction": "asc"}
    if cursor:
        params["after"] = cursor
    if since:
        params["since"] = since

    with _client() as http:
        response = http.get(f"/repos/{repo}/issues", params=params)
        response.raise_for_status()
        issues: list[Issue] = response.json()
        return {"issues": issues, "next": next_cursor_from_link(response.headers.get("link"))}


def list_comments(repo: str, number: int) -> list[Comment]:
    """Every comment on one issue, trimmed.

    Comments page at 100, so a long thread costs several calls. This is the one
    call the indexing run makes, and the one that spends rate limit.
    """
    comments: list[Comment] = []
    page = 1
    with _client() as http:
        while True:
            response = http.get(
                f"/repos/{repo}/issues/{number}/comments",
                params={"per_page": PER_PAGE, "page": page},
            )
            response.raise_for_status()
            batch: list[dict[str, Any]] = response.json()
            comments.extend(trim_comment(comment) for comment in batch)
            if len(batch) < PER_PAGE:
                return comments
            page += 1


def get_issue(repo: str, number: int) -> Issue | None:
    """One issue as it is on GitHub now, or None if this repo no longer has it.

    404 is an issue that was never there, 410 one that was deleted, and 301 one
    transferred to another repo; for this repo's index they all mean gone.
    """
    with _client() as http:
        response = http.get(f"/repos/{repo}/issues/{number}")
        if response.status_code in (301, 404, 410):
            return None
        response.raise_for_status()
        issue: Issue = response.json()
        return issue


def repo_exists(repo: str) -> bool:
    """Whether the token can see this repo, checked before starting a backfill so
    a typo fails here rather than in a run that retries for minutes."""
    with _client() as http:
        response = http.get(f"/repos/{repo}")
        if response.status_code == 404:
            return False
        response.raise_for_status()
        return True


def search_issues(repo: str, query: str, limit: int = 5) -> IssueSearch:
    """GitHub's own issue search for `query`, in one repo.

    Hybrid is its strongest mode: the semantic index and the keyword search
    together. GitHub allows a user 10 of these a minute and answers 403, with
    the time of the reset, once they are spent.
    """
    params: dict[str, str | int] = {
        "q": f"{query} repo:{repo} is:issue",
        "search_type": SEARCH_MODE,
        "per_page": limit,
    }
    with _client() as http:
        response = http.get("/search/issues", params=params)
        spent = response.headers.get("x-ratelimit-remaining") == "0"
        if response.status_code in (403, 429) and spent:
            reset = int(response.headers.get("x-ratelimit-reset", "0"))
            raise SearchLimited(max(1, reset - int(time.time())))
        response.raise_for_status()
        found: dict[str, Any] = response.json()
        return {
            "mode": found.get("search_type", SEARCH_MODE),
            "issues": [
                {
                    "number": item["number"],
                    "title": item["title"],
                    "state": item["state"],
                    "url": item["html_url"],
                }
                for item in found["items"]
            ],
        }


def register_webhook(repo: str, url: str, secret: str) -> dict[str, Any]:
    """Point a repo's issue events at an Inngest webhook URL.

    Nothing of ours is in the request path: GitHub posts to Inngest, a transform
    there shapes the payload into an event, and the indexing function picks it up.
    """
    with _client() as http:
        response = http.post(
            f"/repos/{repo}/hooks",
            json={
                "name": "web",
                "active": True,
                "events": ["issues", "issue_comment"],
                "config": {"url": url, "content_type": "json", "secret": secret},
            },
        )
        response.raise_for_status()
        hook: dict[str, Any] = response.json()
        return hook
