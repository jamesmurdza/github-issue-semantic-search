"""The index next to GitHub's own issue search, as a web page.

One page, one question, two lists, both for the repo chosen on the page from
the ones in the index. It is there to show one thing: a question whose answer
sits in a comment.

The page asks this server for both lists, GitHub's too, so the GitHub token
stays on this side. The endpoints are plain functions, which Starlette runs in
its thread pool: a slow answer from GitHub does not hold up ours.
"""

from pathlib import Path
from typing import TypedDict

import httpx
from starlette.applications import Starlette
from starlette.requests import Request
from starlette.responses import HTMLResponse, JSONResponse, Response
from starlette.routing import Route

from gh_issue.env import github_token
from gh_issue.github import api
from gh_issue.index import query as reads
from gh_issue.index import storage
from gh_issue.index.store import SearchHit
from gh_issue.interface.web.plain import excerpt

# How many issues each list shows.
LIMIT = 5
# About three lines of the page's result text.
SNIPPET = 280

NO_QUESTION = "Type a question first."
NO_REPO = "Choose a repo first."

PAGE = Path(__file__).with_name("index.html")


class OurResult(TypedDict):
    number: int
    title: str
    state: str
    kind: str
    author: str
    date: str
    text: str
    url: str


def _url(hit: SearchHit) -> str:
    """The post itself: the comment when the match is one, else the issue."""
    issue = f"https://github.com/{hit['repo']}/issues/{hit['issue_number']}"
    if hit["kind"] != "comment":
        return issue
    # A comment's chunk id ends in the id GitHub gave it: owner/name#12:comment:345.
    return f"{issue}#issuecomment-{hit['id'].rsplit(':', 1)[1]}"


def _result(hit: SearchHit) -> OurResult:
    text = hit["text"]
    # A body is stored as title and body together, which is what gets embedded.
    if hit["kind"] == "body":
        text = text.removeprefix(f"{hit['title']}\n\n")
    return {
        "number": hit["issue_number"],
        "title": hit["title"],
        "state": hit["state"],
        "kind": hit["kind"],
        "author": hit["author"],
        "date": hit["created_at"][:10],
        "text": excerpt(text, SNIPPET),
        "url": _url(hit),
    }


def _error(status: int, message: str, **more: object) -> JSONResponse:
    """Every failure is one sentence a person can act on."""
    return JSONResponse({"error": message, **more}, status_code=status)


def _asked(request: Request) -> tuple[str, str] | JSONResponse:
    """The question and the repo to ask it of, or what is wrong with them.

    Whatever is sent as the repo ends up inside the query GitHub is sent, so
    only a repo that is in the index, which is what the page offers, gets through.
    """
    question = request.query_params.get("q", "").strip()
    if not question:
        return _error(400, NO_QUESTION)
    repo = request.query_params.get("repo", "").strip()
    if not repo:
        return _error(400, NO_REPO)
    if problem := storage.index_problem():
        return _error(503, problem)
    if repo not in reads.indexed_repos():
        return _error(400, f"{repo} is not synced.")
    return question, repo


def make_app() -> Starlette:
    page = PAGE.read_text(encoding="utf-8")

    def index(request: Request) -> Response:
        return HTMLResponse(page)

    def repos(request: Request) -> Response:
        if problem := storage.index_problem():
            return _error(503, problem)
        return JSONResponse({"repos": reads.indexed_repos()})

    def ours(request: Request) -> Response:
        asked = _asked(request)
        if isinstance(asked, JSONResponse):
            return asked
        question, repo = asked
        hits = reads.search(question, reads.make_filters(repos=[repo]), LIMIT)
        return JSONResponse({"results": [_result(hit) for hit in hits]})

    def github(request: Request) -> Response:
        asked = _asked(request)
        if isinstance(asked, JSONResponse):
            return asked
        question, repo = asked
        if not github_token():
            return _error(503, api.MISSING_TOKEN)
        try:
            found = api.search_issues(repo, question, LIMIT)
        except api.SearchLimited as limited:
            wait = limited.retry_after
            return _error(
                429,
                "GitHub allows 10 of these searches a minute. "
                f"Try again in {wait} second{'' if wait == 1 else 's'}.",
                retry_after=wait,
            )
        except httpx.HTTPStatusError as error:
            return _error(502, f"GitHub's search failed (HTTP {error.response.status_code}).")
        except httpx.HTTPError:
            return _error(502, "Can't reach GitHub.")
        return JSONResponse(
            {"mode": found["mode"], "total": found["total"], "results": found["issues"]}
        )

    return Starlette(
        routes=[
            Route("/", index),
            Route("/api/repos", repos),
            Route("/api/ours", ours),
            Route("/api/github", github),
        ]
    )
