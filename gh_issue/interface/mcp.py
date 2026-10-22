"""The index as MCP tools, for a coding agent.

Three read-only tools over the same query layer the CLI uses: `search_issues`
to find threads by meaning and by wording, `get_issue` to read one in full, and `status` to see
what is synced. Syncing stays a human action on the command line; an agent reads.

The descriptions below are what the agent sees when deciding what to call and
how, so they say when to use each tool, not only what it does.
"""

from typing import Annotated, Literal

from fastmcp import FastMCP
from fastmcp.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from gh_issue.index import query as reads
from gh_issue.index import storage
from gh_issue.index.query import Status, Thread
from gh_issue.index.store import SearchHit

server = FastMCP(
    name="gh-issue",
    instructions=(
        "Search the issue history of GitHub repos synced with gh-issue: bug reports, "
        "the discussion under them, and how they were resolved. Start with "
        "search_issues in the user's own words, or with an exact error message; open "
        "a promising hit with get_issue "
        "to read the whole thread. status lists which repos are synced."
    ),
)

READ_ONLY = ToolAnnotations(read_only_hint=True, open_world_hint=False)


def _check_index() -> None:
    if problem := storage.index_problem():
        raise ToolError(problem)


@server.tool(annotations=READ_ONLY)
def search_issues(
    query: Annotated[
        str,
        Field(
            description="What to look for, in plain words. Describe the symptom or "
            "question; exact keywords are not needed."
        ),
    ],
    repos: Annotated[
        list[str] | None,
        Field(description="Only these repos, as owner/name. Leave out for every synced repo."),
    ] = None,
    state: Annotated[
        Literal["open", "closed"] | None,
        Field(description="Only open or only closed issues."),
    ] = None,
    labels: Annotated[
        list[str] | None,
        Field(description="Only issues carrying all of these labels."),
    ] = None,
    author: Annotated[
        str | None, Field(description="Only posts written by this GitHub user.")
    ] = None,
    kind: Annotated[
        Literal["body", "comment"] | None,
        Field(description="Only issue bodies, or only comments."),
    ] = None,
    since: Annotated[
        str | None,
        Field(description="Only posts written on or after this date, e.g. 2026-01-31."),
    ] = None,
    limit: Annotated[int, Field(ge=1, le=50, description="How many results.")] = 10,
) -> list[SearchHit]:
    """Search GitHub issues and their comments, by meaning and by exact wording.

    Use this before answering anything about a bug, an error message or odd
    behaviour in a synced repo: someone has often reported it, and the fix or
    workaround is usually in a comment rather than the issue itself. Describe the
    problem in plain words, or paste an error message, name or version as it
    appears. Each hit is one issue, shown as its best-matching post (the body or
    a comment) with the issue's number, title, state and labels; `score` only
    orders the hits, higher first. Call get_issue on a hit to read the whole
    thread.
    """
    _check_index()
    filters = reads.make_filters(
        repos=repos, state=state, labels=labels, author=author, kind=kind, since=since
    )
    try:
        return reads.search(query, filters, limit)
    except ValueError as error:
        raise ToolError(str(error)) from error


@server.tool(annotations=READ_ONLY)
def get_issue(
    repo: Annotated[str, Field(description="The repo, as owner/name.")],
    number: Annotated[int, Field(description="The issue number.")],
) -> Thread:
    """Read one issue in full: its title, state, labels and link, then the body
    and every comment, oldest first."""
    _check_index()
    thread = reads.get_issue(repo, number)
    if thread is None:
        raise ToolError(
            f"{repo}#{number} is not in the index. It may not exist, or be a pull "
            "request, or the repo may not be synced: status lists the synced repos."
        )
    return thread


@server.tool(annotations=READ_ONLY)
def status() -> Status:
    """Which repos are synced, with how many issues each, how far each backfill
    has got and when it was last checked for changes, and whether a worker is
    running to keep them current."""
    _check_index()
    return reads.status()
