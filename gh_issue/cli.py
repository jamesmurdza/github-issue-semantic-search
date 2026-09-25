"""The `gh-issue` command.

Settings come from the environment, and from a `.env` in the working directory
for anything the environment does not already set. The `.env` has to be loaded
before the Inngest client is built, because the client reads INNGEST_DEV once,
when it is created. So nothing here imports the functions at module level; each
command imports what it needs when it runs.
"""

import datetime
from typing import Annotated, Literal

import cyclopts
from cyclopts import Parameter
from dotenv import find_dotenv, load_dotenv

app = cyclopts.App(
    name="gh-issue",
    help="Semantic search over a GitHub repo's issue history, indexed with Inngest.",
)


def load_env() -> None:
    """Read `.env` from the working directory (or the nearest parent that has
    one). A variable already set in the environment wins over the file."""
    load_dotenv(find_dotenv(usecwd=True), override=False)


@app.command
def worker() -> None:
    """Run the Inngest functions. Keep it running while anything is syncing; Ctrl+C stops it."""
    from gh_issue.worker import run

    run()


@app.command
def sync(*repos: str) -> None:
    """Index a repo, or bring it up to date. Safe to run again at any time.

    The first sync of a repo starts a backfill. After that, sync checks for
    changes, resumes a backfill that stopped, and retries issues that failed.

    Parameters
    ----------
    repos
        One or more repos as owner/name. Leave out to sync every repo synced before.
    """
    from gh_issue.sync import run

    run(list(repos))


@app.command
def search(
    *words: str,
    repo: list[str] | None = None,
    state: Literal["open", "closed"] | None = None,
    label: list[str] | None = None,
    author: str | None = None,
    kind: Literal["body", "comment"] | None = None,
    since: str | None = None,
    limit: Annotated[int, Parameter(name=["--limit", "-k"])] = 10,
) -> None:
    """Search issue history by meaning, across every synced repo.

    Parameters
    ----------
    words
        What you are looking for, in your own words. No quotes needed.
    repo
        Only this repo, as owner/name. Repeat for several.
    state
        Only open or only closed issues.
    label
        Only issues with this label. Repeat to require several.
    author
        Only posts written by this GitHub user.
    kind
        Only issue bodies, or only comments.
    since
        Only posts written on or after this date, e.g. 2026-01-31.
    limit
        How many results to show.
    """
    from gh_issue import display, query

    _check_index()
    filters = query.make_filters(
        repos=repo, state=state, labels=label, author=author, kind=kind, since=since
    )
    try:
        hits = query.search(" ".join(words), filters, limit)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    print(display.format_hits(hits), flush=True)


@app.command
def status() -> None:
    """What is synced, how far each backfill has got, and whether a worker is up."""
    from gh_issue import display, query

    _check_index()
    now = datetime.datetime.now(datetime.UTC)
    print(display.format_status(query.status(), now), flush=True)


@app.command
def remove(
    repo: str,
    *,
    yes: Annotated[bool, Parameter(name=["--yes", "-y"], negative="")] = False,
) -> None:
    """Take a repo out of the index and stop syncing it.

    Deletes its chunks and its sync records. Work already queued for it in
    Inngest finds it gone and does nothing.

    Parameters
    ----------
    repo
        The repo to remove, as owner/name.
    yes
        Remove without asking first.
    """
    from gh_issue import remove

    _check_index()
    remove.run(repo, yes=yes)


def _check_index() -> None:
    from gh_issue import settings

    if problem := settings.index_problem():
        raise SystemExit(problem)


mcp = cyclopts.App(
    name="mcp",
    help="Serve the index to a coding agent over MCP: search_issues, get_issue, status.",
)
app.command(mcp)


@mcp.command
def stdio() -> None:
    """Over stdio, for an agent that starts the server itself (Claude Code, Cursor...)."""
    from gh_issue.mcp_server import mcp as server

    # No banner: stdout is the protocol's channel here, and clients show stderr.
    server.run(transport="stdio", show_banner=False)


@mcp.command
def http(host: str = "127.0.0.1", port: int = 8000, path: str = "/mcp") -> None:
    """Over streamable HTTP, for an agent that connects to a running server.

    Parameters
    ----------
    host
        Interface to listen on. The default only accepts local connections.
    port
        Port to listen on.
    path
        URL path the MCP endpoint is served at.
    """
    from gh_issue.mcp_server import mcp as server

    server.run(transport="http", host=host, port=port, path=path)


def main() -> None:
    load_env()
    app()
