"""The `gh-issue` command.

Settings come from ~/.gh-issue/config.toml, and from the environment, which
wins (see gh_issue/config.py). The file has to be applied before the Inngest
client is built, because the client reads INNGEST_DEV once, when it is created.
So nothing here imports the functions at module level; each command imports
what it needs when it runs.
"""

import datetime
import io
import re
import sys
from typing import Annotated, Literal

import cyclopts
from cyclopts import Parameter, validators

from gh_issue import config

app = cyclopts.App(
    name="gh-issue",
    help="Semantic search over a GitHub repo's issue history, indexed with Inngest.",
)

REPO_NAME = re.compile(r"^[\w.-]+/[\w.-]+$")


def _owner_slash_name(type_: object, name: str) -> None:
    if not REPO_NAME.match(name):
        raise ValueError("Use owner/name, e.g. inngest/inngest.")


# A repo named on the command line, checked as the line is read.
Repo = Annotated[str, Parameter(validator=_owner_slash_name)]


@app.command
def worker() -> None:
    """Run the Inngest functions. Keep it running while anything is syncing; Ctrl+C stops it."""
    from gh_issue.worker import run

    run()


@app.command
def sync(repos: list[Repo] | None = None, /) -> None:
    """Index a repo, or bring it up to date. Safe to run again at any time.

    The first sync of a repo starts a backfill. After that, sync checks for
    changes, resumes a backfill that stopped, and retries issues that failed.

    Parameters
    ----------
    repos
        One or more repos as owner/name. Leave out to sync every repo synced before.
    """
    from gh_issue.sync import run

    run(repos or [])


@app.command
def search(
    *words: Annotated[str, Parameter(required=True)],
    repo: list[Repo] | None = None,
    state: Literal["open", "closed"] | None = None,
    label: list[str] | None = None,
    author: str | None = None,
    kind: Literal["body", "comment"] | None = None,
    since: datetime.date | None = None,
    limit: Annotated[
        int, Parameter(name=["--limit", "-k"], validator=validators.Number(gte=1, lte=50))
    ] = 10,
) -> None:
    """Search issue history by meaning and by wording, across every synced repo.

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
        How many results to show, 1 to 50.
    """
    from gh_issue import display, query

    _check_index()
    filters = query.make_filters(
        repos=repo,
        state=state,
        labels=label,
        author=author,
        kind=kind,
        since=since.isoformat() if since else None,
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
    repo: Repo,
    /,
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


config_app = cyclopts.App(
    name="config",
    help="See and change the settings. They are kept in ~/.gh-issue/config.toml, "
    "and an environment variable for the same setting wins over the file.",
)
app.command(config_app)

# A setting's name, checked as the line is read: a typo is refused with the
# closest real name.
Setting = Annotated[str, Parameter(choices=list(config.SETTINGS))]


def config_list() -> None:
    """Every setting: its value (secrets masked), where the value came from, and
    what the setting is for."""
    from gh_issue import display

    print(display.format_config(str(config.config_path()), config.shown()), flush=True)


config_app.default(config_list)
config_app.command(config_list, name="list")


@config_app.command(name="get")
def config_get(key: Setting, /) -> None:
    """Print one setting's value, as `config list` shows it: masked if it is a
    secret, the default if nothing sets it.

    Parameters
    ----------
    key
        The setting.
    """
    [row] = [row for row in config.shown() if row["key"] == key]
    if row["value"] is None:
        raise SystemExit(f"{key} is not set.")
    print(row["value"], flush=True)


@config_app.command(name="set")
def config_set(key: Setting, value: str | None = None, /) -> None:
    """Change a setting.

    Parameters
    ----------
    key
        The setting.
    value
        The new value. Leave it out to type it in instead: a secret is then not
        shown as you type, and stays out of your shell's history.
    """
    try:
        config.run_set(key, value)
    except config.ConfigError as error:
        raise SystemExit(str(error)) from error


@config_app.command(name="unset")
def config_unset(key: Setting, /) -> None:
    """Remove a setting from the file, so its default (or the environment) applies.

    Parameters
    ----------
    key
        The setting.
    """
    try:
        config.run_unset(key)
    except config.ConfigError as error:
        raise SystemExit(str(error)) from error


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
def http(
    host: str = "127.0.0.1",
    port: Annotated[int, Parameter(validator=validators.Number(gte=1, lte=65535))] = 8000,
    path: str = "/mcp",
) -> None:
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
    # Issue text is whatever people typed. On Windows, Python writes to a pipe
    # or a file in the system code page, which can't encode most of it, so
    # `gh-issue search ... | grep` would crash on the first U+276F. A console
    # already gets UTF-8, so this changes nothing there.
    for stream in (sys.stdout, sys.stderr):
        if isinstance(stream, io.TextIOWrapper):
            stream.reconfigure(encoding="utf-8")
    try:
        config.apply()
    except config.ConfigError as error:
        raise SystemExit(str(error)) from error
    app()
