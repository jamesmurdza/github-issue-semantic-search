"""The `gh-issue` command.

Settings come from the environment, and from a `.env` in the working directory
for anything the environment does not already set. The `.env` has to be loaded
before the Inngest client is built, because the client reads INNGEST_DEV once,
when it is created. So nothing here imports the functions at module level; each
command imports what it needs when it runs.
"""

import cyclopts
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


def main() -> None:
    load_env()
    app()
