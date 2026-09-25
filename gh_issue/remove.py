"""`gh-issue remove`: take a repo out of the index and stop syncing it.

A synced repo leaves four things behind: its chunks in Qdrant, its sweep
cursor, its backfill record and any failure markers. Removing deletes all four.

What it can't delete is work already queued in Inngest for the repo: the
backfill's next page, a sweep under way, issues waiting behind the throttle.
Nothing here reaches into the queue. Each function checks instead that the repo
is still synced before it does anything, which means its records are still here
(`settings.is_synced`). So the records go first, and every run for the repo
stops at its next step. The chunks go after.

One gap is left. A write already under way as the records go, one issue's
upsert taking milliseconds, can land after the chunks are deleted. `status` then
still lists the repo, and running `remove` again clears it.
"""

from collections.abc import Callable
from typing import TypedDict

from gh_issue.settings import backfills, cursors, failures, is_synced, store


class Removal(TypedDict):
    repo: str
    chunks: int
    issues: int


def removal(repo: str) -> Removal | None:
    """What removing `repo` would delete; None if nothing here belongs to it.

    Chunks alone count, so that leftovers from the gap above can be removed.
    """
    counts = store().stats().get(repo, {"chunks": 0, "issues": 0})
    if counts["chunks"] == 0 and not is_synced(repo):
        return None
    return {"repo": repo, "chunks": counts["chunks"], "issues": counts["issues"]}


def remove_repo(repo: str) -> None:
    # Records first: from here on, each run for the repo stops at its next step.
    cursors().delete(repo)
    backfills().delete(repo)
    failures().clear_repo(repo)
    store().delete_repo(repo)


def _question(found: Removal) -> str:
    return (
        f"Remove {found['repo']}? Its {found['chunks']:,} chunks ({found['issues']:,} issues) "
        "are deleted and it stops syncing.\n"
        "Syncing it again later starts a full backfill. [y/N] "
    )


def run(
    repo: str,
    *,
    yes: bool,
    say: Callable[[str], None] = print,
    ask: Callable[[str], str] = input,
) -> None:
    found = removal(repo)
    if found is None:
        raise SystemExit(f"{repo} is not synced here. `gh-issue status` lists what is.")
    if not yes:
        try:
            answer = ask(_question(found))
        except EOFError:
            raise SystemExit(
                "Nothing removed: there is no terminal to ask on. "
                "Pass --yes to remove without asking."
            ) from None
        if answer.strip().lower() not in ("y", "yes"):
            say("Nothing removed.")
            return
    remove_repo(repo)
    say(f"Removed {repo}. Work already queued for it will find it gone and do nothing.")
