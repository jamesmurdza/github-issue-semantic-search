"""`gh-issue sync`: index a repo, or bring its index up to date.

One command for every case, decided from what is recorded locally:

    no backfill recorded             start one
    backfill finished, index empty   start over (the index was wiped)
    otherwise                        sweep for changes since the last sweep, and
                                     if the backfill never finished, resume it
                                     from the last page it recorded
    and always                       retry the issues that failed every retry

`sync` does no indexing itself. It sends Inngest the events that start the
work, and the worker does it, so the command returns in a second or two. The
events carry ids wherever sending one twice would do the work twice, so running
`sync` again while the last one is still going is harmless.
"""

import uuid
from collections.abc import Callable

import inngest

from gh_issue.clock import now
from gh_issue.functions.backfill import page_event_id
from gh_issue.functions.client import client
from gh_issue.github import api
from gh_issue.heartbeat import worker_status
from gh_issue.settings import (
    EVENT_BACKFILL_PAGE,
    EVENT_ISSUE_CHANGED,
    EVENT_SWEEP_REQUESTED,
    backfills,
    cursors,
    failures,
    store,
)
from gh_issue.worker import preflight, print_now

NO_WORKER = (
    "No worker is running here. The work above is queued in Inngest and starts "
    "when one connects: run `gh-issue worker` in another terminal."
)


class RepoNotFound(Exception):
    pass


class CannotSend(Exception):
    pass


def new_chain() -> str:
    return uuid.uuid4().hex[:12]


def send(events: list[inngest.Event]) -> None:
    try:
        client.send_sync(events)
    except Exception as error:
        where = (
            f"Inngest Cloud at {client.api_origin} (inngest.event_key)"
            if client.is_production
            else f"the Inngest dev server at {client.api_origin} (inngest.dev)"
        )
        raise CannotSend(f"Couldn't send events to {where}: {error}") from error


def worker_running() -> bool:
    return worker_status()["connected"]


def known_repos() -> list[str]:
    return sorted(set(backfills().repos()) | set(cursors().tracked()))


def _page_event(repo: str, chain: str, cursor: str | None) -> inngest.Event:
    data: dict[str, str] = {"repo": repo, "chain": chain}
    if cursor:
        data["cursor"] = cursor
    return inngest.Event(id=page_event_id(chain, cursor), name=EVENT_BACKFILL_PAGE, data=data)


def _start_backfill(repo: str, reason: str) -> list[str]:
    if not api.repo_exists(repo):
        raise RepoNotFound(f"{repo} was not found on GitHub (or the token can't see it).")
    chain = new_chain()
    started = now()
    # Recorded before sending, so a second `sync` finds a backfill in progress
    # rather than an empty index and starting another. The sweep takes over
    # from this moment: anything changed later is the sweep's to catch.
    backfills().set({"repo": repo, "chain": chain, "next": None, "finished": False, "at": started})
    cursors().set(repo, started)
    # A full backfill indexes every issue, the ones that failed before included.
    for failure in failures().for_repo(repo):
        failures().clear(repo, failure["number"])
    send([_page_event(repo, chain, None)])
    return [reason]


def sync_repo(repo: str) -> list[str]:
    """Queue what brings `repo` up to date, and say what was queued."""
    progress = backfills().get(repo)
    if progress is None:
        return _start_backfill(
            repo, "started a backfill; `gh-issue status` shows how far it has got"
        )
    stats = store().stats()
    if progress["finished"] and stats.get(repo, {"chunks": 0, "issues": 0})["chunks"] == 0:
        return _start_backfill(
            repo, "the backfill finished but the index is empty, so starting over"
        )

    events = [inngest.Event(name=EVENT_SWEEP_REQUESTED, data={"repo": repo})]
    lines = [f"checking for changes since {cursors().get(repo) or 'the backfill started'}"]
    if not progress["finished"]:
        events.append(_page_event(repo, progress["chain"], progress["next"]))
        lines.append(
            f"the backfill has not finished (last page recorded {progress['at']}); "
            "resuming where it stopped"
        )

    # Read fresh rather than trusting anything kept from the failed run: the
    # issue may have changed since, and the newest copy is the one to index.
    retried: list[int] = []
    gone: list[int] = []
    for failure in failures().for_repo(repo):
        issue = api.get_issue(repo, failure["number"])
        if issue is None:
            gone.append(failure["number"])
            continue
        events.append(inngest.Event(name=EVENT_ISSUE_CHANGED, data={"repo": repo, "issue": issue}))
        retried.append(failure["number"])

    send(events)
    # Only once the events are out: if sending fails, the markers are still
    # there for the next `sync`.
    for number in retried + gone:
        failures().clear(repo, number)
    if retried:
        lines.append(
            f"retrying {len(retried)} issue(s) that failed: " + ", ".join(f"#{n}" for n in retried)
        )
    if gone:
        lines.append("no longer on GitHub, so not retried: " + ", ".join(f"#{n}" for n in gone))
    return lines


def run(repos: list[str], say: Callable[[str], None] = print_now) -> None:
    """The repos are owner/name; the command line checked them."""
    targets = repos or known_repos()
    if not targets:
        raise SystemExit("Nothing has been synced yet. Name a repo: gh-issue sync owner/name")
    problems = preflight(production=client.is_production)
    if problems:
        raise SystemExit("sync can't run:\n" + "\n".join(f"  - {p}" for p in problems))

    not_found = False
    for repo in targets:
        try:
            for line in sync_repo(repo):
                say(f"{repo}: {line}")
        except RepoNotFound as error:
            say(f"{repo}: {error}")
            not_found = True
        except CannotSend as error:
            raise SystemExit(str(error)) from error
    if not worker_running():
        say(NO_WORKER)
    if not_found:
        raise SystemExit(1)
