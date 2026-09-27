"""Index everything already in the repo, one page per run.

A 25,000-issue repo is 250 pages at 100 each, and the indexing behind it runs
for hours against GitHub's rate limit, so this will be interrupted. The way it
survives that is by never holding more than one page. Each run fetches a page,
fans the issues out, and sends itself an event carrying the cursor for the next
page — which starts a fresh run.

Two things follow. Inngest replays a function from the top on every step, so a
single run looping 250 pages would replay 250 saved results on each pass; one
page per run keeps every replay small. And the cursor lives in the event, so a
dead run resumes at the page it died on with no cursor table to keep in sync.
"""

import datetime
from typing import NotRequired, TypedDict, cast

import inngest

from gh_issue.functions.client import client
from gh_issue.functions.events import EVENT_BACKFILL_PAGE, EVENT_ISSUE_CHANGED
from gh_issue.github import api
from gh_issue.index.records.backfills import BackfillProgress
from gh_issue.index.storage import backfills
from gh_issue.utils import now


class BackfillPage(TypedDict):
    repo: str
    # Absent on the first page. After that, GitHub's opaque `after` cursor.
    cursor: NotRequired[str]
    # Identifies one backfill from start to finish. `gh-issue sync` picks it
    # and records it before sending the first page; every page carries it on.
    chain: str


def page_event_id(chain: str, cursor: str | None) -> str:
    """The id of the event that starts one page of one backfill.

    The chain uses it for the page it sends next, and `gh-issue sync` uses it to
    resume, so both always name a page the same way and Inngest keeps only one.
    """
    return f"backfill:{chain}:{cursor or 'start'}"


def _still_current(repo: str, chain: str) -> bool:
    """Whether this chain is still the repo's backfill. `gh-issue remove`
    deletes the record, and syncing again afterwards records a new chain;
    either way, pages this chain already queued should do nothing."""
    recorded = backfills().get(repo)
    return recorded is not None and recorded["chain"] == chain


def _record_progress(repo: str, chain: str, next_cursor: str | None) -> BackfillProgress | None:
    # Only over this chain's own record. A removal can land between the check
    # at the top of the run and this write, and writing anyway would put the
    # record back: the next page would then find its chain alive again.
    if not _still_current(repo, chain):
        return None
    progress: BackfillProgress = {
        "repo": repo,
        "chain": chain,
        "next": next_cursor,
        "finished": next_cursor is None,
        "at": now(),
    }
    backfills().set(progress)
    return progress


@client.create_function(
    fn_id="backfill-page",
    # Starts on an event it also sends to itself, one page at a time.
    trigger=inngest.TriggerEvent(event=EVENT_BACKFILL_PAGE),
    # One GitHub call per run. The token's 5,000 an hour is split three ways:
    # 4,500 for index-issue, where the volume is, 250 here, and the remainder
    # for the hourly sweep. 250 still pages a 25,000-issue repo inside an hour,
    # well before index-issue could get through what it fans out.
    throttle=inngest.Throttle(limit=250, period=datetime.timedelta(hours=1)),
)
def backfill_page(ctx: inngest.ContextSync) -> dict[str, object]:
    event = cast(BackfillPage, ctx.event.data)
    repo = event["repo"]
    cursor = event.get("cursor")
    chain = event["chain"]

    # The next page is already queued in Inngest when `gh-issue remove` runs,
    # and nothing can take it back out, so each page asks first. Plain Python
    # rather than a step, so it is asked again on every replay and a removal
    # stops the page at its next step.
    if not _still_current(repo, chain):
        return {"repo": repo, "skipped": "no longer this repo's backfill"}

    page = ctx.step.run("fetch-page", api.list_issues, repo, cursor)

    # `/issues` returns pull requests too, marked by a `pull_request` field. On
    # inngest/inngest that is 4,496 of them against 313 issues.
    issues = [issue for issue in page["issues"] if "pull_request" not in issue]

    # One event per issue, so each becomes an independent run with its own
    # retries — one issue that fails cannot take the page with it. The issue
    # rides along, so index-issue never refetches it. The id means that if this
    # page ever runs twice in a chain — a resume after a crash — Inngest drops
    # the repeated hand-offs instead of indexing every issue on it again.
    if issues:
        ctx.step.send_event(
            "fan-out",
            [
                inngest.Event(
                    id=f"issue:{chain}:{issue['number']}",
                    name=EVENT_ISSUE_CHANGED,
                    data={"repo": repo, "issue": issue},
                )
                for issue in issues
            ],
        )

    # Where this backfill got to, for `gh-issue sync` and `status`; a page only
    # ever reads which chain it names. A step for two reasons: the write happens
    # once instead of on every replay, and `at` is read once instead of taking
    # whatever the last replay saw. It comes before the next page is sent, so if
    # the run dies in between, the record names a page nobody has started and
    # `sync` resumes there, rather than naming one that already ran.
    ctx.step.run("record-progress", _record_progress, repo, chain, page["next"])

    # The next page is its own run. The event id is what keeps it to exactly
    # one: Inngest ignores a second event with the same id for 24 hours, so if
    # this send is retried after the event already went out, the duplicate is
    # dropped instead of starting a second chain that pages — and indexes — the
    # rest of the repo twice. The chain is part of the id so that a genuinely
    # new backfill of the same repo is not mistaken for a retry of the old one.
    #
    # Note the check is on the cursor, not on whether this page had issues: a
    # page can be nothing but pull requests and the repo still carry on after it.
    if page["next"]:
        ctx.step.send_event(
            "next-page",
            [
                inngest.Event(
                    id=page_event_id(chain, page["next"]),
                    name=EVENT_BACKFILL_PAGE,
                    data={"repo": repo, "cursor": page["next"], "chain": chain},
                )
            ],
        )

    return {
        "repo": repo,
        "issues": len(issues),
        "pull_requests": len(page["issues"]) - len(issues),
        "next": page["next"],
    }
