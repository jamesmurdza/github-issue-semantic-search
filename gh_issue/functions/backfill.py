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

from gh_issue.clock import now
from gh_issue.functions.client import client
from gh_issue.github import api
from gh_issue.index.backfills import BackfillProgress
from gh_issue.settings import EVENT_BACKFILL_PAGE, EVENT_ISSUE_CHANGED, backfills


class BackfillPage(TypedDict):
    repo: str
    # Absent on the first page. After that, GitHub's opaque `after` cursor.
    cursor: NotRequired[str]
    # Identifies one backfill from start to finish; see the next-page send below.
    chain: NotRequired[str]


def _record_progress(repo: str, chain: str, next_cursor: str | None) -> BackfillProgress:
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
    # The first page's run id becomes the backfill's identity, and every page
    # after it carries that forward.
    chain = event.get("chain") or ctx.run_id

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

    # Where this backfill got to, for `gh-issue sync` and `status`; the chain
    # itself never reads it. A step for two reasons: the write happens once
    # instead of on every replay, and `at` is read once instead of taking
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
                    id=f"backfill:{chain}:{page['next']}",
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
