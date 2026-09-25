"""The hourly sweep: whatever the webhooks missed.

GitHub delivers a webhook once and never retries it. A delivery that fails — the
worker was down, the network blinked — is simply gone. So once an hour this asks
GitHub for everything updated since the last sweep and pushes it through the same
`repo/issue.changed` event as everything else. Indexing an issue twice is a
no-op, so overlapping with the webhooks costs a comment fetch, never correctness.

What `updated_at` does and does not catch, tested against a real repo rather than
assumed: creating, editing and deleting a comment each bump the issue's
`updated_at`, so all three come back from `since` and get reindexed — cleanup
then drops a deleted comment's chunk. A deleted *issue* is the exception. It
vanishes from the API entirely, so no `since` query can return it, and its chunks
stay until something tells the index directly. That is what the webhook's
`deleted` flag is for; the sweep alone cannot see it.

Two functions rather than one. `reconcile` is the cron and only fans out;
`sweep_repo` does one repo per run. Twenty repos looped inside one run would be a
long run replaying every saved step on each pass, and one slow repo would hold
up the rest.
"""

import itertools
from typing import TypedDict, cast

import inngest

from gh_issue.clock import now
from gh_issue.functions.client import client
from gh_issue.github import api
from gh_issue.settings import EVENT_ISSUE_CHANGED, EVENT_SWEEP_REQUESTED, cursors


class SweepRequested(TypedDict):
    repo: str


def _tracked() -> list[str]:
    return cursors().tracked()


def _read_cursor(repo: str) -> str | None:
    return cursors().get(repo)


def _advance_cursor(repo: str, since: str, started: str) -> bool:
    """Move the cursor from `since` to `started`, only if it still says `since`.

    Gone means `gh-issue remove` ran while this sweep was listing, and writing
    it back would have the cron sweep the removed repo every hour. Changed
    means something else moved it, another sweep or a fresh `sync`, and leaving
    theirs costs at most a wider window next time.
    """
    if cursors().get(repo) != since:
        return False
    cursors().set(repo, started)
    return True


@client.create_function(
    fn_id="reconcile",
    # A schedule rather than an event: top of every hour.
    trigger=inngest.TriggerCron(cron="0 * * * *"),
)
def reconcile(ctx: inngest.ContextSync) -> dict[str, object]:
    repos = ctx.step.run("list-repos", _tracked)
    if repos:
        ctx.step.send_event(
            "fan-out",
            [inngest.Event(name=EVENT_SWEEP_REQUESTED, data={"repo": repo}) for repo in repos],
        )
    return {"repos": len(repos)}


@client.create_function(
    fn_id="sweep-repo",
    trigger=inngest.TriggerEvent(event=EVENT_SWEEP_REQUESTED),
)
def sweep_repo(ctx: inngest.ContextSync) -> dict[str, object]:
    repo = cast(SweepRequested, ctx.event.data)["repo"]

    # Where the last sweep started. It does not move while the worker is down,
    # so the first sweep afterwards covers the whole gap.
    since = ctx.step.run("read-cursor", _read_cursor, repo)
    if since is None:
        # Listing without `since` returns the entire repo. That is the backfill's
        # job, and `gh-issue sync` sets the cursor when it starts one.
        return {"repo": repo, "skipped": "no cursor"}

    # The new cursor is the moment *before* listing, so anything updated while we
    # list is caught next time. It has to be read inside a step: the body replays
    # once per step, so a clock read in plain Python would be read again on every
    # pass, and the cursor would land on whatever the final replay saw — after
    # the listing, silently skipping what changed during it.
    started = ctx.step.run("mark-start", now)

    # Usually one page. After downtime it can be several, and each page is a
    # step, so a run that dies partway resumes at the page it was on.
    changed = 0
    cursor: str | None = None
    for position in itertools.count():
        page = ctx.step.run(f"list-since-{position}", api.list_issues, repo, cursor, since)
        issues = [issue for issue in page["issues"] if "pull_request" not in issue]
        if issues:
            ctx.step.send_event(
                f"fan-out-{position}",
                [
                    inngest.Event(name=EVENT_ISSUE_CHANGED, data={"repo": repo, "issue": issue})
                    for issue in issues
                ],
            )
        changed += len(issues)
        cursor = page["next"]
        if cursor is None:
            break

    # Only after every event is out. Die before this and the next sweep covers
    # the same window again, which is harmless for the same reason as above.
    moved = ctx.step.run("advance-cursor", _advance_cursor, repo, since, started)

    return {"repo": repo, "changed": changed, "since": since, "until": started, "moved": moved}
