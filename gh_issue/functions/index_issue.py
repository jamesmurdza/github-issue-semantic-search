"""Index one issue.

This is where the pipeline actually happens, and every trigger ends up here: the
backfill, the GitHub webhook and the hourly sweep all emit the same event, so
there is one indexing path rather than three that have to agree.

The shape of the run is the point. Everything that leaves the process gets its
own `step.run`, so Inngest checkpoints it and a retry resumes rather than
restarting. Everything that is pure computation over data already in hand is
plain Python with no step around it, because re-running it on a replay costs
nothing and wrapping it would only add checkpoints to carry.

The function is sync. Every piece of work it does — the GitHub call, the Qdrant
reads and writes, the ONNX model — is blocking, so declaring it `async` would
buy no concurrency and would stall the event loop while pretending otherwise.
`ContextSync` also takes the handler arguments directly, which is why there is no
`functools.partial` anywhere below.
"""

import datetime
from itertools import batched
from typing import NotRequired, TypedDict, cast

import inngest

from gh_issue.chunking import (
    chunks_for,
    issue_meta,
    metadata_outdated,
    needs_embedding,
    orphaned_ids,
)
from gh_issue.embed import BATCH as EMBED_BATCH
from gh_issue.embed import embed_chunks
from gh_issue.functions.client import client
from gh_issue.github import api
from gh_issue.github.types import Comment, Issue
from gh_issue.settings import EVENT_ISSUE_CHANGED, store


class IssueChanged(TypedDict):
    """The one event every trigger emits.

    The issue travels inside it, so the run never has to fetch what the trigger
    already had.
    """

    repo: str
    issue: Issue
    # Set only by the webhook transform, and only when the issue itself was
    # deleted. A deleted *comment* leaves the issue in place, so it is an
    # ordinary reindex and cleanup drops the chunk.
    deleted: NotRequired[bool]


@client.create_function(
    fn_id="index-issue",
    # Fired by all three sources: backfill, webhook, hourly sweep.
    trigger=inngest.TriggerEvent(event=EVENT_ISSUE_CHANGED),
    # No key on the throttle. GitHub's 5,000 an hour belongs to the token, not to
    # the repository — verified by watching x-ratelimit-remaining fall across
    # calls to four different repos — so keying this on the repo would let two
    # repos spend one budget twice over. 4,500 leaves the backfill's paging room.
    throttle=inngest.Throttle(limit=4500, period=datetime.timedelta(hours=1)),
)
def index_issue(ctx: inngest.ContextSync) -> dict[str, object]:
    event = cast(IssueChanged, ctx.event.data)
    repo = event["repo"]
    issue = event["issue"]
    number = issue["number"]
    index = store()

    if event.get("deleted"):
        ctx.step.run("purge", index.delete_by_issue, repo, number)
        return {"repo": repo, "issue": number, "purged": True}

    # The listing already told us how many comments there are, so a thread with
    # none skips the call entirely. The condition reads event data, which cannot
    # change between replays, so the step graph stays stable.
    comments: list[Comment] = []
    if issue.get("comments", 0) > 0:
        comments = ctx.step.run("fetch-comments", api.list_comments, repo, number)

    # The `or {}` is not defensive noise. A step returning an empty dict hands
    # back None, while every other falsy value survives the round trip — `[]`,
    # `0` and `""` all come back intact. An unindexed issue has nothing stored,
    # so this is the ordinary case on a first backfill, not an edge case.
    existing = ctx.step.run("lookup-stored", index.fingerprints_for_issue, repo, number) or {}

    # No step.run on any of this: pure functions over results already in hand.
    chunks = chunks_for(repo, issue, comments)
    hashes = {chunk_id: stored["content_hash"] for chunk_id, stored in existing.items()}
    stale = needs_embedding(chunks, hashes)
    orphans = orphaned_ids(chunks, hashes)
    meta = issue_meta(issue)
    refresh = metadata_outdated(meta, existing.values())

    # Closing, relabelling or retitling an issue changes the copy of that
    # metadata on every chunk without changing any chunk's text, so the hashes
    # cannot see it. One write by filter fixes the whole thread, and it only
    # runs when a stored copy actually differs. It goes before the upserts so
    # that a run dying partway never leaves old chunks with old metadata behind
    # new ones.
    if refresh:
        ctx.step.run("refresh-metadata", index.set_issue_meta, repo, number, meta)

    # Batched because a step's return value is capped at 4 MiB and one vector is
    # roughly 8 KB of JSON, so an unbatched embed step fails somewhere past four
    # hundred comments. The batch ids come from `stale`, which is derived from
    # memoized step results, so they are identical on every replay.
    for position, group in enumerate(batched(stale, EMBED_BATCH)):
        embedded = ctx.step.run(f"embed-{position}", embed_chunks, list(group))
        ctx.step.run(f"upsert-{position}", index.upsert, embedded)

    # Cleanup is its own step, after every write. Collapse it into the upsert and
    # a run that dies in between leaves the new version written while the old
    # version's leftovers sit in the index still answering queries. Kept apart, a
    # retry picks up at the delete instead of redoing the issue.
    ctx.step.run("cleanup", index.delete_many, orphans)

    return {
        "repo": repo,
        "issue": number,
        "embedded": len(stale),
        "deleted": len(orphans),
        "refreshed": refresh,
    }
