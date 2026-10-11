"""The read side, shared by the CLI and the MCP server.

Nothing here goes through Inngest. Durable execution earns its keep on the write
side, where a run makes dozens of network calls and any of them can fail; a read
is one embedding and one Qdrant query, answered in milliseconds, and a caller
waiting on it wants an answer or an error rather than a retry schedule.
"""

from typing import TypedDict

from gh_issue.index.embed import embed_query
from gh_issue.index.records.backfills import BackfillProgress
from gh_issue.index.records.heartbeat import WorkerStatus, worker_status
from gh_issue.index.storage import backfills, cursors, failures, store
from gh_issue.index.store import SearchFilters, SearchHit


class Post(TypedDict):
    """The issue body or one comment."""

    kind: str
    author: str
    created_at: str
    text: str


class Thread(TypedDict):
    repo: str
    number: int
    url: str
    title: str
    state: str
    labels: list[str]
    posts: list[Post]


class RepoStatus(TypedDict):
    repo: str
    issues: int
    chunks: int
    # When the last sweep started. Everything GitHub changed before this moment
    # has been handed to the indexer. None when the repo was never synced from
    # this machine.
    caught_up_to: str | None
    # Where the backfill got to; None if none was started from this machine.
    backfill: BackfillProgress | None
    # Issues that failed every retry and are waiting for `gh-issue sync`.
    failed: int


class Status(TypedDict):
    worker: WorkerStatus
    repos: list[RepoStatus]


def make_filters(
    *,
    repos: list[str] | None = None,
    state: str | None = None,
    labels: list[str] | None = None,
    author: str | None = None,
    kind: str | None = None,
    since: str | None = None,
) -> SearchFilters:
    """Search filters from options that were each given or not, the shape both
    the CLI and the MCP tool take them in. Anything not given is left out."""
    filters: SearchFilters = {}
    if repos:
        filters["repos"] = repos
    if state:
        filters["state"] = state
    if labels:
        filters["labels"] = labels
    if author:
        filters["author"] = author
    if kind:
        filters["kind"] = kind
    if since:
        filters["since"] = since
    return filters


def search(query: str, filters: SearchFilters | None = None, limit: int = 10) -> list[SearchHit]:
    """The issues that best match `query` by meaning and by wording, each as its
    best-matching post, across every indexed repo unless the filters say
    otherwise."""
    text = query.strip()
    if not text:
        raise ValueError("query is empty")
    vectors = embed_query(text)
    return store().search(vectors["dense"], vectors["sparse"], limit=limit, filters=filters)


def warm_up() -> None:
    """Load the embedding model now, for a caller that stays up and answers
    searches. Otherwise the first search waits for it: 1.9 s against 0.05 s
    for the ones after it, measured on `gh-issue web`."""
    embed_query("warm up")


def get_issue(repo: str, number: int) -> Thread | None:
    """One whole indexed thread: the body, then the comments oldest first."""
    chunks = store().chunks_for_issue(repo, number)
    if not chunks:
        return None
    chunks.sort(key=lambda chunk: (chunk["kind"] != "body", chunk["created_at"], chunk["id"]))
    head = chunks[0]
    # The body chunk's text is the title and body together, because that is
    # what gets embedded. The thread carries the title on its own already.
    title_prefix = f"{head['title']}\n\n"
    return {
        "repo": repo,
        "number": number,
        "url": f"https://github.com/{repo}/issues/{number}",
        "title": head["title"],
        "state": head["state"],
        "labels": head["labels"],
        "posts": [
            {
                "kind": chunk["kind"],
                "author": chunk["author"],
                "created_at": chunk["created_at"],
                "text": (
                    chunk["text"].removeprefix(title_prefix)
                    if chunk["kind"] == "body"
                    else chunk["text"]
                ),
            }
            for chunk in chunks
        ],
    }


def indexed_repos() -> list[str]:
    """The repos a search can find something in, in order. A repo that is
    tracked but has nothing indexed yet is not one of them."""
    return sorted(store().stats())


def status() -> Status:
    """The worker, and every repo that is indexed, tracked, or both."""
    counts = store().stats()
    tracked = cursors()
    progress = backfills()
    repos = sorted(set(counts) | set(tracked.tracked()) | set(progress.repos()))
    return {
        "worker": worker_status(),
        "repos": [
            {
                "repo": repo,
                "issues": counts[repo]["issues"] if repo in counts else 0,
                "chunks": counts[repo]["chunks"] if repo in counts else 0,
                "caught_up_to": tracked.get(repo),
                "backfill": progress.get(repo),
                "failed": len(failures().for_repo(repo)),
            }
            for repo in repos
        ],
    }
