"""The read side, shared by the CLI and the MCP server.

Nothing here goes through Inngest. Durable execution earns its keep on the write
side, where a run makes dozens of network calls and any of them can fail; a read
is one embedding and one Qdrant query, answered in milliseconds, and a caller
waiting on it wants an answer or an error rather than a retry schedule.
"""

from typing import TypedDict

from gh_issue.embed import embed_texts
from gh_issue.index.store import SearchFilters, SearchHit
from gh_issue.settings import cursors, store


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


def search(query: str, filters: SearchFilters | None = None, limit: int = 10) -> list[SearchHit]:
    """Chunks closest in meaning to `query`, across every indexed repo unless
    the filters say otherwise."""
    text = query.strip()
    if not text:
        raise ValueError("query is empty")
    [vector] = embed_texts([text])
    return store().search(vector, limit=limit, filters=filters)


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


def status() -> list[RepoStatus]:
    """Every repo that is indexed, tracked, or both."""
    counts = store().stats()
    tracked = cursors()
    repos = sorted(set(counts) | set(tracked.tracked()))
    return [
        {
            "repo": repo,
            "issues": counts[repo]["issues"] if repo in counts else 0,
            "chunks": counts[repo]["chunks"] if repo in counts else 0,
            "caught_up_to": tracked.get(repo),
        }
        for repo in repos
    ]
