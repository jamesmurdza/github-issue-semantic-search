"""Turning an issue into the records the index stores.

Two ideas carry this module.

An issue is not one document. The title and body go in as one chunk and each
comment as its own, because the answer is usually in a comment rather than the
body, and because comments get edited and deleted one at a time.

Every chunk then gets two identifiers, and they do different jobs. The `id` says
what the chunk *is* and survives edits, so it is what tells you which records to
delete. The `content_hash` says what the chunk currently *says* and changes on
every edit, so it is what tells you which records to embed again. One identifier
cannot do both.
"""

import hashlib
from typing import TypedDict

from gh_issue.github.types import Comment, Issue


class ChunkMeta(TypedDict):
    """Written to every chunk. Qdrant filters on stored payload, so a field that
    is not here cannot be filtered on later."""

    repo: str
    issue_number: int
    state: str
    labels: list[str]
    kind: str
    author: str
    created_at: str


class Chunk(TypedDict):
    id: str
    text: str
    meta: ChunkMeta
    content_hash: str


def _hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _chunk(chunk_id: str, text: str, meta: ChunkMeta) -> Chunk:
    return {"id": chunk_id, "text": text, "meta": meta, "content_hash": _hash(text)}


def chunks_for(repo: str, issue: Issue, comments: list[Comment]) -> list[Chunk]:
    """Every chunk this issue produces right now, body first."""
    number: int = issue["number"]
    state: str = issue["state"]
    labels = [label["name"] for label in issue.get("labels", [])]

    def meta(kind: str, author: str, created_at: str) -> ChunkMeta:
        return {
            "repo": repo,
            "issue_number": number,
            "state": state,
            "labels": list(labels),
            "kind": kind,
            "author": author,
            "created_at": created_at,
        }

    chunks = [
        _chunk(
            f"{repo}#{number}:body",
            f"{issue['title']}\n\n{issue.get('body') or ''}",
            meta("body", issue["user"]["login"], issue["created_at"]),
        )
    ]
    chunks.extend(
        _chunk(
            f"{repo}#{number}:comment:{comment['id']}",
            comment.get("body") or "",
            meta("comment", comment["user"]["login"], comment["created_at"]),
        )
        for comment in comments
    )
    return chunks


def needs_embedding(chunks: list[Chunk], existing: dict[str, str]) -> list[Chunk]:
    """The chunks whose text differs from what we already stored.

    `existing` maps chunk id to stored content hash. A matching hash means the
    text is byte-identical, so a new comment on a thread of two hundred costs
    one embedding rather than two hundred and one.
    """
    return [chunk for chunk in chunks if existing.get(chunk["id"]) != chunk["content_hash"]]


def orphaned_ids(chunks: list[Chunk], existing: dict[str, str]) -> list[str]:
    """Ids the index still holds that this issue no longer produces.

    This is what catches a deleted comment, a secret a maintainer redacted an
    hour after pasting it, and an edit that shortens a comment enough to change
    how it splits. Without it the index only ever grows.
    """
    live = {chunk["id"] for chunk in chunks}
    return [chunk_id for chunk_id in existing if chunk_id not in live]
