"""The vector index.

Qdrant runs as a server, started by the `docker-compose.yml` in this repo and
reached through `QDRANT_URL`. The client also has an embedded mode — pass `path`
instead of `url` and it keeps everything in a local directory with no daemon —
and the test suite uses it so that running the tests needs nothing running.

Embedded mode is not what ships, for one measured reason: it silently ignores
payload indexes. Every filtered read then scans the whole collection, which costs
about 19 ms against 1,500 chunks and 240 ms against 15,000. Since a filtered read
happens once per issue, a 25,000-issue repo would spend hours doing nothing but
looking things up. A server builds the indexes and turns that into a lookup.
"""

import uuid
from collections.abc import Mapping
from typing import Any, TypedDict

from qdrant_client import QdrantClient, models

from gh_issue.chunking import EmbeddedChunk

COLLECTION = "gh_issue_chunks"

# Any fixed UUID works; it only has to stay the same across runs so that the
# same chunk id always maps to the same point.
_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

# Fields worth indexing on a server. `repo` and `issue_number` are the ones every
# read filters on; the rest are here so that adding search filters later is a
# query change rather than a reindex.
_KEYWORD_INDEXES = ("repo", "chunk_id", "kind", "state", "labels", "author")


class StoredChunk(TypedDict):
    """A chunk as it comes back out of the index, payload flattened."""

    id: str
    text: str
    content_hash: str
    repo: str
    issue_number: int
    state: str
    labels: list[str]
    kind: str
    author: str
    created_at: str


class SearchHit(StoredChunk):
    score: float


class RepoStats(TypedDict):
    chunks: int
    issues: int


def point_id(chunk_id: str) -> str:
    """Map our string id onto the UUID Qdrant insists on.

    Qdrant point ids must be integers or UUIDs, but ours are readable strings
    like `owner/name#412:comment:9981`. Hashing is deterministic, so re-indexing
    an edited chunk overwrites the same point instead of adding a second copy —
    which is what makes `upsert` idempotent. The readable id rides along in the
    payload as `chunk_id`, because that is what cleanup compares against.
    """
    return str(uuid.uuid5(_NAMESPACE, chunk_id))


def _stored_from_payload(payload: Mapping[str, Any]) -> StoredChunk:
    return {
        "id": payload["chunk_id"],
        "text": payload["text"],
        "content_hash": payload["content_hash"],
        "repo": payload["repo"],
        "issue_number": payload["issue_number"],
        "state": payload["state"],
        "labels": payload["labels"],
        "kind": payload["kind"],
        "author": payload["author"],
        "created_at": payload["created_at"],
    }


class Store:
    def __init__(self, dim: int, url: str | None = None, path: str | None = None) -> None:
        """Connect to a Qdrant server at `url`, or open an embedded one at `path`.

        Exactly one of the two. `url` is what ships; `path` exists so the tests
        can run without a daemon.
        """
        if (url is None) == (path is None):
            raise ValueError("pass exactly one of url= (server) or path= (embedded)")
        self._server = url is not None
        self._client = QdrantClient(url=url) if url is not None else QdrantClient(path=path)
        self._dim = dim
        self._ensure_collection()

    def close(self) -> None:
        """Embedded Qdrant holds a lock on its directory until this is called."""
        self._client.close()

    def _ensure_collection(self) -> None:
        """Create the collection, and the payload indexes if there is a server.

        Every read below filters on `repo` and `issue_number`, so those two carry
        the load. Embedded Qdrant discards index requests and warns about it, so
        we do not ask — it scans instead, which is correct but linear.
        """
        if self._client.collection_exists(COLLECTION):
            return
        self._client.create_collection(
            COLLECTION,
            vectors_config=models.VectorParams(size=self._dim, distance=models.Distance.COSINE),
        )
        if not self._server:
            return
        for field in _KEYWORD_INDEXES:
            self._client.create_payload_index(
                COLLECTION, field_name=field, field_schema=models.PayloadSchemaType.KEYWORD
            )
        self._client.create_payload_index(
            COLLECTION, field_name="issue_number", field_schema=models.PayloadSchemaType.INTEGER
        )

    @staticmethod
    def _issue_filter(repo: str, number: int) -> models.Filter:
        """Scoped by repo as well as number, so two repos can hold issue 412."""
        return models.Filter(
            must=[
                models.FieldCondition(key="repo", match=models.MatchValue(value=repo)),
                models.FieldCondition(key="issue_number", match=models.MatchValue(value=number)),
            ]
        )

    def _scroll(self, where: models.Filter | None, fields: list[str] | bool) -> list[models.Record]:
        records: list[models.Record] = []
        offset: models.ExtendedPointId | None = None
        while True:
            page, offset = self._client.scroll(
                COLLECTION,
                scroll_filter=where,
                limit=256,
                offset=offset,
                with_payload=fields,
                with_vectors=False,
            )
            records.extend(page)
            if offset is None:
                return records

    def hashes_for_issue(self, repo: str, number: int) -> dict[str, str]:
        """chunk id -> content hash for everything stored for this issue.

        This is the read that makes the rest cheap: comparing these hashes is
        what decides which chunks need embedding again and which are orphans.
        """
        records = self._scroll(self._issue_filter(repo, number), ["chunk_id", "content_hash"])
        return {
            str(record.payload["chunk_id"]): str(record.payload["content_hash"])
            for record in records
            if record.payload is not None
        }

    def chunks_for_issue(self, repo: str, number: int) -> list[StoredChunk]:
        """Full records for one thread. A scroll, not a search — there is no query."""
        records = self._scroll(self._issue_filter(repo, number), True)
        return [_stored_from_payload(r.payload) for r in records if r.payload is not None]

    def upsert(self, chunks: list[EmbeddedChunk]) -> int:
        if not chunks:
            return 0
        self._client.upsert(
            COLLECTION,
            points=[
                models.PointStruct(
                    id=point_id(chunk["id"]),
                    vector=chunk["vector"],
                    payload={
                        "chunk_id": chunk["id"],
                        "content_hash": chunk["content_hash"],
                        "text": chunk["text"],
                        **chunk["meta"],
                    },
                )
                for chunk in chunks
            ],
        )
        return len(chunks)

    def delete_many(self, chunk_ids: list[str]) -> int:
        """Remove named chunks. This is the cleanup step's only job."""
        if not chunk_ids:
            return 0
        self._client.delete(
            COLLECTION,
            points_selector=models.PointIdsList(
                points=[point_id(chunk_id) for chunk_id in chunk_ids]
            ),
        )
        return len(chunk_ids)

    def delete_by_issue(self, repo: str, number: int) -> None:
        """Purge a whole thread, for when the issue itself was deleted."""
        self._client.delete(
            COLLECTION,
            points_selector=models.FilterSelector(filter=self._issue_filter(repo, number)),
        )

    def search(self, vector: list[float], limit: int = 10) -> list[SearchHit]:
        hits = self._client.query_points(
            COLLECTION, query=vector, limit=limit, with_payload=True
        ).points
        return [
            {**_stored_from_payload(hit.payload), "score": hit.score}
            for hit in hits
            if hit.payload is not None
        ]

    def stats(self) -> dict[str, RepoStats]:
        """Chunks and distinct issues per repo, for `gh-issue status`."""
        counts: dict[str, int] = {}
        issues: dict[str, set[int]] = {}
        for record in self._scroll(None, ["repo", "issue_number"]):
            if record.payload is None:
                continue
            repo = str(record.payload["repo"])
            counts[repo] = counts.get(repo, 0) + 1
            issues.setdefault(repo, set()).add(int(record.payload["issue_number"]))
        return {
            repo: {"chunks": count, "issues": len(issues[repo])} for repo, count in counts.items()
        }
