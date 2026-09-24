"""The vector index.

Qdrant runs as a server, started by the `docker-compose.yml` in this repo and
reached through `QDRANT_URL`. The client also has an embedded mode — pass `path`
instead of `url` and it keeps everything in a local directory with no daemon —
and the test suite uses it so that running the tests needs nothing running.

Embedded mode is not what ships, for one measured reason: it silently ignores
payload indexes, so every filtered read scans the whole collection. Since a
filtered read happens once per issue, that cost is paid 25,000 times on a repo
of the size this example is written for. Measured on the same machine, same data:

    chunks    embedded   server
     1,500     19.4 ms   13.1 ms
    15,000    240.3 ms   12.3 ms

Embedded grows with the collection. The server does not — past a few thousand
chunks it is just the round trip.
"""

import datetime
import uuid
from collections.abc import Mapping
from typing import Any, Literal, TypedDict

from qdrant_client import QdrantClient, models

from gh_issue.chunking import EmbeddedChunk, IssueMeta

COLLECTION = "gh_issue_chunks"

# Any fixed UUID works; it only has to stay the same across runs so that the
# same chunk id always maps to the same point.
_NAMESPACE = uuid.UUID("6ba7b810-9dad-11d1-80b4-00c04fd430c8")

# Fields indexed on a server. `repo` and `issue_number` are the ones every
# indexing read filters on; the rest are the search filters.
_KEYWORD_INDEXES = ("repo", "chunk_id", "kind", "state", "labels", "author")

# Facet queries return the top N values only. Nobody indexes this many repos.
_MAX_REPOS = 10_000


class StoredChunk(TypedDict):
    """A chunk as it comes back out of the index, payload flattened."""

    id: str
    text: str
    content_hash: str
    repo: str
    issue_number: int
    title: str
    state: str
    labels: list[str]
    kind: str
    author: str
    created_at: str


class SearchHit(StoredChunk):
    score: float


class Fingerprint(IssueMeta):
    """What a reindex needs to know about one stored chunk: whether its text
    changed (the hash) and whether its copy of the issue metadata did."""

    content_hash: str


class SearchFilters(TypedDict, total=False):
    """Narrowing for a search. Every key is optional; together they are AND."""

    repos: list[str]  # any of these; empty or missing means every repo
    state: str  # "open" or "closed"
    labels: list[str]  # all of these, the way GitHub's own label filter works
    author: str
    kind: str  # "body" or "comment"
    since: str  # written on or after, e.g. "2026-01-31" or "2026-01-31T09:00:00Z"


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
        "title": payload["title"],
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

        Every indexing read filters on `repo` and `issue_number`, so those two
        carry the load. Embedded Qdrant discards index requests and warns about
        it, so we do not ask — it scans instead, which is correct but linear.

        The indexes are requested on every open, not only on create. Qdrant
        treats an index that already exists as a no-op, so a collection made
        before a field was added picks the new index up without a reindex.
        """
        if not self._client.collection_exists(COLLECTION):
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
        self._client.create_payload_index(
            COLLECTION, field_name="created_at", field_schema=models.PayloadSchemaType.DATETIME
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

    def fingerprints_for_issue(self, repo: str, number: int) -> dict[str, Fingerprint]:
        """chunk id -> fingerprint for everything stored for this issue.

        This is the read that makes the rest cheap. The hashes decide which
        chunks need embedding again and which are orphans; the metadata copies
        decide whether the issue was closed, relabelled or retitled since.
        """
        fields = ["chunk_id", "content_hash", "title", "state", "labels"]
        records = self._scroll(self._issue_filter(repo, number), fields)
        return {
            str(record.payload["chunk_id"]): {
                "content_hash": record.payload["content_hash"],
                "title": record.payload["title"],
                "state": record.payload["state"],
                "labels": record.payload["labels"],
            }
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

    def set_issue_meta(self, repo: str, number: int, meta: IssueMeta) -> None:
        """Overwrite the issue metadata on every chunk of one thread, in place.

        One write selected by filter, vectors and text untouched. Measured at
        7 to 37 ms for threads of 20 to 1,000 chunks.
        """
        self._client.set_payload(
            COLLECTION, payload=dict(meta), points=self._issue_filter(repo, number)
        )

    def delete_by_issue(self, repo: str, number: int) -> None:
        """Purge a whole thread, for when the issue itself was deleted."""
        self._client.delete(
            COLLECTION,
            points_selector=models.FilterSelector(filter=self._issue_filter(repo, number)),
        )

    def search(
        self, vector: list[float], limit: int = 10, filters: SearchFilters | None = None
    ) -> list[SearchHit]:
        """Nearest chunks to `vector`, among those the filters allow.

        The filters run inside Qdrant alongside the vector search, not on its
        results afterwards, so asking for ten closed issues returns ten rather
        than whichever of the top ten happened to be closed.
        """
        hits = self._client.query_points(
            COLLECTION,
            query=vector,
            query_filter=_search_filter(filters or {}),
            limit=limit,
            with_payload=True,
        ).points
        return [
            {**_stored_from_payload(hit.payload), "score": hit.score}
            for hit in hits
            if hit.payload is not None
        ]

    def stats(self) -> dict[str, RepoStats]:
        """Chunks and issues per repo, for `gh-issue status`.

        Two facet queries, however large the index: one counts chunks per repo,
        the other counts body chunks, and every issue has exactly one of those.
        """
        chunks = self._count_by_repo(None)
        issues = self._count_by_repo(_match("kind", "body"))
        return {
            repo: {"chunks": count, "issues": issues.get(repo, 0)} for repo, count in chunks.items()
        }

    def _count_by_repo(self, where: models.Filter | None) -> dict[str, int]:
        hits = self._client.facet(
            COLLECTION, key="repo", facet_filter=where, limit=_MAX_REPOS, exact=True
        ).hits
        return {str(hit.value): hit.count for hit in hits}


def _match(key: str, value: str) -> models.Filter:
    return models.Filter(
        must=[models.FieldCondition(key=key, match=models.MatchValue(value=value))]
    )


def _since(value: str) -> datetime.datetime:
    """A date or a moment, read as UTC when it names no zone."""
    try:
        moment = datetime.datetime.fromisoformat(value)
    except ValueError:
        raise ValueError(
            f"since must be a date like 2026-01-31 or a time like 2026-01-31T09:00:00Z, "
            f"not {value!r}"
        ) from None
    return moment if moment.tzinfo else moment.replace(tzinfo=datetime.UTC)


_EXACT_MATCH: tuple[Literal["state", "author", "kind"], ...] = ("state", "author", "kind")


def _search_filter(filters: SearchFilters) -> models.Filter | None:
    conditions: list[models.Condition] = []
    if repos := filters.get("repos"):
        conditions.append(models.FieldCondition(key="repo", match=models.MatchAny(any=repos)))
    for key in _EXACT_MATCH:
        if value := filters.get(key):
            conditions.append(models.FieldCondition(key=key, match=models.MatchValue(value=value)))
    # One condition per label, so a chunk must carry every one of them.
    for label in filters.get("labels", []):
        conditions.append(models.FieldCondition(key="labels", match=models.MatchValue(value=label)))
    if since := filters.get("since"):
        conditions.append(
            models.FieldCondition(key="created_at", range=models.DatetimeRange(gte=_since(since)))
        )
    return models.Filter(must=conditions) if conditions else None
