import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from gh_issue.chunking import EmbeddedChunk
from gh_issue.index.store import COLLECTION, SearchFilters, Store

DIM = 4

# Set this to run the same tests against the Qdrant that ships, rather than the
# embedded one. Off by default so `uv run pytest` needs no daemon.
#   GH_ISSUE_TEST_QDRANT_URL=http://localhost:6333 uv run pytest tests/test_store.py
SERVER_URL = os.environ.get("GH_ISSUE_TEST_QDRANT_URL")

# Never the real collection: the server may be the one holding your index, and
# these tests drop their collection before and after every test.
TEST_COLLECTION = f"{COLLECTION}_test"


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    """A real Qdrant. Nothing here is mocked either way.

    Embedded mode locks its directory, so the store has to be closed before the
    next test opens its own. Server mode shares one collection between tests, so
    each test drops it first and again when done.
    """
    if not SERVER_URL:
        opened = Store(dim=DIM, path=str(tmp_path), collection=TEST_COLLECTION)
        yield opened
        opened.close()
        return
    admin = QdrantClient(url=SERVER_URL)
    admin.delete_collection(TEST_COLLECTION)
    opened = Store(dim=DIM, url=SERVER_URL, collection=TEST_COLLECTION)
    yield opened
    opened.close()
    admin.delete_collection(TEST_COLLECTION)
    admin.close()


def test_a_store_only_touches_its_own_collection(tmp_path: Path) -> None:
    """What keeps these tests off a real index living in the same Qdrant."""
    where = {"url": SERVER_URL} if SERVER_URL else {"path": str(tmp_path)}
    first = Store(dim=DIM, collection=f"{TEST_COLLECTION}_first", **where)
    first.upsert([_chunk("owner/name#412:body", "h1")])
    first.close()

    second = Store(dim=DIM, collection=f"{TEST_COLLECTION}_second", **where)
    assert second.fingerprints_for_issue("owner/name", 412) == {}
    second.close()

    reopened = Store(dim=DIM, collection=f"{TEST_COLLECTION}_first", **where)
    assert list(reopened.fingerprints_for_issue("owner/name", 412)) == ["owner/name#412:body"]
    reopened.close()
    if SERVER_URL:
        admin = QdrantClient(url=SERVER_URL)
        admin.delete_collection(f"{TEST_COLLECTION}_first")
        admin.delete_collection(f"{TEST_COLLECTION}_second")
        admin.close()


def test_store_needs_exactly_one_of_url_or_path() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Store(dim=DIM)
    with pytest.raises(ValueError, match="exactly one"):
        Store(dim=DIM, url="http://localhost:6333", path="./somewhere")


VECTOR = [0.1, 0.2, 0.3, 0.4]


def _chunk(
    chunk_id: str,
    content_hash: str = "h",
    number: int = 412,
    *,
    repo: str = "owner/name",
    state: str = "open",
    labels: list[str] | None = None,
    author: str = "reporter",
    created_at: str = "2026-01-01T00:00:00Z",
) -> EmbeddedChunk:
    return {
        "id": chunk_id,
        "content_hash": content_hash,
        "text": "some text",
        "meta": {
            "title": "App freezes when I scroll fast",
            "state": state,
            "labels": ["bug"] if labels is None else labels,
            "repo": repo,
            "issue_number": number,
            "kind": "comment" if ":comment:" in chunk_id else "body",
            "author": author,
            "created_at": created_at,
        },
        "vector": VECTOR,
    }


def hashes(store: Store, number: int = 412, repo: str = "owner/name") -> dict[str, str]:
    """chunk id -> content hash, the part of a fingerprint most tests care about."""
    stored = store.fingerprints_for_issue(repo, number)
    return {chunk_id: fingerprint["content_hash"] for chunk_id, fingerprint in stored.items()}


def test_nothing_is_stored_before_anything_is_written(store: Store) -> None:
    assert store.fingerprints_for_issue("owner/name", 412) == {}


def test_upsert_then_read_back_hashes(store: Store) -> None:
    store.upsert([_chunk("owner/name#412:body", "h1")])
    assert hashes(store) == {"owner/name#412:body": "h1"}


def test_upsert_is_idempotent_on_the_same_id(store: Store) -> None:
    """Re-indexing an edited chunk must overwrite, not add a second copy."""
    store.upsert([_chunk("owner/name#412:body", "h1")])
    store.upsert([_chunk("owner/name#412:body", "h2")])
    assert hashes(store) == {"owner/name#412:body": "h2"}


def test_upsert_of_nothing_is_a_no_op(store: Store) -> None:
    assert store.upsert([]) == 0


def test_delete_many_removes_only_the_named_chunks(store: Store) -> None:
    store.upsert(
        [
            _chunk("owner/name#412:body", "h1"),
            _chunk("owner/name#412:comment:1", "h2"),
        ]
    )
    store.delete_many(["owner/name#412:comment:1"])
    assert list(hashes(store)) == ["owner/name#412:body"]


def test_delete_by_issue_leaves_other_issues_alone(store: Store) -> None:
    store.upsert(
        [
            _chunk("owner/name#412:body", "h1"),
            _chunk("owner/name#413:body", "h3", number=413),
        ]
    )
    store.delete_by_issue("owner/name", 412)
    assert hashes(store) == {}
    assert hashes(store, 413) != {}


def test_hashes_are_scoped_to_one_repo(store: Store) -> None:
    """Two repos can hold the same issue number without colliding."""
    store.upsert(
        [
            _chunk("owner/name#412:body", "h1"),
            _chunk("other/repo#412:body", "h9", repo="other/repo"),
        ]
    )
    assert hashes(store) == {"owner/name#412:body": "h1"}
    assert hashes(store, repo="other/repo") == {"other/repo#412:body": "h9"}


def test_search_returns_the_stored_text_and_metadata(store: Store) -> None:
    store.upsert([_chunk("owner/name#412:body", "h1")])
    hits = store.search(VECTOR, limit=5)
    assert hits[0]["id"] == "owner/name#412:body"
    assert hits[0]["title"] == "App freezes when I scroll fast"
    assert hits[0]["text"] == "some text"
    assert hits[0]["repo"] == "owner/name"
    assert hits[0]["labels"] == ["bug"]
    assert hits[0]["score"] == pytest.approx(1.0)


def test_chunks_for_issue_returns_the_whole_thread(store: Store) -> None:
    store.upsert(
        [
            _chunk("owner/name#412:body", "h1"),
            _chunk("owner/name#412:comment:1", "h2"),
        ]
    )
    chunks = store.chunks_for_issue("owner/name", 412)
    assert sorted(c["id"] for c in chunks) == [
        "owner/name#412:body",
        "owner/name#412:comment:1",
    ]
    assert chunks[0]["content_hash"] in ("h1", "h2")


def test_stats_counts_chunks_and_issues_per_repo(store: Store) -> None:
    store.upsert(
        [
            _chunk("owner/name#412:body", "h1"),
            _chunk("owner/name#412:comment:1", "h2"),
            _chunk("owner/name#413:body", "h3", number=413),
        ]
    )
    assert store.stats() == {"owner/name": {"chunks": 3, "issues": 2}}


def test_stats_covers_every_repo(store: Store) -> None:
    store.upsert(
        [
            _chunk("owner/name#412:body", "h1"),
            _chunk("other/repo#7:body", "h2", number=7, repo="other/repo"),
            _chunk("other/repo#7:comment:1", "h3", number=7, repo="other/repo"),
        ]
    )
    assert store.stats() == {
        "owner/name": {"chunks": 1, "issues": 1},
        "other/repo": {"chunks": 2, "issues": 1},
    }


def test_stats_of_an_empty_index_is_empty(store: Store) -> None:
    assert store.stats() == {}


# The issue metadata copied onto every chunk ---------------------------------


def test_a_fingerprint_is_the_hash_plus_the_issue_metadata(store: Store) -> None:
    store.upsert([_chunk("owner/name#412:body", "h1")])
    assert store.fingerprints_for_issue("owner/name", 412) == {
        "owner/name#412:body": {
            "content_hash": "h1",
            "title": "App freezes when I scroll fast",
            "state": "open",
            "labels": ["bug"],
        }
    }


def test_set_issue_meta_rewrites_every_copy_on_that_issue_only(store: Store) -> None:
    store.upsert(
        [
            _chunk("owner/name#412:body", "h1"),
            _chunk("owner/name#412:comment:1", "h2"),
            _chunk("owner/name#413:body", "h3", number=413),
        ]
    )

    store.set_issue_meta(
        "owner/name", 412, {"title": "Scrolling freezes", "state": "closed", "labels": ["p1"]}
    )

    thread = store.fingerprints_for_issue("owner/name", 412).values()
    assert [(f["title"], f["state"], f["labels"]) for f in thread] == [
        ("Scrolling freezes", "closed", ["p1"]),
        ("Scrolling freezes", "closed", ["p1"]),
    ]
    assert hashes(store) == {"owner/name#412:body": "h1", "owner/name#412:comment:1": "h2"}
    assert store.fingerprints_for_issue("owner/name", 413)["owner/name#413:body"]["state"] == "open"


# Search filters ---------------------------------------------------------------

A_BODY = "owner/name#1:body"
A_COMMENT = "owner/name#1:comment:10"
B_BODY = "owner/name#2:body"
OTHER = "other/repo#1:body"


@pytest.fixture
def searchable(store: Store) -> Store:
    """Four chunks, each differing from the others in a field a filter reads."""
    store.upsert(
        [
            _chunk(A_BODY, number=1, author="ann", created_at="2026-01-01T00:00:00Z"),
            _chunk(A_COMMENT, number=1, author="bob", created_at="2026-03-01T00:00:00Z"),
            _chunk(
                B_BODY,
                number=2,
                state="closed",
                labels=["bug", "p1"],
                author="ann",
                created_at="2026-02-01T00:00:00Z",
            ),
            _chunk(
                OTHER,
                number=1,
                repo="other/repo",
                labels=["docs"],
                author="cat",
                created_at="2026-02-15T00:00:00Z",
            ),
        ]
    )
    return store


@pytest.mark.parametrize(
    ("filters", "expected"),
    [
        pytest.param({}, [A_BODY, A_COMMENT, B_BODY, OTHER], id="none"),
        pytest.param({"repos": ["other/repo"]}, [OTHER], id="one-repo"),
        pytest.param(
            {"repos": ["owner/name", "other/repo"]},
            [A_BODY, A_COMMENT, B_BODY, OTHER],
            id="any-of-several-repos",
        ),
        pytest.param({"repos": []}, [A_BODY, A_COMMENT, B_BODY, OTHER], id="empty-means-all"),
        pytest.param({"state": "closed"}, [B_BODY], id="state"),
        pytest.param({"labels": ["bug"]}, [A_BODY, A_COMMENT, B_BODY], id="label"),
        pytest.param({"labels": ["bug", "p1"]}, [B_BODY], id="labels-are-all-of"),
        pytest.param({"author": "ann"}, [A_BODY, B_BODY], id="author"),
        pytest.param({"kind": "comment"}, [A_COMMENT], id="kind"),
        pytest.param({"since": "2026-02-01"}, [A_COMMENT, B_BODY, OTHER], id="since-a-date"),
        pytest.param({"since": "2026-02-15T00:00:01Z"}, [A_COMMENT], id="since-a-moment"),
        pytest.param(
            {"repos": ["owner/name"], "state": "open", "kind": "body"}, [A_BODY], id="combined"
        ),
    ],
)
def test_search_filters(searchable: Store, filters: SearchFilters, expected: list[str]) -> None:
    hits = searchable.search(VECTOR, limit=10, filters=filters)
    assert sorted(hit["id"] for hit in hits) == sorted(expected)


def test_a_since_that_is_not_a_date_is_refused(searchable: Store) -> None:
    with pytest.raises(ValueError, match="since"):
        searchable.search(VECTOR, filters={"since": "last tuesday"})


def test_stats_forgets_a_repo_once_its_chunks_are_gone(store: Store) -> None:
    """Qdrant's server keeps a deleted repo's value in its facet index with a
    count of zero; `status` must not list it."""
    store.upsert([_chunk("owner/name#412:body", "h1")])
    store.delete_by_issue("owner/name", 412)
    assert store.stats() == {}


def test_delete_repo_removes_that_repo_and_nothing_else(store: Store) -> None:
    store.upsert(
        [
            _chunk("owner/name#412:body", "h1"),
            _chunk("owner/name#413:comment:1", "h2", number=413),
            _chunk("other/repo#7:body", "h3", number=7, repo="other/repo"),
        ]
    )

    store.delete_repo("owner/name")

    assert store.stats() == {"other/repo": {"chunks": 1, "issues": 1}}


def test_a_collection_made_for_another_vector_size_is_refused_on_open(tmp_path: Path) -> None:
    """Otherwise every write fails with a 400 that Inngest retries for minutes,
    and the run just looks stuck — which is how this was found: a test run had
    made the real collection with 4-dimension vectors."""
    where = {"url": SERVER_URL} if SERVER_URL else {"path": str(tmp_path)}
    name = f"{TEST_COLLECTION}_size"
    Store(dim=DIM, collection=name, **where).close()
    try:
        with pytest.raises(ValueError, match=rf"{name}.*{DIM}.*{DIM * 2}"):
            Store(dim=DIM * 2, collection=name, **where)
    finally:
        if SERVER_URL:
            admin = QdrantClient(url=SERVER_URL)
            admin.delete_collection(name)
            admin.close()
