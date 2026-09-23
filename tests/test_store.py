from collections.abc import Iterator
from pathlib import Path

import pytest

from gh_issue.chunking import EmbeddedChunk
from gh_issue.index.store import Store

DIM = 4


@pytest.fixture
def store(tmp_path: Path) -> Iterator[Store]:
    """A real embedded Qdrant in a throwaway directory. Nothing here is mocked.

    Embedded mode locks its directory, so the store has to be closed before the
    next test opens its own.
    """
    opened = Store(dim=DIM, path=str(tmp_path))
    yield opened
    opened.close()


def test_store_needs_exactly_one_of_url_or_path() -> None:
    with pytest.raises(ValueError, match="exactly one"):
        Store(dim=DIM)
    with pytest.raises(ValueError, match="exactly one"):
        Store(dim=DIM, url="http://localhost:6333", path="./somewhere")


def _chunk(chunk_id: str, content_hash: str, number: int = 412) -> EmbeddedChunk:
    return {
        "id": chunk_id,
        "content_hash": content_hash,
        "text": "some text",
        "meta": {
            "repo": "owner/name",
            "issue_number": number,
            "state": "open",
            "labels": ["bug"],
            "kind": "body",
            "author": "reporter",
            "created_at": "2026-01-01T00:00:00Z",
        },
        "vector": [0.1, 0.2, 0.3, 0.4],
    }


def test_hashes_for_issue_is_empty_before_anything_is_written(store: Store) -> None:
    assert store.hashes_for_issue("owner/name", 412) == {}


def test_upsert_then_read_back_hashes(store: Store) -> None:
    store.upsert([_chunk("owner/name#412:body", "h1")])
    assert store.hashes_for_issue("owner/name", 412) == {"owner/name#412:body": "h1"}


def test_upsert_is_idempotent_on_the_same_id(store: Store) -> None:
    """Re-indexing an edited chunk must overwrite, not add a second copy."""
    store.upsert([_chunk("owner/name#412:body", "h1")])
    store.upsert([_chunk("owner/name#412:body", "h2")])
    assert store.hashes_for_issue("owner/name", 412) == {"owner/name#412:body": "h2"}


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
    assert list(store.hashes_for_issue("owner/name", 412)) == ["owner/name#412:body"]


def test_delete_by_issue_leaves_other_issues_alone(store: Store) -> None:
    store.upsert(
        [
            _chunk("owner/name#412:body", "h1"),
            _chunk("owner/name#413:body", "h3", number=413),
        ]
    )
    store.delete_by_issue("owner/name", 412)
    assert store.hashes_for_issue("owner/name", 412) == {}
    assert store.hashes_for_issue("owner/name", 413) != {}


def test_hashes_are_scoped_to_one_repo(store: Store) -> None:
    """Two repos can hold the same issue number without colliding."""
    other = _chunk("other/repo#412:body", "h9")
    other["meta"]["repo"] = "other/repo"
    store.upsert([_chunk("owner/name#412:body", "h1"), other])
    assert store.hashes_for_issue("owner/name", 412) == {"owner/name#412:body": "h1"}
    assert store.hashes_for_issue("other/repo", 412) == {"other/repo#412:body": "h9"}


def test_search_returns_the_stored_text_and_metadata(store: Store) -> None:
    store.upsert([_chunk("owner/name#412:body", "h1")])
    hits = store.search([0.1, 0.2, 0.3, 0.4], limit=5)
    assert hits[0]["id"] == "owner/name#412:body"
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
