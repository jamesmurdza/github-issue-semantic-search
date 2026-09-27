from collections.abc import Iterator
from pathlib import Path

import pytest

from gh_issue.index import storage
from gh_issue.index.embed import DIM


@pytest.fixture(autouse=True)
def clean_caches() -> Iterator[None]:
    # Held on to, because a test may still have a stand-in in its place when
    # this runs again at the end.
    store = storage.store
    store.cache_clear()
    yield
    store.cache_clear()


def test_the_records_live_in_the_gh_issue_home(data_dir: Path) -> None:
    """Beside the config file, so every command finds them from any directory."""
    storage.cursors().set("owner/name", "2026-09-25T10:00:00Z")
    storage.backfills().set(
        {"repo": "owner/name", "chain": "c1", "next": None, "finished": True, "at": "t"}
    )
    storage.failures().add(
        {"repo": "owner/name", "number": 9, "error": "boom", "run_id": "r", "at": "t"}
    )

    assert sorted(path.name for path in data_dir.iterdir()) == ["backfills", "cursors", "failed"]


def test_the_store_is_opened_once_and_reused(monkeypatch: pytest.MonkeyPatch) -> None:
    """One process, one connection. Every function reaches for this.

    A stand-in for the store, so this neither needs a Qdrant running nor opens
    the index that lives in one — it used to, and failed the day that index's
    collection had been made with the wrong vector size.
    """
    opened: list[dict[str, object]] = []

    class Opening:
        def __init__(self, **options: object) -> None:
            opened.append(options)

    monkeypatch.setattr(storage, "Store", Opening)
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.internal:6333")

    assert storage.store() is storage.store()
    assert opened == [{"dim": DIM, "url": "http://qdrant.internal:6333"}]


def test_an_index_that_opens_has_no_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(storage, "store", lambda: None)
    assert storage.index_problem() is None


def test_qdrant_out_of_reach_says_where_it_looked_and_which_setting_that_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever runs Qdrant is the user's business; what they need to know is
    the address that failed and the setting that holds it."""

    def unreachable() -> None:
        raise ConnectionError("refused")

    monkeypatch.setattr(storage, "store", unreachable)
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.internal:6333")

    assert storage.index_problem() == (
        "Can't connect to Qdrant at http://qdrant.internal:6333 (qdrant.url)."
    )


def test_a_collection_of_the_wrong_size_is_reported_as_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qdrant is reachable here; saying it is not would send you the wrong way."""

    def wrong_size() -> None:
        raise ValueError("The Qdrant collection gh_issue_chunks holds 4-dimension vectors")

    monkeypatch.setattr(storage, "store", wrong_size)

    problem = storage.index_problem()
    assert problem is not None
    assert "4-dimension" in problem
    assert "not reachable" not in problem


# Whether a repo is synced here ----------------------------------------------------


@pytest.mark.usefixtures("data_dir")
def test_a_repo_nothing_was_recorded_for_is_not_synced() -> None:
    assert storage.is_synced("owner/name") is False


@pytest.mark.usefixtures("data_dir")
def test_a_sweep_cursor_alone_means_synced() -> None:
    storage.cursors().set("owner/name", "2026-09-25T10:00:00Z")
    assert storage.is_synced("owner/name") is True
    assert storage.is_synced("other/repo") is False


@pytest.mark.usefixtures("data_dir")
def test_a_backfill_record_alone_means_synced() -> None:
    storage.backfills().set(
        {
            "repo": "owner/name",
            "chain": "c1",
            "next": None,
            "finished": False,
            "at": "2026-09-25T10:00:00Z",
        }
    )
    assert storage.is_synced("owner/name") is True
