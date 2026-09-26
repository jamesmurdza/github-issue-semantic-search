from collections.abc import Iterator
from pathlib import Path

import pytest

from gh_issue import settings
from gh_issue.embed import DIM
from gh_issue.functions.client import client


@pytest.fixture(autouse=True)
def clean_caches() -> Iterator[None]:
    # Held on to, because a test may still have a stand-in in its place when
    # this runs again at the end.
    store = settings.store
    store.cache_clear()
    yield
    store.cache_clear()


def test_the_three_event_names_are_distinct() -> None:
    """A copy-paste here is not a typo, it is an infinite loop: the backfill
    sends itself its own trigger, so it must not share a name with anything."""
    names = {
        settings.EVENT_ISSUE_CHANGED,
        settings.EVENT_BACKFILL_PAGE,
        settings.EVENT_SWEEP_REQUESTED,
    }
    assert len(names) == 3


def test_qdrant_url_defaults_to_the_compose_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)
    assert settings.qdrant_url() == "http://localhost:6333"


def test_qdrant_url_honours_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.internal:6333")
    assert settings.qdrant_url() == "http://qdrant.internal:6333"


def test_the_records_live_in_the_gh_issue_home(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Beside the config file, so every command finds them from any directory."""
    monkeypatch.setenv("GH_ISSUE_HOME", str(tmp_path))
    assert settings.data_dir() == str(tmp_path)


def test_the_inngest_app_is_named() -> None:
    assert client.app_id == "gh-issue"


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

    monkeypatch.setattr(settings, "Store", Opening)
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.internal:6333")

    assert settings.store() is settings.store()
    assert opened == [{"dim": DIM, "url": "http://qdrant.internal:6333"}]


def test_an_index_that_opens_has_no_problem(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(settings, "store", lambda: None)
    assert settings.index_problem() is None


def test_qdrant_out_of_reach_says_where_it_looked_and_which_setting_that_is(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Whatever runs Qdrant is the user's business; what they need to know is
    the address that failed and the setting that holds it."""

    def unreachable() -> None:
        raise ConnectionError("refused")

    monkeypatch.setattr(settings, "store", unreachable)
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.internal:6333")

    assert settings.index_problem() == (
        "Can't connect to Qdrant at http://qdrant.internal:6333 (qdrant.url)."
    )


def test_a_collection_of_the_wrong_size_is_reported_as_itself(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Qdrant is reachable here; saying it is not would send you the wrong way."""

    def wrong_size() -> None:
        raise ValueError("The Qdrant collection gh_issue_chunks holds 4-dimension vectors")

    monkeypatch.setattr(settings, "store", wrong_size)

    problem = settings.index_problem()
    assert problem is not None
    assert "4-dimension" in problem
    assert "not reachable" not in problem


# Whether a repo is synced here ----------------------------------------------------


@pytest.mark.usefixtures("data_dir")
def test_a_repo_nothing_was_recorded_for_is_not_synced() -> None:
    assert settings.is_synced("owner/name") is False


@pytest.mark.usefixtures("data_dir")
def test_a_sweep_cursor_alone_means_synced() -> None:
    settings.cursors().set("owner/name", "2026-09-25T10:00:00Z")
    assert settings.is_synced("owner/name") is True
    assert settings.is_synced("other/repo") is False


@pytest.mark.usefixtures("data_dir")
def test_a_backfill_record_alone_means_synced() -> None:
    settings.backfills().set(
        {
            "repo": "owner/name",
            "chain": "c1",
            "next": None,
            "finished": False,
            "at": "2026-09-25T10:00:00Z",
        }
    )
    assert settings.is_synced("owner/name") is True
