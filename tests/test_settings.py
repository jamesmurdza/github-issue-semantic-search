from collections.abc import Iterator

import inngest
import pytest

from gh_issue import settings
from gh_issue.embed import DIM
from gh_issue.functions.client import client


@pytest.fixture(autouse=True)
def clean_caches() -> Iterator[None]:
    settings.store.cache_clear()
    yield
    settings.store.cache_clear()


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


def test_data_dir_defaults_beside_the_project(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GH_ISSUE_DATA_DIR", raising=False)
    assert settings.data_dir() == "./.gh-issue"


def test_the_inngest_app_is_named() -> None:
    assert client.app_id == "gh-issue"


def test_the_client_talks_to_cloud_unless_inngest_dev_is_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """`is_production` is deliberately not passed. Leaving it to the SDK is what
    lets the same code run against the dev server locally and cloud in
    production, by setting INNGEST_DEV rather than editing the source."""
    monkeypatch.delenv("INNGEST_DEV", raising=False)
    cloud = inngest.Inngest(app_id="probe")
    assert cloud.is_production is True
    assert "api.inngest.com" in cloud.api_origin

    monkeypatch.setenv("INNGEST_DEV", "1")
    dev = inngest.Inngest(app_id="probe")
    assert dev.is_production is False
    assert "8288" in dev.api_origin


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


def test_qdrant_not_running_says_how_to_start_it(monkeypatch: pytest.MonkeyPatch) -> None:
    def unreachable() -> None:
        raise ConnectionError("refused")

    monkeypatch.setattr(settings, "store", unreachable)

    problem = settings.index_problem()
    assert problem is not None
    assert "Qdrant" in problem
    assert "docker compose up -d" in problem


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
