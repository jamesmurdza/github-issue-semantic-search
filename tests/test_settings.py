import os
from collections.abc import Iterator

import httpx
import inngest
import pytest

from gh_issue import settings
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


def _server_reachable() -> bool:
    try:
        httpx.get(os.environ.get("QDRANT_URL", "http://localhost:6333") + "/readyz", timeout=1.0)
    except httpx.HTTPError:
        return False
    return True


@pytest.mark.skipif(not _server_reachable(), reason="needs the Qdrant from docker-compose")
def test_the_store_is_opened_once_and_reused() -> None:
    """One process, one connection. Every function reaches for this."""
    assert settings.store() is settings.store()
