import threading
from collections.abc import Iterator
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
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
    assert opened == [{"dim": DIM, "url": "http://qdrant.internal:6333", "api_key": None}]


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


# A Qdrant that wants a key -----------------------------------------------------


@pytest.fixture
def locked_qdrant(monkeypatch: pytest.MonkeyPatch) -> Iterator[list[str | None]]:
    """A server that refuses every request the way a key-protected Qdrant
    does, and keeps the key each request carried. Seen on Qdrant 1.19.1
    started with an API key: 401 and one of these two lines, for no key and for
    a wrong one. qdrant.url points at it for the test."""
    keys: list[str | None] = []

    class Refusing(BaseHTTPRequestHandler):
        def refuse(self) -> None:
            key = self.headers.get("api-key")
            keys.append(key)
            said = (
                b"Invalid API key or JWT"
                if key
                else b"Must provide an API key or an Authorization bearer token"
            )
            self.send_response(401)
            self.send_header("Content-Length", str(len(said)))
            self.end_headers()
            self.wfile.write(said)

        do_GET = do_PUT = do_POST = do_DELETE = refuse

        def log_message(self, format: str, *args: object) -> None:
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Refusing)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    monkeypatch.setenv("QDRANT_URL", f"http://127.0.0.1:{server.server_port}")
    yield keys
    server.shutdown()
    server.server_close()


# qdrant-client warns about two things here, and both are the point of the
# test server: it refuses the version check like every other request, and it is
# plain http, where a real hosted Qdrant is https.
quiet = pytest.mark.filterwarnings(
    "ignore:Failed to obtain server version", "ignore:Api key is used with an insecure connection"
)


@quiet
def test_a_qdrant_that_wants_a_key_says_which_setting_holds_it(
    locked_qdrant: list[str | None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """A hosted Qdrant with no key set. "Can't connect" would be wrong: it
    answered."""
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)

    problem = storage.index_problem()

    assert problem is not None
    assert problem.startswith("Qdrant at http://127.0.0.1:")
    assert problem.endswith(" needs an API key. Set it with: gh-issue config set qdrant.api_key")


@quiet
def test_a_key_that_qdrant_refuses_is_reported_as_the_key(
    locked_qdrant: list[str | None], monkeypatch: pytest.MonkeyPatch
) -> None:
    """The key in the setting is the one sent, and a refusal names it."""
    monkeypatch.setenv("QDRANT_API_KEY", "a-key-that-is-wrong")

    problem = storage.index_problem()

    assert "a-key-that-is-wrong" in locked_qdrant
    assert problem is not None
    assert problem.startswith("Qdrant at http://127.0.0.1:")
    assert problem.endswith(" refused the API key (qdrant.api_key).")
