"""The checks `worker` and `sync` run first: what would make every run fail,
found before the worker connects or `sync` queues anything."""

import http.server
import socket
import threading
from collections.abc import Callable
from typing import Any

import pytest

from gh_issue.interface.cli import preflight


@pytest.fixture
def ready(monkeypatch: pytest.MonkeyPatch) -> Callable[..., list[str]]:
    """Every check passing, so each test can break exactly one thing."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    monkeypatch.setenv("INNGEST_SIGNING_KEY", "signkey-test")
    monkeypatch.setenv("INNGEST_EVENT_KEY", "evtkey-test")
    monkeypatch.setattr(preflight, "index_problem", lambda: None)
    monkeypatch.setattr(preflight, "inngest_problem", lambda origin, production: None)
    return preflight.preflight


def test_preflight_passes_when_everything_is_in_place(ready: Callable[..., list[str]]) -> None:
    assert ready(production=True) == []


def test_a_missing_github_token_is_named(
    ready: Callable[..., list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN")
    (problem,) = ready(production=False)
    assert problem == "github.token is not set. Set it with: gh-issue config set github.token"


def test_inngest_cloud_needs_both_keys_and_the_dev_server_needs_neither(
    ready: Callable[..., list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("INNGEST_SIGNING_KEY")
    monkeypatch.delenv("INNGEST_EVENT_KEY")

    problems = ready(production=True)

    assert problems == [
        "inngest.signing_key is not set, and Inngest Cloud needs it. Set it with: "
        "gh-issue config set inngest.signing_key (or, to use a local dev server "
        "instead: gh-issue config set inngest.dev 1)",
        "inngest.event_key is not set, and Inngest Cloud needs it. Set it with: "
        "gh-issue config set inngest.event_key (or, to use a local dev server "
        "instead: gh-issue config set inngest.dev 1)",
    ]
    assert ready(production=False) == []


def test_a_problem_with_the_index_is_passed_on(
    ready: Callable[..., list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """What the problem says is storage.index_problem's to decide; tested there."""
    monkeypatch.setattr(preflight, "index_problem", lambda: "Qdrant is not reachable")

    assert ready(production=False) == ["Qdrant is not reachable"]


def test_inngest_out_of_reach_is_passed_on(
    ready: Callable[..., list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(preflight, "inngest_problem", lambda origin, production: "no Inngest")
    assert ready(production=False) == ["no Inngest"]


def closed_port() -> int:
    """A port nothing listens on: bound, noted, and let go."""
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
    return port


def test_a_dev_server_that_does_not_answer_is_named_with_its_setting() -> None:
    """Without this the worker waits in silence: the SDK keeps trying to
    connect and says nothing while it does."""
    origin = f"http://127.0.0.1:{closed_port()}"
    assert preflight.inngest_problem(origin, production=False) == (
        f"Can't connect to the Inngest dev server at {origin} (inngest.dev)."
    )


def test_inngest_cloud_out_of_reach_is_named_too() -> None:
    origin = f"http://127.0.0.1:{closed_port()}"
    assert preflight.inngest_problem(origin, production=True) == (
        f"Can't connect to Inngest Cloud at {origin}."
    )


def test_any_answer_from_inngest_will_do() -> None:
    """Even a 404: all the check is for is that something is there."""

    class Answering(http.server.BaseHTTPRequestHandler):
        def do_GET(self) -> None:
            self.send_response(404)
            self.end_headers()

        def log_message(self, *args: Any) -> None:
            pass

    server = http.server.HTTPServer(("127.0.0.1", 0), Answering)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        origin = f"http://127.0.0.1:{server.server_address[1]}"
        assert preflight.inngest_problem(origin, production=False) is None
    finally:
        server.shutdown()
        server.server_close()
