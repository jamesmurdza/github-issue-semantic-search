"""The MCP server, through a real MCP client.

Most tests talk to the server in memory, which is the same protocol without a
process in between, over a real embedded Qdrant and the real embedding model.
The last two start the server the way an MCP client would, over stdio and over
streamable HTTP, and list its tools: that is what shows the transports work.
"""

import asyncio
import os
import socket
import subprocess
import sys
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest
from fastmcp import Client
from fastmcp.client.transports import StdioTransport

from gh_issue.github.types import Issue
from gh_issue.index import query, storage
from gh_issue.index.chunking import chunks_for
from gh_issue.index.embed import DIM, embed_chunks
from gh_issue.index.records.backfills import Backfills
from gh_issue.index.records.cursors import Cursors
from gh_issue.index.records.failures import Failures
from gh_issue.index.records.heartbeat import WorkerStatus
from gh_issue.index.store import Store
from gh_issue.interface import mcp

REPO = "owner/name"
NO_WORKER: WorkerStatus = {
    "running": False,
    "connected": False,
    "state": None,
    "mode": None,
    "last_seen": None,
}


def issue(number: int, title: str, body: str, *, state: str = "open") -> Issue:
    return {
        "number": number,
        "title": title,
        "body": body,
        "state": state,
        "labels": [{"name": "bug"}],
        "user": {"login": "reporter"},
        "created_at": "2026-01-01T00:00:00Z",
        "comments": 0,
    }


FREEZE = issue(412, "App freezes when I scroll fast", "Happens on long lists.")
LOGIN = issue(
    7, "Login fails after the token expires", "Getting a 401 every morning.", state="closed"
)


@pytest.fixture
def indexed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Store]:
    store = Store(dim=DIM, path=str(tmp_path / "qdrant"))
    for item in (FREEZE, LOGIN):
        store.upsert(embed_chunks(chunks_for(REPO, item, [])))
    monkeypatch.setattr(query, "store", lambda: store)
    monkeypatch.setattr(query, "cursors", lambda: Cursors(str(tmp_path / "cursors")))
    monkeypatch.setattr(query, "backfills", lambda: Backfills(str(tmp_path / "backfills")))
    monkeypatch.setattr(query, "failures", lambda: Failures(str(tmp_path / "failed")))
    monkeypatch.setattr(query, "worker_status", lambda: NO_WORKER)
    monkeypatch.setattr(storage, "index_problem", lambda: None)
    yield store
    store.close()


def call(tool: str, arguments: dict[str, Any] | None = None) -> Any:
    """Call a tool in memory; returns the result, error or not."""

    async def go() -> Any:
        async with Client(mcp.server) as client:
            return await client.call_tool(tool, arguments or {}, raise_on_error=False)

    return asyncio.run(go())


def error_text(result: Any) -> str:
    assert result.is_error, "expected the tool to report an error"
    return " ".join(getattr(part, "text", "") for part in result.content)


# The tools ------------------------------------------------------------------------


def test_the_server_offers_three_read_only_tools() -> None:
    async def go() -> Any:
        async with Client(mcp.server) as client:
            return await client.list_tools()

    tools = {tool.name: tool for tool in asyncio.run(go())}

    assert sorted(tools) == ["get_issue", "search_issues", "status"]
    for tool in tools.values():
        assert tool.annotations is not None
        assert tool.annotations.read_only_hint is True
        assert tool.description


def test_search_takes_every_filter_the_cli_does() -> None:
    async def go() -> Any:
        async with Client(mcp.server) as client:
            return await client.list_tools()

    (search,) = [tool for tool in asyncio.run(go()) if tool.name == "search_issues"]
    properties = search.input_schema["properties"]

    assert sorted(properties) == sorted(
        ["query", "repos", "state", "labels", "author", "kind", "since", "limit"]
    )
    assert search.input_schema["required"] == ["query"]
    assert properties["state"]["anyOf"][0]["enum"] == ["open", "closed"]
    assert properties["kind"]["anyOf"][0]["enum"] == ["body", "comment"]


def test_search_issues_finds_by_meaning(indexed: Store) -> None:
    result = call("search_issues", {"query": "the page locks up when I swipe through a feed"})

    hits = result.structured_content["result"]
    assert hits[0]["issue_number"] == 412
    assert hits[0]["title"] == "App freezes when I scroll fast"


def test_search_issues_applies_filters(indexed: Store) -> None:
    result = call(
        "search_issues",
        {"query": "the page locks up when I swipe through a feed", "state": "closed", "limit": 5},
    )

    assert [hit["issue_number"] for hit in result.structured_content["result"]] == [7]


def test_a_bad_filter_comes_back_as_a_message_the_agent_can_act_on(indexed: Store) -> None:
    result = call("search_issues", {"query": "anything", "since": "last tuesday"})

    assert "since must be a date like 2026-01-31" in error_text(result)


def test_get_issue_returns_the_thread(indexed: Store) -> None:
    result = call("get_issue", {"repo": REPO, "number": 412})

    thread = result.structured_content
    assert thread["title"] == "App freezes when I scroll fast"
    assert thread["url"] == "https://github.com/owner/name/issues/412"
    assert thread["posts"][0]["text"] == "Happens on long lists."


def test_get_issue_of_something_not_indexed_says_where_to_look(indexed: Store) -> None:
    text = error_text(call("get_issue", {"repo": REPO, "number": 999}))

    assert "owner/name#999" in text
    assert "status" in text


def test_status_lists_the_repos_and_the_worker(indexed: Store) -> None:
    result = call("status")

    status = result.structured_content
    assert status["worker"] == NO_WORKER
    assert [(r["repo"], r["issues"]) for r in status["repos"]] == [(REPO, 2)]


@pytest.mark.parametrize(
    ("tool", "arguments"),
    [
        ("search_issues", {"query": "anything"}),
        ("get_issue", {"repo": REPO, "number": 1}),
        ("status", {}),
    ],
)
def test_every_tool_says_what_is_wrong_with_the_index(
    monkeypatch: pytest.MonkeyPatch, tool: str, arguments: dict[str, Any]
) -> None:
    monkeypatch.setattr(storage, "index_problem", lambda: "Qdrant is not reachable")

    assert "Qdrant is not reachable" in error_text(call(tool, arguments))


# The transports, for real ------------------------------------------------------------

SERVE = [sys.executable, "-c", "from gh_issue.interface.cli.app import main; main()", "mcp"]
PROJECT = Path(__file__).resolve().parent.parent


def list_tool_names(client: Client[Any]) -> list[str]:
    async def go() -> list[str]:
        async with client:
            return sorted(tool.name for tool in await client.list_tools())

    return asyncio.run(go())


def test_over_stdio() -> None:
    """Also shows nothing else writes to stdout, which is the protocol's channel."""
    transport = StdioTransport(
        command=SERVE[0], args=[*SERVE[1:], "stdio"], cwd=str(PROJECT), env=dict(os.environ)
    )
    assert list_tool_names(Client(transport)) == ["get_issue", "search_issues", "status"]


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port: int = probe.getsockname()[1]
        return port


def test_over_streamable_http() -> None:
    port = free_port()
    server = subprocess.Popen(
        [*SERVE, "http", "--port", str(port)],
        cwd=PROJECT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        deadline = time.time() + 30
        while True:
            try:
                socket.create_connection(("127.0.0.1", port), timeout=0.5).close()
                break
            except OSError:
                if time.time() > deadline or server.poll() is not None:
                    raise AssertionError("the HTTP server never started") from None
                time.sleep(0.2)
        url = f"http://127.0.0.1:{port}/mcp"
        assert list_tool_names(Client(url)) == ["get_issue", "search_issues", "status"]
    finally:
        server.terminate()
        server.wait(timeout=10)
