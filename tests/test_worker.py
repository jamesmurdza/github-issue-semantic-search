"""The worker: the long-running process that executes the Inngest functions.

The connection itself is the SDK's and is exercised against the dev server; here
it is a fake, and what is tested is everything around it: the heartbeat that
`status` and `sync` read, the checks that stop a worker starting in a state
where every run would fail, and the shutdown that cleans up after it.
"""

import asyncio
import datetime
import os
import socket
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from inngest.connect import ConnectionState

from gh_issue import worker
from gh_issue.index import repo_files
from gh_issue.worker import Beat, Heartbeat, is_fresh, read_heartbeat

States = ConnectionState


class FakeClock:
    """Monotonic seconds for scheduling, and a wall clock for what gets written."""

    def __init__(self) -> None:
        self.seconds = 0.0

    def monotonic(self) -> float:
        return self.seconds

    def wall(self) -> str:
        return f"2026-09-25T10:00:{int(self.seconds):02d}Z"


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> FakeClock:
    fake = FakeClock()
    monkeypatch.setattr(worker, "now", fake.wall)
    return fake


def make_beat(path: Path, clock: FakeClock, said: list[str]) -> Beat:
    return Beat(path, "dev", clock=clock.monotonic, say=said.append)


# The heartbeat ------------------------------------------------------------------


def test_the_first_tick_writes_a_heartbeat(tmp_path: Path, clock: FakeClock) -> None:
    path = tmp_path / "worker.json"
    make_beat(path, clock, [])(States.CONNECTING)

    assert read_heartbeat(path) == {
        "pid": os.getpid(),
        "host": socket.gethostname(),
        "mode": "dev",
        "state": "CONNECTING",
        "started_at": "2026-09-25T10:00:00Z",
        "beat_at": "2026-09-25T10:00:00Z",
    }


def test_it_is_rewritten_on_a_state_change_and_every_ten_seconds_otherwise(
    tmp_path: Path, clock: FakeClock, monkeypatch: pytest.MonkeyPatch
) -> None:
    writes: list[str] = []
    real_write = repo_files.write_json

    def counting_write(path: Path, record: Any) -> None:
        writes.append(f"{record['state']}@{int(clock.seconds)}")
        real_write(path, record)

    monkeypatch.setattr(worker, "write_json", counting_write)
    beat = make_beat(tmp_path / "worker.json", clock, [])

    for second, state in [
        (0, States.CONNECTING),
        (1, States.CONNECTING),
        (2, States.ACTIVE),
        (5, States.ACTIVE),
        (11, States.ACTIVE),
        (12, States.ACTIVE),
        (13, States.RECONNECTING),
    ]:
        clock.seconds = second
        beat(state)

    assert writes == ["CONNECTING@0", "ACTIVE@2", "ACTIVE@12", "RECONNECTING@13"]


def test_each_state_change_is_said_once(tmp_path: Path, clock: FakeClock) -> None:
    said: list[str] = []
    beat = make_beat(tmp_path / "worker.json", clock, said)

    for state in [States.CONNECTING, States.ACTIVE, States.ACTIVE, States.RECONNECTING]:
        beat(state)

    assert len(said) == 3
    assert "Connected" in said[1]
    assert "reconnecting" in said[2].lower()


def test_stop_removes_the_heartbeat(tmp_path: Path, clock: FakeClock) -> None:
    """So `status` says the worker stopped straight away, rather than after its
    heartbeat goes stale."""
    path = tmp_path / "worker.json"
    beat = make_beat(path, clock, [])
    beat(States.ACTIVE)

    beat.stop()

    assert read_heartbeat(path) is None


def heartbeat(beat_at: str) -> Heartbeat:
    return {
        "pid": 1,
        "host": "h",
        "mode": "dev",
        "state": "ACTIVE",
        "started_at": "2026-09-25T09:00:00Z",
        "beat_at": beat_at,
    }


def test_no_heartbeat_means_no_worker(tmp_path: Path) -> None:
    at = datetime.datetime(2026, 9, 25, 10, 0, tzinfo=datetime.UTC)
    assert read_heartbeat(tmp_path / "worker.json") is None
    assert not is_fresh(None, at)


def test_a_heartbeat_goes_stale_after_thirty_seconds() -> None:
    """A worker killed outright never removes its file; the age is what tells."""
    at = datetime.datetime(2026, 9, 25, 10, 0, 30, tzinfo=datetime.UTC)
    assert is_fresh(heartbeat("2026-09-25T10:00:05Z"), at)
    assert not is_fresh(heartbeat("2026-09-25T09:59:59Z"), at)


# Serving -------------------------------------------------------------------------


class FakeConnection:
    """Starts CONNECTING, blocks in start() until closed, like the SDK's."""

    def __init__(self) -> None:
        self.state = ConnectionState.CONNECTING
        self.closed = asyncio.Event()

    def get_state(self) -> ConnectionState:
        return self.state

    async def start(self) -> None:
        await self.closed.wait()
        self.state = ConnectionState.CLOSED


def test_serve_ticks_while_connected_and_stops_when_the_connection_closes() -> None:
    seen: list[ConnectionState] = []

    async def scenario() -> int:
        connection = FakeConnection()
        serving = asyncio.create_task(worker.serve(connection, seen.append, tick=0.01))
        await asyncio.sleep(0.05)
        connection.state = ConnectionState.ACTIVE
        await asyncio.sleep(0.05)
        connection.closed.set()
        await asyncio.wait_for(serving, timeout=1)
        ticks_at_close = len(seen)
        await asyncio.sleep(0.05)
        return ticks_at_close

    ticks_at_close = asyncio.run(scenario())

    assert ConnectionState.CONNECTING in seen
    assert ConnectionState.ACTIVE in seen
    assert len(seen) == ticks_at_close, "kept ticking after the connection closed"


class BriefConnection(FakeConnection):
    """Connects, then closes almost at once — as after a Ctrl+C, which the SDK
    turns into a graceful close that makes start() return."""

    def __init__(self, heartbeat: Path) -> None:
        super().__init__()
        self.state = ConnectionState.ACTIVE
        self.heartbeat = heartbeat
        self.heartbeat_while_up = False

    async def start(self) -> None:
        await asyncio.sleep(0.05)
        self.heartbeat_while_up = self.heartbeat.exists()
        self.state = ConnectionState.CLOSED


def test_run_keeps_a_heartbeat_while_up_and_removes_it_on_the_way_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("GH_ISSUE_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(worker, "preflight", lambda production: [])
    connection = BriefConnection(tmp_path / "worker.json")
    monkeypatch.setattr(worker, "connect", lambda apps: connection)
    monkeypatch.setattr(worker, "TICK", 0.01)

    worker.run(say=lambda _: None)

    assert connection.heartbeat_while_up
    assert not (tmp_path / "worker.json").exists()


def test_run_refuses_to_start_when_preflight_finds_problems(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(worker, "preflight", lambda production: ["GITHUB_TOKEN is not set."])

    def never(apps: Any) -> FakeConnection:
        raise AssertionError("connected despite failing preflight")

    monkeypatch.setattr(worker, "connect", never)

    with pytest.raises(SystemExit, match="GITHUB_TOKEN is not set"):
        worker.run(say=lambda _: None)


def test_the_worker_serves_every_function() -> None:
    assert sorted(fn.local_id for fn in worker.FUNCTIONS) == [
        "backfill-page",
        "index-issue",
        "reconcile",
        "sweep-repo",
    ]


# Preflight ---------------------------------------------------------------------


@pytest.fixture
def ready(monkeypatch: pytest.MonkeyPatch) -> Callable[..., list[str]]:
    """Every check passing, so each test can break exactly one thing."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_test")
    monkeypatch.setenv("INNGEST_SIGNING_KEY", "signkey-test")
    monkeypatch.setenv("INNGEST_EVENT_KEY", "evtkey-test")
    monkeypatch.setattr(worker, "store", lambda: None)
    return worker.preflight


def test_preflight_passes_when_everything_is_in_place(ready: Callable[..., list[str]]) -> None:
    assert ready(production=True) == []


def test_a_missing_github_token_is_named(
    ready: Callable[..., list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("GITHUB_TOKEN")
    (problem,) = ready(production=False)
    assert "GITHUB_TOKEN" in problem
    assert ".env" in problem


def test_inngest_cloud_needs_both_keys_and_the_dev_server_needs_neither(
    ready: Callable[..., list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("INNGEST_SIGNING_KEY")
    monkeypatch.delenv("INNGEST_EVENT_KEY")

    problems = ready(production=True)

    assert len(problems) == 2
    assert any("INNGEST_SIGNING_KEY" in p for p in problems)
    assert any("INNGEST_EVENT_KEY" in p for p in problems)
    assert all("INNGEST_DEV=1" in p for p in problems)
    assert ready(production=False) == []


def test_qdrant_not_running_says_how_to_start_it(
    ready: Callable[..., list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    def unreachable() -> None:
        raise ConnectionError("refused")

    monkeypatch.setattr(worker, "store", unreachable)

    (problem,) = ready(production=False)
    assert "Qdrant" in problem
    assert "docker compose up -d" in problem


def test_a_collection_of_the_wrong_size_is_reported_as_itself(
    ready: Callable[..., list[str]], monkeypatch: pytest.MonkeyPatch
) -> None:
    """Qdrant is reachable here; saying it is not would send you the wrong way."""

    def wrong_size() -> None:
        raise ValueError("The Qdrant collection gh_issue_chunks holds 4-dimension vectors")

    monkeypatch.setattr(worker, "store", wrong_size)

    (problem,) = ready(production=False)
    assert "4-dimension" in problem
    assert "not reachable" not in problem
