"""`gh-issue worker`: the process that runs the Inngest functions.

It opens an outbound connection to Inngest — the dev server or Inngest Cloud,
whichever INNGEST_DEV says — and runs whatever work Inngest hands it. Nothing
needs to reach in, so there is no port to open and no URL to register.

While it runs it keeps a heartbeat file, which is how `status` and `sync` know
whether anything is here to do the work they queue. The connection's own state
lives inside this process; the file is what other commands can see.
"""

import asyncio
import os
import socket
import time
from collections.abc import Callable
from contextlib import suppress
from pathlib import Path
from typing import Any, Protocol

import inngest
from inngest.connect import ConnectionState, connect

from gh_issue.functions.backfill import backfill_page
from gh_issue.functions.client import client
from gh_issue.functions.index_issue import index_issue
from gh_issue.functions.reconcile import reconcile, sweep_repo
from gh_issue.index.records.heartbeat import (
    Heartbeat,
    heartbeat_path,
    remove_heartbeat,
    write_heartbeat,
)
from gh_issue.interface.cli.preflight import preflight
from gh_issue.utils import now, print_now

FUNCTIONS: list[inngest.Function[Any]] = [index_issue, backfill_page, reconcile, sweep_repo]

# How often the connection state is checked, and how often the heartbeat is
# rewritten when nothing has changed. Readers treat a heartbeat older than
# heartbeat.STALE_AFTER as a worker that is gone.
TICK = 1.0
HEARTBEAT_EVERY = 10.0


class Connection(Protocol):
    """The part of the SDK's connection the worker uses."""

    def get_state(self) -> ConnectionState: ...

    async def start(self) -> None: ...


_SAY = {
    ConnectionState.CONNECTING: "Connecting to {where}...",
    ConnectionState.ACTIVE: "Connected to {where}. Waiting for work; Ctrl+C stops.",
    ConnectionState.RECONNECTING: "Lost the connection to {where}; reconnecting...",
    ConnectionState.PAUSED: "Paused by {where}.",
    ConnectionState.CLOSING: "Stopping: finishing the steps already running...",
    ConnectionState.CLOSED: "Stopped.",
}


class Beat:
    """Called on every tick with the connection's state. Writes the heartbeat
    on the first tick, on every change of state, and every HEARTBEAT_EVERY
    seconds otherwise, and says each change of state once."""

    def __init__(
        self,
        path: Path,
        mode: str,
        *,
        clock: Callable[[], float] = time.monotonic,
        say: Callable[[str], None] = print_now,
    ) -> None:
        self._path = path
        self._mode = mode
        self._clock = clock
        self._say = say
        self._where = "the local dev server" if mode == "dev" else "Inngest Cloud"
        self._started_at = now()
        self._state: ConnectionState | None = None
        self._written_at = 0.0

    def __call__(self, state: ConnectionState) -> None:
        changed = state != self._state
        if changed:
            self._say(
                _SAY.get(state, "Connection {state}.").format(
                    where=self._where, state=state.name.lower()
                )
            )
            self._state = state
        if changed or self._clock() - self._written_at >= HEARTBEAT_EVERY:
            self._write(state)

    def stop(self) -> None:
        remove_heartbeat(self._path)

    def _write(self, state: ConnectionState) -> None:
        heartbeat: Heartbeat = {
            "pid": os.getpid(),
            "host": socket.gethostname(),
            "mode": self._mode,
            "state": state.name,
            "started_at": self._started_at,
            "beat_at": now(),
        }
        write_heartbeat(self._path, heartbeat)
        self._written_at = self._clock()


async def serve(
    connection: Connection, on_tick: Callable[[ConnectionState], None], tick: float = TICK
) -> None:
    """Run the connection until it closes, calling `on_tick` with its state."""

    async def ticking() -> None:
        while True:
            on_tick(connection.get_state())
            await asyncio.sleep(tick)

    ticker = asyncio.create_task(ticking())
    try:
        await connection.start()
    finally:
        ticker.cancel()
        with suppress(asyncio.CancelledError):
            await ticker


def run(say: Callable[[str], None] = print_now) -> None:
    problems = preflight(production=client.is_production)
    if problems:
        raise SystemExit(
            "The worker can't start:\n" + "\n".join(f"  - {problem}" for problem in problems)
        )
    beat = Beat(heartbeat_path(), "cloud" if client.is_production else "dev", say=say)
    # Created here, on the main thread, so the SDK can take over Ctrl+C and
    # turn it into a graceful close: steps already running finish first.
    connection = connect([(client, FUNCTIONS)])
    try:
        asyncio.run(serve(connection, beat, tick=TICK))
    finally:
        beat.stop()
