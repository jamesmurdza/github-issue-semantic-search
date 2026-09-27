"""The worker's heartbeat file, `~/.gh-issue/worker.json`.

The worker rewrites it while it runs (see `interface/cli/worker.py`), and `status`, `sync`
and the MCP server read it to tell whether a worker is up. Both sides go through
this module, so they can't disagree on the format, and neither side has to
import the other.
"""

import datetime
from pathlib import Path
from typing import TypedDict, cast

from gh_issue.env import home
from gh_issue.utils import patiently, read_json, write_json

# A heartbeat older than this means the worker is gone, even if it never got to
# remove its file. The worker writes one at least every 10 s.
STALE_AFTER = datetime.timedelta(seconds=30)


class Heartbeat(TypedDict):
    pid: int
    host: str
    mode: str  # "dev" or "cloud"
    state: str  # the connection's state, e.g. ACTIVE or RECONNECTING
    started_at: str
    beat_at: str


class WorkerStatus(TypedDict):
    # A heartbeat from the last 30 s.
    running: bool
    # Running, and its connection to Inngest is up.
    connected: bool
    # What it last reported; all None if no worker has run here.
    state: str | None
    mode: str | None
    last_seen: str | None


def heartbeat_path() -> Path:
    return home() / "worker.json"


def write_heartbeat(path: Path, heartbeat: Heartbeat) -> None:
    write_json(path, heartbeat)


def remove_heartbeat(path: Path) -> None:
    # Through `patiently`: on Windows a reader holding it open blocks the delete.
    patiently(lambda: path.unlink(missing_ok=True))


def read_heartbeat(path: Path | None = None) -> Heartbeat | None:
    path = path or heartbeat_path()
    # Asked, not checked first: the worker can remove it between a check and
    # the read.
    try:
        return cast(Heartbeat, read_json(path))
    except FileNotFoundError:
        return None


def is_fresh(heartbeat: Heartbeat | None, at: datetime.datetime) -> bool:
    if heartbeat is None:
        return False
    beat_at = datetime.datetime.fromisoformat(heartbeat["beat_at"])
    return at - beat_at <= STALE_AFTER


def worker_status(at: datetime.datetime | None = None, path: Path | None = None) -> WorkerStatus:
    heartbeat = read_heartbeat(path)
    if heartbeat is None:
        return {
            "running": False,
            "connected": False,
            "state": None,
            "mode": None,
            "last_seen": None,
        }
    running = is_fresh(heartbeat, at or datetime.datetime.now(datetime.UTC))
    return {
        "running": running,
        "connected": running and heartbeat["state"] == "ACTIVE",
        "state": heartbeat["state"],
        "mode": heartbeat["mode"],
        "last_seen": heartbeat["beat_at"],
    }
