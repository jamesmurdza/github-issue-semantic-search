"""The worker's heartbeat file, read from outside the worker.

The worker rewrites `.gh-issue/worker.json` while it runs (see `worker.py`). This
is the reading side, kept apart so that `status`, `sync` and the MCP server can
ask whether a worker is up without importing the worker itself.
"""

import datetime
from pathlib import Path
from typing import TypedDict, cast

from gh_issue.index.repo_files import read_json
from gh_issue.settings import data_dir

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
    return Path(data_dir()) / "worker.json"


def read_heartbeat(path: Path | None = None) -> Heartbeat | None:
    path = path or heartbeat_path()
    return cast(Heartbeat, read_json(path)) if path.exists() else None


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
