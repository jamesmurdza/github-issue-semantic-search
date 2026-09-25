"""Reading the worker's heartbeat: how `status` and `sync` tell whether a worker
is here to do the work they queue."""

import datetime
from pathlib import Path

from gh_issue.heartbeat import Heartbeat, is_fresh, read_heartbeat, worker_status
from gh_issue.index.repo_files import write_json

AT = datetime.datetime(2026, 9, 25, 10, 0, 30, tzinfo=datetime.UTC)


def heartbeat(beat_at: str, state: str = "ACTIVE") -> Heartbeat:
    return {
        "pid": 1,
        "host": "h",
        "mode": "dev",
        "state": state,
        "started_at": "2026-09-25T09:00:00Z",
        "beat_at": beat_at,
    }


def test_no_heartbeat_means_no_worker(tmp_path: Path) -> None:
    assert read_heartbeat(tmp_path / "worker.json") is None
    assert not is_fresh(None, AT)


def test_a_heartbeat_goes_stale_after_thirty_seconds() -> None:
    """A worker killed outright never removes its file; the age is what tells."""
    assert is_fresh(heartbeat("2026-09-25T10:00:05Z"), AT)
    assert not is_fresh(heartbeat("2026-09-25T09:59:59Z"), AT)


def test_status_of_a_worker_that_never_ran_here(tmp_path: Path) -> None:
    assert worker_status(AT, tmp_path / "worker.json") == {
        "running": False,
        "connected": False,
        "state": None,
        "mode": None,
        "last_seen": None,
    }


def test_status_of_a_connected_worker(tmp_path: Path) -> None:
    path = tmp_path / "worker.json"
    write_json(path, heartbeat("2026-09-25T10:00:25Z"))

    assert worker_status(AT, path) == {
        "running": True,
        "connected": True,
        "state": "ACTIVE",
        "mode": "dev",
        "last_seen": "2026-09-25T10:00:25Z",
    }


def test_a_running_worker_that_lost_its_connection_is_not_connected(tmp_path: Path) -> None:
    path = tmp_path / "worker.json"
    write_json(path, heartbeat("2026-09-25T10:00:25Z", state="RECONNECTING"))

    status = worker_status(AT, path)

    assert status["running"] is True
    assert status["connected"] is False
    assert status["state"] == "RECONNECTING"


def test_a_stale_heartbeat_is_not_running_but_says_when_it_was_last_seen(
    tmp_path: Path,
) -> None:
    path = tmp_path / "worker.json"
    write_json(path, heartbeat("2026-09-25T09:50:00Z"))

    status = worker_status(AT, path)

    assert status["running"] is False
    assert status["connected"] is False
    assert status["last_seen"] == "2026-09-25T09:50:00Z"
