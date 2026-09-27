"""Helpers that know nothing about issues, repos, Inngest or Qdrant.

Each is used by more than one part of the package; a helper with one caller
stays next to it.
"""

import datetime
import json
import os
import tempfile
import time
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

# How long to keep trying a file Windows says is in use: up to 100 x 10 ms.
_ATTEMPTS = 100
_PAUSE = 0.01


def now() -> str:
    """UTC to the second, in the format GitHub's `since` takes.

    Truncation rounds down, so a sweep window that starts here overlaps the
    previous one slightly rather than leaving a gap.

    Inside an Inngest function, call this only inside a step. The body replays
    once per step, so a read in plain Python happens again on every pass and
    the value that sticks is whatever the last pass saw.
    """
    return datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def print_now(message: str) -> None:
    """Print and flush. Unflushed, the lines sit in a buffer whenever the output
    goes to a file or a pipe rather than a terminal, and a worker that is killed
    outright never shows them at all."""
    print(message, flush=True)


def patiently[T](action: Callable[[], T]) -> T:
    """Run a file operation, trying again while the file is briefly in use.

    On Windows a file can't be replaced or deleted while anything has it open,
    and opening it fails while it is being replaced. The worker reads the sync
    records all the time, since every run checks its repo is still synced, so a
    write or a delete from anywhere else keeps landing on a read that lasts
    microseconds. Measured with one busy reader, three in four replaces and nine
    in ten deletes failed. Retried, none did, and the worst needed 13 tries.

    Elsewhere this never comes up, and a real permission problem still raises
    after the last try.
    """
    for _ in range(_ATTEMPTS - 1):
        try:
            return action()
        except PermissionError:
            time.sleep(_PAUSE)
    return action()


def write_json(path: Path, record: Mapping[str, Any]) -> None:
    write_text(path, json.dumps(dict(record)))


def write_text(path: Path, text: str) -> None:
    """Write atomically: a temp file in the same directory, then a rename over
    the old one. A reader sees the old contents or the new, never half.

    The temp file, and so the file, can be read by its owner only, which is
    what the settings file needs: it holds the GitHub token.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            file.write(text)
        patiently(lambda: os.replace(temporary, path))
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = json.loads(patiently(lambda: path.read_text(encoding="utf-8")))
    return record
