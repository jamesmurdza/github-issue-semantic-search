"""Small per-repo state on disk: one JSON file per repo, written atomically.

The sweep's cursor and the backfill's progress both live like this.

One file per repo rather than one shared file is deliberate. The sweep fans out
a run per repo and a worker runs them concurrently. With a single file, every
write is a read-modify-write of the whole map, and two racing writes lose one
repo's update. Separate files mean a write never touches another repo's state.
"""

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any


def repo_slug(repo: str) -> str:
    """`owner/name` becomes `owner__name`. Readable when you go looking, and the
    real name is stored inside each record, so nothing depends on reversing it."""
    return repo.replace("/", "__")


def write_json(path: Path, record: Mapping[str, Any]) -> None:
    """Write atomically: a temp file in the same directory, then a rename over
    the old one. A reader sees the old record or the new one, never half."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle, temporary = tempfile.mkstemp(dir=path.parent, suffix=".tmp")
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as file:
            json.dump(dict(record), file)
        os.replace(temporary, path)
    except BaseException:
        Path(temporary).unlink(missing_ok=True)
        raise


def read_json(path: Path) -> dict[str, Any]:
    record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return record


class RepoFiles:
    def __init__(self, directory: str) -> None:
        self._directory = Path(directory)

    def read(self, repo: str) -> dict[str, Any] | None:
        path = self._path(repo)
        return read_json(path) if path.exists() else None

    def write(self, record: Mapping[str, Any]) -> None:
        """Store `record` under its own `repo` key."""
        write_json(self._path(record["repo"]), record)

    def delete(self, repo: str) -> None:
        self._path(repo).unlink(missing_ok=True)

    def repos(self) -> list[str]:
        """Every repo with a record here."""
        if not self._directory.exists():
            return []
        return sorted(read_json(path)["repo"] for path in self._directory.glob("*.json"))

    def _path(self, repo: str) -> Path:
        return self._directory / f"{repo_slug(repo)}.json"
