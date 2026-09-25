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


def _filename(repo: str) -> str:
    """`owner/name` becomes `owner__name.json`. Readable when you go looking, and
    the real name is stored inside, so nothing depends on reversing this."""
    return repo.replace("/", "__") + ".json"


class RepoFiles:
    def __init__(self, directory: str) -> None:
        self._directory = Path(directory)

    def read(self, repo: str) -> dict[str, Any] | None:
        path = self._directory / _filename(repo)
        if not path.exists():
            return None
        return self._load(path)

    def write(self, record: Mapping[str, Any]) -> None:
        """Store `record` under its own `repo` key.

        Atomically: a temp file in the same directory, then a rename over the
        old one. A reader sees the old record or the new one, never half.
        """
        self._directory.mkdir(parents=True, exist_ok=True)
        handle, temporary = tempfile.mkstemp(dir=self._directory, suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                json.dump(dict(record), file)
            os.replace(temporary, self._directory / _filename(record["repo"]))
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def repos(self) -> list[str]:
        """Every repo with a record here."""
        if not self._directory.exists():
            return []
        return sorted(self._load(path)["repo"] for path in self._directory.glob("*.json"))

    @staticmethod
    def _load(path: Path) -> dict[str, Any]:
        record: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
        return record
