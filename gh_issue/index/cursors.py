"""Where each repo's hourly sweep got to.

One small JSON file per repo, holding the moment its last sweep started. The
sweep asks GitHub for everything updated since then, so a cursor that does not
move while the worker is down means the first sweep afterwards covers the whole
gap — one dropped webhook and three days offline recover the same way.

One file per repo rather than one shared file is deliberate. The sweep fans out
a run per repo and a worker runs them concurrently. With a single file, every
write is a read-modify-write of the whole map, and two racing writes lose one
repo's update. Separate files mean a write never touches another repo's cursor.

The backfill keeps no cursor here: its cursor rides in the event that starts the
next page.
"""

import json
import os
import tempfile
from pathlib import Path
from typing import TypedDict


class _CursorFile(TypedDict):
    repo: str
    since: str


def _filename(repo: str) -> str:
    """`owner/name` becomes `owner__name.json`. Readable when you go looking, and
    the real name is stored inside, so nothing depends on reversing this."""
    return repo.replace("/", "__") + ".json"


class Cursors:
    def __init__(self, directory: str) -> None:
        self._directory = Path(directory)

    def get(self, repo: str) -> str | None:
        path = self._directory / _filename(repo)
        if not path.exists():
            return None
        return self._read(path)["since"]

    def set(self, repo: str, since: str) -> None:
        """Write atomically: a temp file in the same directory, then a rename over
        the old one. A reader sees the old cursor or the new one, never half."""
        self._directory.mkdir(parents=True, exist_ok=True)
        record: _CursorFile = {"repo": repo, "since": since}
        handle, temporary = tempfile.mkstemp(dir=self._directory, suffix=".tmp")
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as file:
                json.dump(record, file)
            os.replace(temporary, self._directory / _filename(repo))
        except BaseException:
            Path(temporary).unlink(missing_ok=True)
            raise

    def tracked(self) -> list[str]:
        """Every repo with a cursor — which is every repo that has been indexed."""
        if not self._directory.exists():
            return []
        return sorted(self._read(path)["repo"] for path in self._directory.glob("*.json"))

    @staticmethod
    def _read(path: Path) -> _CursorFile:
        record: _CursorFile = json.loads(path.read_text(encoding="utf-8"))
        return record
