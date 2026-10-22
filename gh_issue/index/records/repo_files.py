"""Small per-repo state on disk: one JSON file per repo, written atomically.

The sweep's cursor and the backfill's progress both live like this.

One file per repo rather than one shared file is deliberate. The sweep fans out
a run per repo and a worker runs them concurrently. With a single file, every
write is a read-modify-write of the whole map, and two racing writes lose one
repo's update. Separate files mean a write never touches another repo's state.

Every operation on a file here goes through `patiently`, for Windows; see there.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Any

from gh_issue.utils import patiently, read_json, write_json


def repo_slug(repo: str) -> str:
    """`owner/name` becomes `owner__name`. Readable when you go looking, and the
    real name is stored inside each record, so nothing depends on reversing it."""
    return repo.replace("/", "__")


class RepoFiles:
    def __init__(self, directory: str) -> None:
        self._directory = Path(directory)

    def read(self, repo: str) -> dict[str, Any] | None:
        # Asked, not checked first: a record deleted between a check and the
        # read would otherwise raise instead of reading as absent.
        try:
            return read_json(self._path(repo))
        except FileNotFoundError:
            return None

    def write(self, record: Mapping[str, Any]) -> None:
        """Store `record` under its own `repo` key."""
        write_json(self._path(record["repo"]), record)

    def delete(self, repo: str) -> None:
        path = self._path(repo)
        patiently(lambda: path.unlink(missing_ok=True))

    def repos(self) -> list[str]:
        """Every repo with a record here."""
        if not self._directory.exists():
            return []
        return sorted(read_json(path)["repo"] for path in self._directory.glob("*.json"))

    def _path(self, repo: str) -> Path:
        return self._directory / f"{repo_slug(repo)}.json"
