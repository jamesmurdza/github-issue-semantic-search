"""Issues that failed every retry, waiting for `gh-issue sync` to try again.

Inngest retries a failed run on its own, with backoff. When the retries run out
the run is marked failed and stays that way: nothing in the pipeline will index
that issue again until it next changes on GitHub, which for a closed issue may
be never. So the failure handler leaves a marker here, `status` counts them, and
`sync` sends each one again and clears it.

One file per issue — `failed/owner__name/412.json` — rather than a list per
repo. Failures come from separate runs and often together (a bad token fails a
whole page of issues at once), and a shared list would lose some of them to
racing writes. A marker names the issue and keeps no copy of it: `sync` reads
the issue from GitHub again before retrying, so a retry never carries old data.
"""

import contextlib
import shutil
from pathlib import Path
from typing import TypedDict, cast

from gh_issue.index.repo_files import patiently, read_json, repo_slug, write_json


class Failure(TypedDict):
    repo: str
    number: int
    # The last error, for `status`; the full trace is on the run in Inngest.
    error: str
    run_id: str
    at: str


class Failures:
    def __init__(self, directory: str) -> None:
        self._directory = Path(directory)

    def add(self, failure: Failure) -> None:
        """Mark an issue as failed. Failing again replaces the earlier marker."""
        write_json(self._path(failure["repo"], failure["number"]), failure)

    def for_repo(self, repo: str) -> list[Failure]:
        folder = self._directory / repo_slug(repo)
        if not folder.exists():
            return []
        found = [cast(Failure, read_json(path)) for path in folder.glob("*.json")]
        return sorted(found, key=lambda failure: failure["number"])

    def clear(self, repo: str, number: int) -> None:
        path = self._path(repo, number)
        patiently(lambda: path.unlink(missing_ok=True))

    def clear_repo(self, repo: str) -> None:
        """Every marker for `repo`, for when the repo itself is removed."""
        folder = self._directory / repo_slug(repo)
        with contextlib.suppress(FileNotFoundError):
            patiently(lambda: shutil.rmtree(folder))

    def _path(self, repo: str, number: int) -> Path:
        return self._directory / repo_slug(repo) / f"{number}.json"
