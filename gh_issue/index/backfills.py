"""Where each repo's backfill got to, for `gh-issue sync` and `status`.

The backfill does not read this. Its cursor rides in the event that starts each
page, so a run never has to look anything up to know where it is. This record is
for the outside: `sync` reads it to decide whether a backfill is unfinished and
where to resume it, and `status` reads it to show whether one is moving.

Kept apart from the sweep's cursor so that the two never overwrite each other.
"""

from typing import TypedDict, cast

from gh_issue.index.repo_files import RepoFiles


class BackfillProgress(TypedDict):
    repo: str
    # The first page's run id, carried by every page after it. Resuming re-sends
    # this chain's own event id, so a chain that is still alive drops it.
    chain: str
    # GitHub's cursor for the next page to fetch; None once there is none.
    next: str | None
    finished: bool
    # When the last page was recorded. A backfill whose `at` stops moving has
    # stalled.
    at: str


class Backfills:
    def __init__(self, directory: str) -> None:
        self._files = RepoFiles(directory)

    def get(self, repo: str) -> BackfillProgress | None:
        return cast(BackfillProgress | None, self._files.read(repo))

    def set(self, progress: BackfillProgress) -> None:
        self._files.write(progress)

    def repos(self) -> list[str]:
        """Every repo a backfill was ever started for."""
        return self._files.repos()
