"""Where each repo's backfill got to, for `gh-issue sync` and `status`.

The backfill does not read its cursor from here. The cursor rides in the event
that starts each page, so a run never has to look anything up to know where it
is. What a page does check is that the record still names its chain, so a page
of a removed or replaced backfill stops. The rest is for the outside: `sync`
reads it to decide whether a backfill is unfinished and where to resume it, and
`status` reads it to show whether one is moving.

Kept apart from the sweep's cursor so that the two never overwrite each other.
"""

from typing import TypedDict, cast

from gh_issue.index.records.repo_files import RepoFiles


class BackfillProgress(TypedDict):
    repo: str
    # Picked by `gh-issue sync` when it starts the backfill, and carried by
    # every page. Resuming re-sends this chain's own event id, so a chain that
    # is still alive drops it; a page whose chain this no longer names stops.
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

    def delete(self, repo: str) -> None:
        self._files.delete(repo)

    def repos(self) -> list[str]:
        """Every repo a backfill was ever started for."""
        return self._files.repos()
