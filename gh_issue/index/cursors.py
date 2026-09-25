"""Where each repo's hourly sweep got to.

One small file per repo, holding the moment its last sweep started. The sweep
asks GitHub for everything updated since then, so a cursor that does not move
while the worker is down means the first sweep afterwards covers the whole gap —
one dropped webhook and three days offline recover the same way.

The backfill keeps no cursor here: its cursor rides in the event that starts the
next page, and where it got to is recorded separately in `backfills`.
"""

from gh_issue.index.repo_files import RepoFiles


class Cursors:
    def __init__(self, directory: str) -> None:
        self._files = RepoFiles(directory)

    def get(self, repo: str) -> str | None:
        record = self._files.read(repo)
        return None if record is None else str(record["since"])

    def set(self, repo: str, since: str) -> None:
        self._files.write({"repo": repo, "since": since})

    def delete(self, repo: str) -> None:
        self._files.delete(repo)

    def tracked(self) -> list[str]:
        """Every repo with a cursor — which is every repo that has been indexed."""
        return self._files.repos()
