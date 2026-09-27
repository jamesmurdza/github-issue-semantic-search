"""One shared handle each on the store and on the records.

Everything that reads or writes the index gets its handles here, so a process
opens one Qdrant connection and one of each record directory, however many
functions and commands reach for them.
"""

import os
from functools import lru_cache

from gh_issue.env import home, qdrant_url
from gh_issue.index.embed import DIM
from gh_issue.index.records.backfills import Backfills
from gh_issue.index.records.cursors import Cursors
from gh_issue.index.records.failures import Failures
from gh_issue.index.store import Store


@lru_cache(maxsize=1)
def store() -> Store:
    """One connection per process, reused by every function and by the CLI."""
    return Store(dim=DIM, url=qdrant_url())


def index_problem() -> str | None:
    """Why the index can't be opened, in words to act on; None if it can."""
    try:
        store()
    except ValueError as error:
        # Qdrant answered, but the collection there is unusable; say which way.
        return str(error)
    except Exception:
        return f"Can't connect to Qdrant at {qdrant_url()} (qdrant.url)."
    return None


# The records live beside the settings file, so every command finds them
# whatever directory it runs from.


@lru_cache(maxsize=1)
def cursors() -> Cursors:
    """Each tracked repo's sweep cursor, one small file per repo."""
    return Cursors(os.path.join(home(), "cursors"))


@lru_cache(maxsize=1)
def backfills() -> Backfills:
    """Each repo's backfill progress, one small file per repo."""
    return Backfills(os.path.join(home(), "backfills"))


@lru_cache(maxsize=1)
def failures() -> Failures:
    """Issues that failed every retry, one small file per issue."""
    return Failures(os.path.join(home(), "failed"))


def is_synced(repo: str) -> bool:
    """Whether `repo` is synced here: it has a sweep cursor or a backfill record.

    `gh-issue remove` deletes both, and can't reach work already queued in
    Inngest for the repo. So the functions ask this before doing anything, and
    whatever was queued for a removed repo finds it gone and stops.
    """
    return cursors().get(repo) is not None or backfills().get(repo) is not None
