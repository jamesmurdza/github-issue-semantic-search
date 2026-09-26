"""Configuration, and the one shared handle on the index.

Event names live here rather than next to the functions that send them, because
three different triggers emit `repo/issue.changed` and only one function consumes
it. Keeping the strings in one place is what makes that a single indexing path
instead of three that happen to agree.
"""

import os
from functools import lru_cache

from gh_issue.config import home
from gh_issue.embed import DIM
from gh_issue.index.backfills import Backfills
from gh_issue.index.cursors import Cursors
from gh_issue.index.failures import Failures
from gh_issue.index.store import Store

# The backfill, the webhook transform and the hourly sweep all emit this one.
EVENT_ISSUE_CHANGED = "repo/issue.changed"
# Sent by the backfill to itself, carrying the cursor for the next page.
EVENT_BACKFILL_PAGE = "repo/backfill.page"
# Fanned out by the cron so each repo sweeps in its own run.
EVENT_SWEEP_REQUESTED = "repo/sweep.requested"

DEFAULT_QDRANT_URL = "http://localhost:6333"


def qdrant_url() -> str:
    """Where the index lives: the qdrant.url setting."""
    return os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL)


def data_dir() -> str:
    """Local state that is not vectors: sweep cursors, backfill progress,
    failure markers and the worker's heartbeat. Beside the settings file, so
    every command finds it whatever directory it runs from."""
    return str(home())


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


@lru_cache(maxsize=1)
def cursors() -> Cursors:
    """Each tracked repo's sweep cursor, one small file per repo."""
    return Cursors(os.path.join(data_dir(), "cursors"))


@lru_cache(maxsize=1)
def backfills() -> Backfills:
    """Each repo's backfill progress, one small file per repo."""
    return Backfills(os.path.join(data_dir(), "backfills"))


@lru_cache(maxsize=1)
def failures() -> Failures:
    """Issues that failed every retry, one small file per issue."""
    return Failures(os.path.join(data_dir(), "failed"))


def is_synced(repo: str) -> bool:
    """Whether `repo` is synced here: it has a sweep cursor or a backfill record.

    `gh-issue remove` deletes both, and can't reach work already queued in
    Inngest for the repo. So the functions ask this before doing anything, and
    whatever was queued for a removed repo finds it gone and stops.
    """
    return cursors().get(repo) is not None or backfills().get(repo) is not None
