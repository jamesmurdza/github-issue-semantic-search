from collections.abc import Iterator
from pathlib import Path

import pytest

from gh_issue.index import storage


def _forget_handles() -> None:
    storage.cursors.cache_clear()
    storage.backfills.cache_clear()
    storage.failures.cache_clear()


# The embedding models, downloaded once and shared by every test, rather than
# once per test into each test's own home.
MODEL_CACHE = Path(__file__).parent.parent / ".fastembed_cache"


@pytest.fixture(autouse=True)
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """A gh-issue home of each test's own, so that no test reads your settings
    or your sync records, or writes to them. Real local records live here.

    The handles are built once per process and cached, so they are dropped on
    the way in, to pick up this directory, and on the way out, so that no later
    test inherits it.
    """
    directory = tmp_path / "home"
    monkeypatch.setenv("GH_ISSUE_HOME", str(directory))
    monkeypatch.setenv("GH_ISSUE_MODEL_CACHE", str(MODEL_CACHE))
    _forget_handles()
    yield directory
    _forget_handles()
