from collections.abc import Iterator
from pathlib import Path

import pytest

from gh_issue import settings


def _forget_handles() -> None:
    settings.cursors.cache_clear()
    settings.backfills.cache_clear()
    settings.failures.cache_clear()


@pytest.fixture
def data_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Path]:
    """Real local records, in a directory of this test's own.

    The handles are built once per process and cached, so they are dropped on
    the way in, to pick up this directory, and on the way out, so that no later
    test inherits it.
    """
    directory = tmp_path / "data"
    monkeypatch.setenv("GH_ISSUE_DATA_DIR", str(directory))
    _forget_handles()
    yield directory
    _forget_handles()
