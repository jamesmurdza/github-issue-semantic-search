"""`patiently`: trying a file operation again while Windows has the file open.

The records' test of the same thing under a real concurrent reader is in
`index/records/test_repo_files.py`; these pin down the retrying itself.
"""

import pytest

from gh_issue import utils


def test_a_file_that_stays_locked_still_fails_in_the_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrying is for a lock held for a moment, not for a real permission
    problem, which has to surface."""
    monkeypatch.setattr(utils, "_PAUSE", 0.0)
    tries: list[int] = []

    def always_locked() -> None:
        tries.append(1)
        raise PermissionError("locked")

    with pytest.raises(PermissionError):
        utils.patiently(always_locked)
    assert len(tries) == utils._ATTEMPTS


def test_a_lock_that_clears_is_waited_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(utils, "_PAUSE", 0.0)
    tries: list[int] = []

    def locked_twice() -> str:
        tries.append(1)
        if len(tries) <= 2:
            raise PermissionError("locked")
        return "done"

    assert utils.patiently(locked_twice) == "done"
    assert len(tries) == 3
