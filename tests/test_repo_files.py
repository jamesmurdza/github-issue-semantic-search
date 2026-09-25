"""The per-repo record files under concurrent use.

On Windows a file can't be replaced or deleted while anything has it open, and
opening one fails while it is being replaced. The worker reads these records
all the time, since every run checks its repo is still synced, so the other
operations have to get through anyway. On other systems these pass trivially.
"""

import threading
from pathlib import Path

import pytest

from gh_issue.index import repo_files
from gh_issue.index.repo_files import RepoFiles

REPO = "owner/name"


def test_writes_deletes_and_reads_all_get_through_while_another_thread_reads(
    tmp_path: Path,
) -> None:
    """Measured before the fix, with one busy reader: three in four replaces
    and nine in ten deletes failed, and so did some of the reads."""
    files = RepoFiles(str(tmp_path))
    files.write({"repo": REPO, "since": "t0"})
    stop = threading.Event()
    reader_errors: list[BaseException] = []

    def keep_reading() -> None:
        while not stop.is_set():
            try:
                files.read(REPO)
            except Exception as error:
                reader_errors.append(error)

    reader = threading.Thread(target=keep_reading)
    reader.start()
    try:
        for i in range(500):
            files.write({"repo": REPO, "since": f"t{i}"})
            files.delete(REPO)
            assert files.read(REPO) is None
    finally:
        stop.set()
        reader.join()

    assert reader_errors == []


def test_a_file_that_stays_locked_still_fails_in_the_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Retrying is for a lock held for a moment, not for a real permission
    problem, which has to surface."""
    monkeypatch.setattr(repo_files, "_PAUSE", 0.0)
    tries: list[int] = []

    def always_locked() -> None:
        tries.append(1)
        raise PermissionError("locked")

    with pytest.raises(PermissionError):
        repo_files.patiently(always_locked)
    assert len(tries) == repo_files._ATTEMPTS


def test_a_lock_that_clears_is_waited_out(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(repo_files, "_PAUSE", 0.0)
    tries: list[int] = []

    def locked_twice() -> str:
        tries.append(1)
        if len(tries) <= 2:
            raise PermissionError("locked")
        return "done"

    assert repo_files.patiently(locked_twice) == "done"
    assert len(tries) == 3
