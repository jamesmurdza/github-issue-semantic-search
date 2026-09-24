import threading
from pathlib import Path

from gh_issue.index.cursors import Cursors


def test_an_untracked_repo_has_no_cursor(tmp_path: Path) -> None:
    assert Cursors(str(tmp_path)).get("owner/name") is None


def test_set_then_get(tmp_path: Path) -> None:
    cursors = Cursors(str(tmp_path))
    cursors.set("owner/name", "2026-09-24T10:00:00Z")
    assert cursors.get("owner/name") == "2026-09-24T10:00:00Z"


def test_a_later_set_overwrites(tmp_path: Path) -> None:
    cursors = Cursors(str(tmp_path))
    cursors.set("owner/name", "2026-09-24T10:00:00Z")
    cursors.set("owner/name", "2026-09-24T11:00:00Z")
    assert cursors.get("owner/name") == "2026-09-24T11:00:00Z"


def test_cursors_survive_a_restart(tmp_path: Path) -> None:
    Cursors(str(tmp_path)).set("owner/name", "2026-09-24T10:00:00Z")
    assert Cursors(str(tmp_path)).get("owner/name") == "2026-09-24T10:00:00Z"


def test_tracked_lists_every_repo_with_a_cursor(tmp_path: Path) -> None:
    cursors = Cursors(str(tmp_path))
    cursors.set("a/b", "t1")
    cursors.set("c/d", "t2")
    assert sorted(cursors.tracked()) == ["a/b", "c/d"]


def test_tracked_is_empty_before_anything_is_indexed(tmp_path: Path) -> None:
    assert Cursors(str(tmp_path / "not-created-yet")).tracked() == []


def test_awkward_repo_names_round_trip(tmp_path: Path) -> None:
    cursors = Cursors(str(tmp_path))
    cursors.set("my-org/repo_name.js", "t1")
    assert cursors.tracked() == ["my-org/repo_name.js"]
    assert cursors.get("my-org/repo_name.js") == "t1"


def test_concurrent_writes_to_different_repos_all_land(tmp_path: Path) -> None:
    """The hourly sweep fans out one run per repo, and a worker runs them at the
    same time. If every repo shared one file, each write would be a read-modify-
    write of the whole map and two of them racing would lose one repo's cursor.
    One file per repo means a write never touches anyone else's."""
    cursors = Cursors(str(tmp_path))
    repos = [f"owner/repo-{i}" for i in range(50)]

    threads = [
        threading.Thread(target=cursors.set, args=(repo, "2026-09-24T10:00:00Z")) for repo in repos
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert sorted(cursors.tracked()) == sorted(repos)
