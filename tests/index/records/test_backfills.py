from pathlib import Path

from gh_issue.index.records.backfills import BackfillProgress, Backfills


def progress(repo: str, next_cursor: str | None) -> BackfillProgress:
    return {
        "repo": repo,
        "chain": "c1",
        "next": next_cursor,
        "finished": next_cursor is None,
        "at": "2026-09-25T10:00:00Z",
    }


def test_a_repo_never_backfilled_has_no_progress(tmp_path: Path) -> None:
    assert Backfills(str(tmp_path)).get("owner/name") is None


def test_progress_survives_a_restart(tmp_path: Path) -> None:
    Backfills(str(tmp_path)).set(progress("owner/name", "CURSOR-3"))
    assert Backfills(str(tmp_path)).get("owner/name") == progress("owner/name", "CURSOR-3")


def test_each_repo_keeps_its_own_progress(tmp_path: Path) -> None:
    backfills = Backfills(str(tmp_path))
    backfills.set(progress("a/b", "CURSOR-3"))
    backfills.set(progress("c/d", None))
    assert backfills.get("a/b") == progress("a/b", "CURSOR-3")
    assert backfills.get("c/d") == progress("c/d", None)


def test_delete_forgets_that_repo_only(tmp_path: Path) -> None:
    backfills = Backfills(str(tmp_path))
    backfills.set(progress("a/b", "CURSOR-3"))
    backfills.set(progress("c/d", None))

    backfills.delete("a/b")

    assert backfills.get("a/b") is None
    assert backfills.repos() == ["c/d"]


def test_deleting_what_is_not_there_is_fine(tmp_path: Path) -> None:
    Backfills(str(tmp_path)).delete("owner/name")


def test_repos_lists_every_repo_with_a_backfill(tmp_path: Path) -> None:
    backfills = Backfills(str(tmp_path))
    backfills.set(progress("c/d", None))
    backfills.set(progress("a/b", "CURSOR-3"))
    assert backfills.repos() == ["a/b", "c/d"]
