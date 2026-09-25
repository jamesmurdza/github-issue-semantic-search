import threading
from pathlib import Path

from gh_issue.index.failures import Failure, Failures


def failure(repo: str, number: int, error: str = "boom") -> Failure:
    return {
        "repo": repo,
        "number": number,
        "error": error,
        "run_id": f"run-{number}",
        "at": "2026-09-25T10:00:00Z",
    }


def test_nothing_has_failed_at_first(tmp_path: Path) -> None:
    assert Failures(str(tmp_path)).for_repo("owner/name") == []


def test_a_failure_is_kept_until_cleared(tmp_path: Path) -> None:
    Failures(str(tmp_path)).add(failure("owner/name", 412))
    assert Failures(str(tmp_path)).for_repo("owner/name") == [failure("owner/name", 412)]


def test_failing_again_replaces_the_marker_rather_than_adding_one(tmp_path: Path) -> None:
    failures = Failures(str(tmp_path))
    failures.add(failure("owner/name", 412, error="first"))
    failures.add(failure("owner/name", 412, error="second"))
    assert [f["error"] for f in failures.for_repo("owner/name")] == ["second"]


def test_clear_removes_only_that_issue(tmp_path: Path) -> None:
    failures = Failures(str(tmp_path))
    failures.add(failure("owner/name", 412))
    failures.add(failure("owner/name", 413))
    failures.clear("owner/name", 412)
    assert [f["number"] for f in failures.for_repo("owner/name")] == [413]


def test_clearing_what_is_not_there_is_fine(tmp_path: Path) -> None:
    """`sync` clears after re-sending, and a second `sync` may get there first."""
    Failures(str(tmp_path)).clear("owner/name", 412)


def test_repos_are_kept_apart(tmp_path: Path) -> None:
    failures = Failures(str(tmp_path))
    failures.add(failure("a/b", 1))
    failures.add(failure("c/d", 1))
    assert failures.for_repo("a/b") == [failure("a/b", 1)]


def test_failures_in_one_repo_at_the_same_moment_all_land(tmp_path: Path) -> None:
    """Failures arrive from separate runs, often together — a bad token fails a
    whole page of issues at once. One file per issue means no write touches
    another issue's marker."""
    failures = Failures(str(tmp_path))
    threads = [
        threading.Thread(target=failures.add, args=(failure("owner/name", n),)) for n in range(50)
    ]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()

    assert [f["number"] for f in failures.for_repo("owner/name")] == list(range(50))
