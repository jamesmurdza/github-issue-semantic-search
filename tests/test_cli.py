"""The CLI's start-up: `.env` has to be loaded before the Inngest client exists,
because the client reads INNGEST_DEV once, when it is built."""

import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

from gh_issue import cli, query, remove, settings
from gh_issue.index.store import SearchFilters
from gh_issue.query import Status


def test_importing_the_cli_does_not_build_the_inngest_client() -> None:
    """If it did, a `.env` saying INNGEST_DEV=1 would be read too late and the
    worker would try Inngest Cloud. A fresh interpreter, so nothing earlier in
    the test run has imported the client already."""
    probe = "import sys, gh_issue.cli; print('gh_issue.functions.client' in sys.modules)"
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_env_file_in_the_working_directory_is_loaded(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    (tmp_path / ".env").write_text("GH_ISSUE_TEST_SETTING=from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv("GH_ISSUE_TEST_SETTING", raising=False)

    cli.load_env()

    assert os.environ["GH_ISSUE_TEST_SETTING"] == "from-dotenv"


def test_a_variable_already_set_wins_over_env_file(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """So `INNGEST_DEV=1 gh-issue worker` works without editing .env."""
    (tmp_path / ".env").write_text("GH_ISSUE_TEST_SETTING=from-dotenv\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("GH_ISSUE_TEST_SETTING", "from-shell")

    cli.load_env()

    assert os.environ["GH_ISSUE_TEST_SETTING"] == "from-shell"


# search and status --------------------------------------------------------------


def test_search_takes_its_words_unquoted_and_repeatable_filters() -> None:
    command, bound, _ = cli.app.parse_args(
        ["search", "event", "key", "--repo", "a/b", "--repo", "c/d", "--label", "bug", "-k", "3"]
    )
    assert command is cli.search
    assert bound.args == ("event", "key")
    assert bound.kwargs == {"repo": ["a/b", "c/d"], "label": ["bug"], "limit": 3}


def test_search_refuses_a_state_that_does_not_exist() -> None:
    with pytest.raises(Exception, match='Choose from: "open", "closed"'):
        cli.app.parse_args(
            ["search", "x", "--state", "shut"], exit_on_error=False, print_error=False
        )


def test_search_hands_the_options_to_the_query_layer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    asked: list[tuple[str, SearchFilters | None, int]] = []

    def fake_search(text: str, filters: SearchFilters | None = None, limit: int = 10) -> list[Any]:
        asked.append((text, filters, limit))
        return []

    monkeypatch.setattr(settings, "index_problem", lambda: None)
    monkeypatch.setattr(query, "search", fake_search)

    cli.search("event", "key", repo=["a/b"], state="closed", label=["bug"], limit=3)

    assert asked == [("event key", {"repos": ["a/b"], "state": "closed", "labels": ["bug"]}, 3)]
    assert capsys.readouterr().out.strip() == "No matches."


def test_a_bad_filter_value_is_a_message_not_a_traceback(monkeypatch: pytest.MonkeyPatch) -> None:
    def refusing(text: str, filters: SearchFilters | None = None, limit: int = 10) -> list[Any]:
        raise ValueError("since must be a date like 2026-01-31, not 'yesterday'")

    monkeypatch.setattr(settings, "index_problem", lambda: None)
    monkeypatch.setattr(query, "search", refusing)

    with pytest.raises(SystemExit, match="since must be a date"):
        cli.search("x", since="yesterday")


@pytest.mark.parametrize("command", ["search", "status", "remove"])
def test_each_says_what_is_wrong_with_the_index_before_trying(
    monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    monkeypatch.setattr(settings, "index_problem", lambda: "Qdrant is not reachable")
    with pytest.raises(SystemExit, match="Qdrant is not reachable"):
        if command == "search":
            cli.search("x")
        elif command == "status":
            cli.status()
        else:
            cli.remove("a/b", yes=True)


def test_status_prints_the_worker_and_the_repos(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    nothing: Status = {
        "worker": {
            "running": False,
            "connected": False,
            "state": None,
            "mode": None,
            "last_seen": None,
        },
        "repos": [],
    }
    monkeypatch.setattr(settings, "index_problem", lambda: None)
    monkeypatch.setattr(query, "status", lambda: nothing)

    cli.status()

    out = capsys.readouterr().out
    assert out.startswith("Worker: not running.")
    assert "gh-issue sync owner/name" in out


# remove ------------------------------------------------------------------------


@pytest.mark.parametrize("flag", ["--yes", "-y"])
def test_remove_takes_one_repo_and_a_yes_flag(flag: str) -> None:
    command, bound, _ = cli.app.parse_args(["remove", "a/b", flag])
    assert command is cli.remove
    assert bound.args == ("a/b",)
    assert bound.kwargs == {"yes": True}


def test_remove_asks_unless_told_not_to() -> None:
    _, bound, _ = cli.app.parse_args(["remove", "a/b"])
    assert bound.kwargs == {}


def test_remove_hands_the_repo_and_the_flag_on(monkeypatch: pytest.MonkeyPatch) -> None:
    handed: list[tuple[str, bool]] = []

    def fake_run(repo: str, *, yes: bool) -> None:
        handed.append((repo, yes))

    monkeypatch.setattr(settings, "index_problem", lambda: None)
    monkeypatch.setattr(remove, "run", fake_run)

    cli.remove("a/b", yes=True)

    assert handed == [("a/b", True)]
