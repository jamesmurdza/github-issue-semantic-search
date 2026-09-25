"""The CLI's start-up: `.env` has to be loaded before the Inngest client exists,
because the client reads INNGEST_DEV once, when it is built."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

from gh_issue import cli


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
