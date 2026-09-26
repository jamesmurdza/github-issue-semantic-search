"""Tests for the settings file, ~/.gh-issue/config.toml, and the environment
variables that override it. Real files in a directory of each test's own."""

import os
import stat
from pathlib import Path

import pytest

from gh_issue import config

ENV_VARS = ["GITHUB_TOKEN", "INNGEST_DEV", "INNGEST_EVENT_KEY", "INNGEST_SIGNING_KEY", "QDRANT_URL"]


@pytest.fixture(autouse=True)
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """An empty home, and none of the settings' variables set."""
    monkeypatch.setenv("GH_ISSUE_HOME", str(tmp_path / "home"))
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    return tmp_path / "home"


def test_home_is_dot_gh_issue_in_the_user_home_unless_moved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GH_ISSUE_HOME")
    assert config.home() == Path.home() / ".gh-issue"
    monkeypatch.setenv("GH_ISSUE_HOME", "/elsewhere")
    assert config.home() == Path("/elsewhere")


def test_set_writes_one_toml_section_per_service(home: Path) -> None:
    config.set_value("qdrant.url", "http://qdrant.internal:6333")
    config.set_value("inngest.dev", "1")

    assert (home / "config.toml").read_text(encoding="utf-8") == (
        '[inngest]\ndev = "1"\n\n[qdrant]\nurl = "http://qdrant.internal:6333"\n'
    )
    assert config.read() == {"inngest.dev": "1", "qdrant.url": "http://qdrant.internal:6333"}


def test_set_replaces_and_unset_removes(home: Path) -> None:
    config.set_value("github.token", "ghp_first")
    config.set_value("github.token", "ghp_second")
    config.set_value("qdrant.url", "http://q:6333")

    assert config.read()["github.token"] == "ghp_second"
    assert config.unset("github.token") is True
    assert config.read() == {"qdrant.url": "http://q:6333"}
    assert config.unset("github.token") is False


def test_an_unknown_setting_is_refused_with_the_known_ones() -> None:
    with pytest.raises(config.ConfigError, match=r"qdrant\.host.*qdrant\.url"):
        config.set_value("qdrant.host", "x")
    with pytest.raises(config.ConfigError, match=r"qdrant.host"):
        config.unset("qdrant.host")


def test_a_file_that_is_not_toml_is_named_in_the_error(home: Path) -> None:
    home.mkdir()
    (home / "config.toml").write_text("[qdrant\nurl = ", encoding="utf-8")
    with pytest.raises(config.ConfigError, match=r"config\.toml is not valid TOML"):
        config.read()


def test_a_value_written_by_hand_as_a_boolean_still_works(home: Path) -> None:
    home.mkdir()
    (home / "config.toml").write_text("[inngest]\ndev = true\n", encoding="utf-8")
    assert config.read() == {"inngest.dev": "1"}


def test_unknown_names_in_the_file_are_ignored(home: Path) -> None:
    home.mkdir()
    (home / "config.toml").write_text('[qdrant]\nurl = "http://q"\nport = "1"\n', encoding="utf-8")
    assert config.read() == {"qdrant.url": "http://q"}


def test_the_file_is_readable_by_its_owner_only(home: Path) -> None:
    """It holds the GitHub token and the Inngest keys. Windows has no such
    permission bits, and a user's home there is private already."""
    config.set_value("github.token", "ghp_secret")
    mode = stat.S_IMODE((home / "config.toml").stat().st_mode)
    if os.name != "nt":
        assert mode & 0o077 == 0


def test_apply_fills_in_what_the_environment_does_not_set(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config.set_value("qdrant.url", "http://from-file:6333")
    config.set_value("inngest.dev", "1")
    monkeypatch.setenv("QDRANT_URL", "http://from-env:6333")

    config.apply()

    assert os.environ["QDRANT_URL"] == "http://from-env:6333"
    assert os.environ["INNGEST_DEV"] == "1"


def test_apply_treats_an_empty_variable_as_unset(monkeypatch: pytest.MonkeyPatch) -> None:
    config.set_value("github.token", "ghp_file")
    monkeypatch.setenv("GITHUB_TOKEN", "")

    config.apply()

    assert os.environ["GITHUB_TOKEN"] == "ghp_file"


def test_apply_without_a_file_changes_nothing() -> None:
    config.apply()
    assert not any(name in os.environ for name in ENV_VARS)


def test_shown_masks_secrets_and_says_where_each_value_came_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config.set_value("github.token", "ghp_abcdefgh1234")
    config.set_value("inngest.dev", "1")
    monkeypatch.setenv("INNGEST_SIGNING_KEY", "signkey-prod-abcdef987654")

    shown = {row["key"]: (row["value"], row["source"]) for row in config.shown()}

    assert shown == {
        "github.token": ("****1234", "config file"),
        "inngest.dev": ("1", "config file"),
        "inngest.event_key": (None, None),
        "inngest.signing_key": ("****7654", "environment (INNGEST_SIGNING_KEY)"),
        "qdrant.url": ("http://localhost:6333", "default"),
    }


def test_shown_after_apply_still_credits_the_file(monkeypatch: pytest.MonkeyPatch) -> None:
    """The CLI applies the file before running any command, `config` included."""
    config.set_value("qdrant.url", "http://from-file:6333")
    config.apply()

    [qdrant] = [row for row in config.shown() if row["key"] == "qdrant.url"]

    assert qdrant["source"] == "config file"


def test_an_environment_variable_that_overrides_the_file_is_credited(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config.set_value("qdrant.url", "http://from-file:6333")
    monkeypatch.setenv("QDRANT_URL", "http://from-env:6333")
    config.apply()

    [qdrant] = [row for row in config.shown() if row["key"] == "qdrant.url"]

    assert (qdrant["value"], qdrant["source"]) == (
        "http://from-env:6333",
        "environment (QDRANT_URL)",
    )


def test_a_short_secret_is_masked_entirely() -> None:
    config.set_value("inngest.event_key", "abc123")
    [row] = [row for row in config.shown() if row["key"] == "inngest.event_key"]
    assert row["value"] == "****"
