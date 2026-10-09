"""The environment variables the package reads itself. The CLI fills them in
from ~/.gh-issue/config.toml first; that part is tested in tests/interface/cli/."""

from pathlib import Path

import pytest

from gh_issue import env


def test_home_is_dot_gh_issue_in_the_user_home_unless_moved(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("GH_ISSUE_HOME")
    assert env.home() == Path.home() / ".gh-issue"
    monkeypatch.setenv("GH_ISSUE_HOME", "/elsewhere")
    assert env.home() == Path("/elsewhere")


def test_qdrant_url_defaults_to_the_compose_file(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("QDRANT_URL", raising=False)
    assert env.qdrant_url() == "http://localhost:6333"


def test_qdrant_url_honours_the_environment(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://qdrant.internal:6333")
    assert env.qdrant_url() == "http://qdrant.internal:6333"


def test_the_model_is_cached_in_the_gh_issue_home_unless_moved(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Not the working directory: the model is 65 MB, and a command run from
    anywhere else would download it again."""
    monkeypatch.setenv("GH_ISSUE_HOME", str(tmp_path))
    monkeypatch.delenv("GH_ISSUE_MODEL_CACHE")
    assert env.model_cache() == str(tmp_path / "models")

    monkeypatch.setenv("GH_ISSUE_MODEL_CACHE", "/shared/models")
    assert env.model_cache() == "/shared/models"


def test_an_empty_github_token_counts_as_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """So an exported-but-blank variable gets the "not set" message, rather
    than a request GitHub refuses."""
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_abc")
    assert env.github_token() == "ghp_abc"
    monkeypatch.setenv("GITHUB_TOKEN", "")
    assert env.github_token() is None
    monkeypatch.delenv("GITHUB_TOKEN")
    assert env.github_token() is None


def test_an_empty_qdrant_api_key_counts_as_not_set(monkeypatch: pytest.MonkeyPatch) -> None:
    """A Qdrant on your own machine has no key, so not set is the usual case,
    and a blank variable must mean the same."""
    monkeypatch.setenv("QDRANT_API_KEY", "key-from-the-cloud-console")
    assert env.qdrant_api_key() == "key-from-the-cloud-console"
    monkeypatch.setenv("QDRANT_API_KEY", "")
    assert env.qdrant_api_key() is None
    monkeypatch.delenv("QDRANT_API_KEY")
    assert env.qdrant_api_key() is None
