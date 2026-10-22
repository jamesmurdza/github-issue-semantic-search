"""The CLI: its start-up, where the settings file has to be applied before the
Inngest client exists, because the client reads INNGEST_DEV once, when it is
built; and each command's handling of its arguments."""

import datetime
import os
import re
import shlex
import subprocess
import sys
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import cyclopts
import pytest
from starlette.testclient import TestClient

from gh_issue.github.types import Issue
from gh_issue.index import embed, query, storage
from gh_issue.index.chunking import chunks_for
from gh_issue.index.embed import DIM, embed_chunks
from gh_issue.index.query import Status
from gh_issue.index.store import SearchFilters, Store
from gh_issue.interface.cli import app as cli
from gh_issue.interface.cli import config, remove


def test_importing_the_cli_does_not_build_the_inngest_client() -> None:
    """If it did, a config file saying inngest.dev = "1" would be read too late
    and the worker would try Inngest Cloud. A fresh interpreter, so nothing earlier in
    the test run has imported the client already."""
    probe = (
        "import sys, gh_issue.interface.cli.app; print('gh_issue.functions.client' in sys.modules)"
    )
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, check=True
    )
    assert result.stdout.strip() == "False"


def test_output_piped_elsewhere_is_utf8_whatever_the_system_code_page(tmp_path: Path) -> None:
    """Issue text is anything people typed. Found in the full run:
    `gh-issue search ... | grep` crashed on Windows, where Python writes piped
    output in the system code page (cp1252 here) and a comment held U+276F."""
    probe = (
        "from gh_issue.interface.cli import app as cli; "
        "cli.app = lambda: print('\\u276f ok'); cli.main()"
    )
    environment = {
        key: value
        for key, value in os.environ.items()
        if key not in ("PYTHONIOENCODING", "PYTHONUTF8")
    }
    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, cwd=tmp_path, env=environment
    )
    assert result.returncode == 0, result.stderr.decode(errors="replace")
    assert result.stdout.decode("utf-8").strip() == chr(0x276F) + " ok"


@pytest.mark.parametrize(
    ("config_file", "production"),
    [
        pytest.param('[inngest]\ndev = "1"\n', False, id="the-file-says-dev-server"),
        pytest.param(None, True, id="nothing-set-means-cloud"),
    ],
)
def test_the_inngest_client_uses_the_dev_server_only_when_a_setting_says_so(
    tmp_path: Path, config_file: str | None, production: bool
) -> None:
    """The client reads INNGEST_DEV once, when it is built, so the file has to
    be applied before anything builds it. And the client is left to decide for
    itself: one built for the dev server would still pass the first case. A
    fresh interpreter, with INNGEST_DEV out of the environment."""
    if config_file is not None:
        (tmp_path / "config.toml").write_text(config_file, encoding="utf-8")
    probe = (
        "from gh_issue.interface.cli import app as cli\n"
        "def app():\n"
        "    from gh_issue.functions.client import client\n"
        "    print(client.is_production)\n"
        "cli.app = app\n"
        "cli.main()"
    )
    environment = {key: value for key, value in os.environ.items() if key != "INNGEST_DEV"}
    environment["GH_ISSUE_HOME"] = str(tmp_path)

    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=environment
    )

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == str(production)


def test_a_broken_config_file_is_a_message_not_a_traceback(tmp_path: Path) -> None:
    (tmp_path / "config.toml").write_text("[inngest\n", encoding="utf-8")
    probe = "from gh_issue.interface.cli import app as cli; cli.app = lambda: None; cli.main()"
    environment = {**os.environ, "GH_ISSUE_HOME": str(tmp_path)}

    result = subprocess.run(
        [sys.executable, "-c", probe], capture_output=True, text=True, env=environment
    )

    assert result.returncode == 1
    assert "config.toml is not valid TOML" in result.stderr
    assert "Traceback" not in result.stderr


# config ------------------------------------------------------------------------


@pytest.fixture
def home(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    monkeypatch.setenv("GH_ISSUE_HOME", str(tmp_path))
    for name in ("GITHUB_TOKEN", "INNGEST_DEV", "INNGEST_EVENT_KEY", "INNGEST_SIGNING_KEY"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.delenv("QDRANT_URL", raising=False)
    monkeypatch.delenv("QDRANT_API_KEY", raising=False)
    return tmp_path


@pytest.mark.parametrize("argv", [["config", "list"], ["config"]])
def test_config_list_shows_every_setting_and_what_it_is_for(
    home: Path, capsys: pytest.CaptureFixture[str], argv: list[str]
) -> None:
    command, _, _ = cli.app.parse_args(argv)
    command()
    out = capsys.readouterr().out
    assert "http://localhost:6333  default" in out
    for key, setting in config.SETTINGS.items():
        assert key in out
        assert setting["help"] in out


def test_config_get_prints_a_value_as_config_list_shows_it(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Masked if it is a secret, and the default if nothing sets it."""
    config.set_value("github.token", "ghp_abcdefgh5678")

    for key in ("github.token", "qdrant.url"):
        command, bound, _ = cli.app.parse_args(["config", "get", key])
        command(*bound.args, **bound.kwargs)

    assert capsys.readouterr().out.splitlines() == ["****5678", "http://localhost:6333"]


def test_config_get_of_a_setting_nothing_sets_says_so(home: Path) -> None:
    command, bound, _ = cli.app.parse_args(["config", "get", "inngest.event_key"])
    with pytest.raises(SystemExit, match=r"inngest\.event_key is not set"):
        command(*bound.args, **bound.kwargs)


def test_config_set_saves_a_value_given_on_the_command_line(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    command, bound, _ = cli.app.parse_args(["config", "set", "qdrant.url", "http://q:6333"])
    command(*bound.args, **bound.kwargs)

    assert config.read() == {"qdrant.url": "http://q:6333"}
    assert "Saved qdrant.url" in capsys.readouterr().out


def test_config_set_asks_for_a_secret_without_echoing_it(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """So the token stays out of the shell's history, and off the screen."""
    asked: list[str] = []

    def hidden(prompt: str) -> str:
        asked.append(prompt)
        return "ghp_typed_in_1234"

    monkeypatch.setattr(config, "ask_secret", hidden)

    cli.config_set("github.token")

    assert asked == ["github.token (not shown as you type): "]
    assert config.read() == {"github.token": "ghp_typed_in_1234"}
    assert "ghp_typed_in_1234" not in capsys.readouterr().out


def test_config_set_asks_for_a_plain_value_in_the_open(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(config, "ask", lambda prompt: "1")
    monkeypatch.setattr(config, "ask_secret", lambda prompt: pytest.fail("hid a plain value"))

    cli.config_set("inngest.dev")

    assert config.read() == {"inngest.dev": "1"}


def test_config_set_with_nothing_typed_changes_nothing(
    home: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    config.set_value("github.token", "ghp_kept_12345678")
    monkeypatch.setattr(config, "ask_secret", lambda prompt: "  ")

    with pytest.raises(SystemExit, match=r"github.token is unchanged"):
        cli.config_set("github.token")

    assert config.read() == {"github.token": "ghp_kept_12345678"}


def test_config_set_says_when_the_environment_still_wins(
    home: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setenv("QDRANT_URL", "http://from-env:6333")

    cli.config_set("qdrant.url", "http://from-file:6333")

    assert "QDRANT_URL is set in the environment" in capsys.readouterr().out


def test_config_set_does_not_mistake_the_files_own_value_for_the_environment(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    """Start-up copies the file into the environment, so the old value is
    there by the time `config set` runs; it is not a variable that wins."""
    config.set_value("qdrant.url", "http://old:6333")
    config.apply()

    cli.config_set("qdrant.url", "http://new:6333")

    assert "Note" not in capsys.readouterr().out


def test_config_unset_removes_a_setting(home: Path, capsys: pytest.CaptureFixture[str]) -> None:
    config.set_value("qdrant.url", "http://q:6333")

    command, bound, _ = cli.app.parse_args(["config", "unset", "qdrant.url"])
    command(*bound.args, **bound.kwargs)

    assert config.read() == {}
    assert "Removed qdrant.url" in capsys.readouterr().out


def test_config_unset_of_a_setting_the_file_lacks_says_so(
    home: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    cli.config_unset("qdrant.url")
    assert "qdrant.url is not in" in capsys.readouterr().out


# mistakes on the command line ---------------------------------------------------


@pytest.mark.parametrize(
    ("argv", "says"),
    [
        pytest.param(["search"], "WORDS requires an argument", id="search-with-no-words"),
        pytest.param(
            ["search", "x", "--state", "shut"], 'Choose from: "open", "closed"', id="state"
        ),
        pytest.param(
            ["search", "x", "--since", "yesterday"], "--since", id="since-that-is-not-a-date"
        ),
        pytest.param(["search", "x", "-k", "0"], "Must be >= 1", id="no-results-asked-for"),
        pytest.param(
            ["search", "x", "-k", "51"], "Must be <= 50", id="more-results-than-mcp-allows"
        ),
        pytest.param(["mcp", "http", "--port", "99999"], "Must be <= 65535", id="no-such-port"),
        pytest.param(["sync", "a/b", "inngest"], '"inngest" for REPOS. Use owner/name', id="sync"),
        pytest.param(["remove", "inngest"], '"inngest" for REPO. Use owner/name', id="remove"),
        pytest.param(
            ["search", "x", "--repo", "inngest"],
            '"inngest" for --repo. Use owner/name',
            id="search-repo",
        ),
        pytest.param(["remove"], "REPO requires an argument", id="remove-with-no-repo"),
        pytest.param(["web", "--port", "99999"], "Must be <= 65535", id="web-port"),
        # The repo is chosen on the page; one named here must not become the host.
        pytest.param(["web", "a/b"], "Unused Tokens: ['a/b']", id="web-with-a-repo"),
        pytest.param(
            ["config", "set", "qdrant.host", "x"],
            'Did you mean "qdrant.url"?',
            id="no-such-setting",
        ),
        pytest.param(
            ["config", "unset", "github.tokn"], 'Did you mean "github.token"?', id="unset-typo"
        ),
        pytest.param(["config", "set"], "KEY requires an argument", id="set-with-no-setting"),
        # Names people reach for, from other tools, pointed at the command here.
        pytest.param(["rm", "a/b"], 'Did you mean "remove"?', id="rm"),
        pytest.param(["delete", "a/b"], 'Did you mean "remove"?', id="delete"),
        pytest.param(["find", "x"], 'Did you mean "search"?', id="find"),
        pytest.param(["query", "x"], 'Did you mean "search"?', id="query"),
        pytest.param(["index", "a/b"], 'Did you mean "sync"?', id="index"),
        pytest.param(["add", "a/b"], 'Did you mean "sync"?', id="add"),
    ],
)
def test_a_mistake_on_the_command_line_is_refused_before_anything_runs(
    argv: list[str], says: str
) -> None:
    """Refused while the command line is read, so the refusal names the
    argument, and nothing has connected to Qdrant or loaded the model yet."""
    with pytest.raises(cyclopts.CycloptsError) as refused:
        cli.app.parse_args(argv, exit_on_error=False, print_error=False)
    assert says in str(refused.value)


# help ---------------------------------------------------------------------------


def help_for(argv: list[str], capsys: pytest.CaptureFixture[str]) -> str:
    with pytest.raises(SystemExit) as done:
        cli.app([*argv, "--help"])
    assert done.value.code == 0
    return capsys.readouterr().out


def test_help_groups_the_commands_in_the_order_you_use_them(
    capsys: pytest.CaptureFixture[str],
) -> None:
    text = help_for([], capsys)

    groups = re.findall(r"^\W+ (Set up|Index|Search|Help) \W", text, re.MULTILINE)
    rows = re.findall(r"^\W (\w[\w-]*) {2,}\S", text, re.MULTILINE)
    assert groups == ["Set up", "Index", "Search", "Help"]
    assert rows[:8] == ["config", "worker", "sync", "status", "remove", "search", "mcp", "web"]
    assert "Start here" in text
    assert "Start here" not in help_for(["status"], capsys)


def commands(app: cyclopts.App, path: tuple[str, ...] = ()) -> Iterator[tuple[str, ...]]:
    """Every command's path, e.g. ("config", "set"), the root's as ()."""
    yield path
    for name in app:
        if not name.startswith("-"):
            yield from commands(app[name], (*path, name))


def command_at(path: tuple[str, ...]) -> cyclopts.App:
    command = cli.app
    for name in path:
        command = command[name]
    return command


def examples(path: tuple[str, ...]) -> list[str]:
    return re.findall(r"`(gh-issue [^`]+)`", command_at(path).help_epilogue or "")


EXAMPLES = [line for path in commands(cli.app) for line in examples(path)]


def test_the_commands_with_options_to_learn_show_examples_in_their_help() -> None:
    shown = {path for path in commands(cli.app) if examples(path)}
    assert {("sync",), ("search",), ("remove",), ("config",), ("mcp",), ("web",)} <= shown


@pytest.mark.parametrize("line", EXAMPLES)
def test_every_example_in_the_help_is_a_command_line_that_works(line: str) -> None:
    """So an example can't go stale when an option changes."""
    words = shlex.split(line.split("  #")[0])
    cli.app.parse_args(words[1:], exit_on_error=False, print_error=False)


def test_list_options_offer_no_empty_variants(capsys: pytest.CaptureFixture[str]) -> None:
    """cyclopts adds --empty-repo beside --repo unless told not to; it only
    clutters the help."""
    assert "--empty" not in help_for(["search"], capsys)


# search and status --------------------------------------------------------------


def test_search_takes_its_words_unquoted_and_repeatable_filters() -> None:
    command, bound, _ = cli.app.parse_args(
        ["search", "event", "key", "--repo", "a/b", "--repo", "c/d", "--label", "bug", "-k", "3"]
    )
    assert command is cli.search
    assert bound.args == ("event", "key")
    assert bound.kwargs == {"repo": ["a/b", "c/d"], "label": ["bug"], "limit": 3}


def test_search_takes_the_short_flags_gh_issue_list_does() -> None:
    _, bound, _ = cli.app.parse_args(
        ["search", "x", "-R", "a/b", "-s", "closed", "-l", "bug", "-A", "ann", "-L", "5"]
    )
    assert bound.kwargs == {
        "repo": ["a/b"],
        "state": "closed",
        "label": ["bug"],
        "author": "ann",
        "limit": 5,
    }


def test_search_hands_the_options_to_the_query_layer(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    asked: list[tuple[str, SearchFilters | None, int]] = []

    def fake_search(text: str, filters: SearchFilters | None = None, limit: int = 10) -> list[Any]:
        asked.append((text, filters, limit))
        return []

    monkeypatch.setattr(storage, "index_problem", lambda: None)
    monkeypatch.setattr(query, "search", fake_search)

    cli.search(
        "event",
        "key",
        repo=["a/b"],
        state="closed",
        label=["bug"],
        since=datetime.date(2026, 1, 31),
        limit=3,
    )

    assert asked == [
        (
            "event key",
            {"repos": ["a/b"], "state": "closed", "labels": ["bug"], "since": "2026-01-31"},
            3,
        )
    ]
    assert capsys.readouterr().out.strip() == "No matches."


def test_words_that_are_only_spaces_are_a_message_not_a_traceback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The one refusal the query layer still makes that the command line can't:
    `gh-issue search "  "` has words, but nothing to search for."""
    monkeypatch.setattr(storage, "index_problem", lambda: None)

    with pytest.raises(SystemExit, match="query is empty"):
        cli.search("  ")


@pytest.mark.parametrize("command", ["search", "status", "remove", "web"])
def test_each_says_what_is_wrong_with_the_index_before_trying(
    monkeypatch: pytest.MonkeyPatch, command: str
) -> None:
    monkeypatch.setattr(storage, "index_problem", lambda: "Qdrant is not reachable")
    with pytest.raises(SystemExit, match="Qdrant is not reachable"):
        if command == "search":
            cli.search("x")
        elif command == "status":
            cli.status()
        elif command == "remove":
            cli.remove("a/b", yes=True)
        else:
            cli.web()


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
    monkeypatch.setattr(storage, "index_problem", lambda: None)
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

    monkeypatch.setattr(storage, "index_problem", lambda: None)
    monkeypatch.setattr(remove, "run", fake_run)

    cli.remove("a/b", yes=True)

    assert handed == [("a/b", True)]


# web ----------------------------------------------------------------------------

ONE_ISSUE: Issue = {
    "number": 1,
    "title": "Anything",
    "body": "One issue is enough for the repo to count as synced.",
    "state": "open",
    "labels": [],
    "user": {"login": "reporter"},
    "created_at": "2026-01-01T00:00:00Z",
    "comments": 0,
}


@pytest.fixture
def index(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Store]:
    """A real, empty embedded index in place of the Qdrant server."""
    store = Store(dim=DIM, path=str(tmp_path / "qdrant"))
    monkeypatch.setattr(storage, "store", lambda: store)
    monkeypatch.setattr(query, "store", lambda: store)
    yield store
    store.close()


def test_web_refuses_to_start_with_nothing_synced(index: Store) -> None:
    """Otherwise the page opens with no repo to choose."""
    with pytest.raises(
        SystemExit, match=r"Nothing is synced yet\. Sync a repo with: gh-issue sync owner/name"
    ):
        cli.web()


def test_web_serves_the_page_where_it_was_told_to(
    index: Store, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """uvicorn.run is the one thing replaced: it blocks until the process is
    stopped. The browser test runs the real uvicorn server."""
    index.upsert(embed_chunks(chunks_for("a/b", ONE_ISSUE, [])))
    served: dict[str, Any] = {}

    def run(app: Any, **options: Any) -> None:
        served.update(options, repos=TestClient(app).get("/api/repos").json())

    monkeypatch.setattr("uvicorn.run", run)

    cli.web(port=9000)

    assert (served["host"], served["port"]) == ("127.0.0.1", 9000)
    assert served["repos"] == {"repos": ["a/b"]}
    assert "http://127.0.0.1:9000" in capsys.readouterr().out


def test_web_loads_the_model_before_it_serves(
    index: Store, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Otherwise the first search waits for the model. Measured on 2026-10-11:
    1.9 s for the first search and 0.05 s for the ones after it, so on a first
    try the index answered after GitHub did."""
    index.upsert(embed_chunks(chunks_for("a/b", ONE_ISSUE, [])))
    embed._model.cache_clear()
    loaded: list[int] = []

    def run(app: Any, **options: Any) -> None:
        loaded.append(embed._model.cache_info().currsize)

    monkeypatch.setattr("uvicorn.run", run)

    cli.web()

    assert loaded == [1]
