"""The settings file, ~/.gh-issue/config.toml, and the `gh-issue config` commands.

Every setting also has an environment variable, and a variable that is set
wins over the file, so one run can point somewhere else without editing it:
`QDRANT_URL=http://other:6333 gh-issue status`.

At start-up the CLI copies the file's values into the environment for every
variable not already set (`apply`). Everything else then reads the
environment only (see `env.py`), and that includes the Inngest SDK, which
reads INNGEST_DEV and its keys once, when the client is built.
"""

import getpass
import os
import tomllib
from collections.abc import Callable
from pathlib import Path
from typing import Any, TypedDict

import tomli_w

from gh_issue.env import DEFAULT_QDRANT_URL, home
from gh_issue.utils import write_text


class Setting(TypedDict):
    env: str
    # Shown masked, and typed in without echo when `config set` asks for it.
    secret: bool
    default: str | None
    help: str


SETTINGS: dict[str, Setting] = {
    "github.token": {
        "env": "GITHUB_TOKEN",
        "secret": True,
        "default": None,
        "help": "A GitHub token; for public repos, the public_repo scope is enough.",
    },
    "inngest.dev": {
        "env": "INNGEST_DEV",
        "secret": False,
        "default": None,
        "help": "1 for a local Inngest dev server, or its URL; unset means Inngest Cloud.",
    },
    "inngest.event_key": {
        "env": "INNGEST_EVENT_KEY",
        "secret": True,
        "default": None,
        "help": "Inngest Cloud's event key.",
    },
    "inngest.signing_key": {
        "env": "INNGEST_SIGNING_KEY",
        "secret": True,
        "default": None,
        "help": "Inngest Cloud's signing key.",
    },
    "qdrant.api_key": {
        "env": "QDRANT_API_KEY",
        "secret": True,
        "default": None,
        "help": "A hosted Qdrant's API key, for Qdrant Cloud; a local Qdrant needs none.",
    },
    "qdrant.url": {
        "env": "QDRANT_URL",
        "secret": False,
        "default": DEFAULT_QDRANT_URL,
        "help": "Where Qdrant is.",
    },
}


class Shown(TypedDict):
    key: str
    # Masked if secret; None when not set anywhere.
    value: str | None
    # "config file", "environment (NAME)", "default", or None when not set.
    source: str | None


class ConfigError(Exception):
    pass


def config_path() -> Path:
    return home() / "config.toml"


def read() -> dict[str, str]:
    """The file's settings, as setting name -> value. Names this version does
    not know are left out."""
    path = config_path()
    if not path.exists():
        return {}
    try:
        sections = tomllib.loads(path.read_text(encoding="utf-8"))
    except tomllib.TOMLDecodeError as error:
        raise ConfigError(
            f"{path} is not valid TOML ({error}). Fix it, or delete it and set the values again."
        ) from error
    values: dict[str, str] = {}
    for section, entries in sections.items():
        if not isinstance(entries, dict):
            continue
        for name, value in entries.items():
            key = f"{section}.{name}"
            if key in SETTINGS:
                values[key] = _text(value)
    return values


def _text(value: Any) -> str:
    # `dev = true` written by hand means what INNGEST_DEV=1 means.
    if isinstance(value, bool):
        return "1" if value else "0"
    return str(value)


def set_value(key: str, value: str) -> None:
    _check(key)
    values = read()
    values[key] = value
    _write(values)


def unset(key: str) -> bool:
    """Remove a setting from the file. False if the file did not have it."""
    _check(key)
    values = read()
    if key not in values:
        return False
    del values[key]
    _write(values)
    return True


def _check(key: str) -> None:
    if key not in SETTINGS:
        raise ConfigError(f"There is no setting {key!r}. The settings are: {', '.join(SETTINGS)}.")


def _write(values: dict[str, str]) -> None:
    sections: dict[str, dict[str, str]] = {}
    for key in sorted(values):
        section, name = key.split(".")
        sections.setdefault(section, {})[name] = values[key]
    write_text(config_path(), tomli_w.dumps(sections))


def apply() -> None:
    """Copy the file's settings into the environment wherever the environment
    does not set them. An empty variable counts as not set."""
    for key, value in read().items():
        env = SETTINGS[key]["env"]
        if not os.environ.get(env):
            os.environ[env] = value


def shown() -> list[Shown]:
    """Every setting as it stands now: its value, masked if secret, and where
    it came from. Right after `apply`, a variable holding exactly the file's
    value is taken to be the file's."""
    in_file = read()
    rows: list[Shown] = []
    for key, setting in SETTINGS.items():
        from_env = os.environ.get(setting["env"])
        value: str | None
        source: str | None
        if from_env and from_env != in_file.get(key):
            value, source = from_env, f"environment ({setting['env']})"
        elif key in in_file:
            value, source = in_file[key], "config file"
        elif setting["default"] is not None:
            value, source = setting["default"], "default"
        else:
            value, source = None, None
        if value is not None and setting["secret"]:
            value = _mask(value)
        rows.append({"key": key, "value": value, "source": source})
    return rows


def _mask(secret: str) -> str:
    """Enough to tell two tokens apart, not enough to use one."""
    return "****" + secret[-4:] if len(secret) >= 12 else "****"


# How `config set` asks for a value left off the command line. A secret is
# typed without echo, which also keeps it out of the shell's history.
ask: Callable[[str], str] = input
ask_secret: Callable[[str], str] = getpass.getpass


def run_set(key: str, value: str | None, *, say: Callable[[str], None] = print) -> None:
    """`gh-issue config set`: save a value, asking for it if not given."""
    _check(key)
    setting = SETTINGS[key]
    if value is None:
        value = (
            ask_secret(f"{key} (not shown as you type): ") if setting["secret"] else ask(f"{key}: ")
        )
    value = value.strip()
    if not value:
        raise ConfigError(f"No value given, so {key} is unchanged.")
    before = read().get(key)
    set_value(key, value)
    say(f"Saved {key} in {config_path()}.")
    # Right after start-up, a variable holding the file's old value came from
    # the file itself. One holding anything else was set outside, and wins.
    outside = os.environ.get(setting["env"])
    if outside and outside not in (before, value):
        say(
            f"Note: {setting['env']} is set in the environment, and wins over the file while it is."
        )


def run_unset(key: str, *, say: Callable[[str], None] = print) -> None:
    """`gh-issue config unset`."""
    if unset(key):
        say(f"Removed {key} from {config_path()}.")
    else:
        say(f"{key} is not in {config_path()}; nothing to remove.")
