"""The environment variables the package reads itself, each in one place.

The CLI copies ~/.gh-issue/config.toml into the environment at start-up (see
`interface/cli/config.py`), so these see the file's values too. The Inngest SDK reads its
own: INNGEST_DEV, INNGEST_EVENT_KEY and INNGEST_SIGNING_KEY.
"""

import os
from pathlib import Path

DEFAULT_QDRANT_URL = "http://localhost:6333"


def home() -> Path:
    """Where this machine's state lives: the settings file, the sync records,
    the worker's heartbeat and the embedding models. GH_ISSUE_HOME moves all of
    it. It can't be set in the settings file, since the file is found here."""
    return Path(os.environ.get("GH_ISSUE_HOME") or Path.home() / ".gh-issue")


def model_cache() -> str:
    """Where the embedding models are downloaded to. fastembed defaults to the
    system temp directory, which gets cleaned, and then the next run pays for
    the 65 MB download again. GH_ISSUE_MODEL_CACHE moves it, which the tests use
    to share one download."""
    return os.environ.get("GH_ISSUE_MODEL_CACHE") or str(home() / "models")


def qdrant_url() -> str:
    """Where the index lives: the qdrant.url setting."""
    return os.environ.get("QDRANT_URL", DEFAULT_QDRANT_URL)


def github_token() -> str | None:
    """The github.token setting; None if it is not set or is blank."""
    return os.environ.get("GITHUB_TOKEN") or None
