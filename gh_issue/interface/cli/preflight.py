"""What would make every run fail, checked before `worker` connects and before
`sync` queues anything, rather than found after the first dozen runs have
failed on it."""

import os

import httpx

from gh_issue.env import github_token
from gh_issue.functions.client import client
from gh_issue.github.api import MISSING_TOKEN
from gh_issue.index.storage import index_problem


def preflight(production: bool) -> list[str]:
    """Every problem found, in words to act on; empty if there are none."""
    problems: list[str] = []
    if not github_token():
        problems.append(MISSING_TOKEN)
    if production:
        for key, env in (
            ("inngest.signing_key", "INNGEST_SIGNING_KEY"),
            ("inngest.event_key", "INNGEST_EVENT_KEY"),
        ):
            if not os.environ.get(env):
                problems.append(
                    f"{key} is not set, and Inngest Cloud needs it. Set it with: "
                    f"gh-issue config set {key} (or, to use a local dev server "
                    "instead: gh-issue config set inngest.dev 1)"
                )
    if problem := inngest_problem(client.api_origin, production):
        problems.append(problem)
    if problem := index_problem():
        problems.append(problem)
    return problems


def inngest_problem(origin: str, production: bool) -> str | None:
    """Whether anything answers at `origin`; any answer will do. Without this a
    worker that can't reach Inngest waits in silence, since the SDK keeps trying
    to connect and says nothing while it does."""
    try:
        httpx.get(origin, timeout=5.0)
    except httpx.TransportError:
        if production:
            return f"Can't connect to Inngest Cloud at {origin}."
        return f"Can't connect to the Inngest dev server at {origin} (inngest.dev)."
    return None
