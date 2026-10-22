"""The shapes we read out of GitHub's REST responses.

These describe only the fields this project touches, not everything GitHub
sends. Responses arrive as `Any` from `response.json()`, so declaring them here
is what gives the rest of the codebase real types to check against.
"""

from typing import Any, NotRequired, TypedDict


class User(TypedDict):
    login: str


class Label(TypedDict):
    name: str


class Issue(TypedDict):
    """One issue as the listing endpoint returns it.

    `pull_request` is the field that matters most by its absence: `/issues`
    returns pull requests too, and this key is the only thing distinguishing
    them. On `inngest/inngest` that is 4,496 pull requests against 313 issues,
    so dropping them is not an optimisation.
    """

    number: int
    title: str
    body: str | None
    state: str
    labels: list[Label]
    user: User
    created_at: str
    comments: int
    pull_request: NotRequired[dict[str, Any]]


class Comment(TypedDict):
    """One comment, trimmed to what chunking reads.

    GitHub sends about ten times this, mostly URLs. The trim matters because
    this travels as the return value of an Inngest step, which is capped.
    """

    id: int
    body: str
    user: User
    created_at: str
