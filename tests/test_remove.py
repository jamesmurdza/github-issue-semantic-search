"""`gh-issue remove`: what it deletes, when it asks, and in what order.

A real embedded Qdrant and real record files; only the terminal is faked.
"""

from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path

import pytest

from gh_issue import remove as mod
from gh_issue import settings
from gh_issue.chunking import EmbeddedChunk
from gh_issue.index.store import Store

REPO = "owner/name"
OTHER = "other/repo"
DIM = 4


def chunk(repo: str, number: int, kind: str = "body") -> EmbeddedChunk:
    suffix = "body" if kind == "body" else f"comment:{number}"
    return {
        "id": f"{repo}#{number}:{suffix}",
        "content_hash": "h",
        "text": "some text",
        "meta": {
            "title": "a title",
            "state": "open",
            "labels": [],
            "repo": repo,
            "issue_number": number,
            "kind": kind,
            "author": "someone",
            "created_at": "2026-01-01T00:00:00Z",
        },
        "vector": [0.1, 0.2, 0.3, 0.4],
    }


class WatchedStore(Store):
    """The real store, noting whether the repo still looked synced at the
    moment its chunks were deleted."""

    synced_when_deleted: bool | None = None

    def delete_repo(self, repo: str) -> None:
        self.synced_when_deleted = settings.is_synced(repo)
        super().delete_repo(repo)


@dataclass
class Rig:
    store: WatchedStore
    answers: list[str] = field(default_factory=list)
    asked: list[str] = field(default_factory=list)
    said: list[str] = field(default_factory=list)

    def synced(self, repo: str) -> None:
        """A repo as `sync` and a finished backfill leave it: records, a failure
        marker, and two issues in the index, one with a comment."""
        settings.cursors().set(repo, "2026-09-25T10:00:00Z")
        settings.backfills().set(
            {"repo": repo, "chain": "c1", "next": None, "finished": True, "at": "t"}
        )
        settings.failures().add(
            {"repo": repo, "number": 9, "error": "boom", "run_id": "r", "at": "t"}
        )
        self.store.upsert([chunk(repo, 1), chunk(repo, 1, "comment"), chunk(repo, 2)])

    def ask(self, question: str) -> str:
        self.asked.append(question)
        if not self.answers:
            raise EOFError  # what input() raises with no terminal attached
        return self.answers.pop(0)

    def remove(self, repo: str = REPO, *, yes: bool = False) -> None:
        mod.run(repo, yes=yes, say=self.said.append, ask=self.ask)

    def still_there(self, repo: str = REPO) -> dict[str, bool]:
        return {
            "chunks": repo in self.store.stats(),
            "cursor": settings.cursors().get(repo) is not None,
            "backfill": settings.backfills().get(repo) is not None,
            "failures": bool(settings.failures().for_repo(repo)),
        }


ALL_GONE = {"chunks": False, "cursor": False, "backfill": False, "failures": False}
ALL_THERE = {"chunks": True, "cursor": True, "backfill": True, "failures": True}


@pytest.fixture
def rig(tmp_path: Path, data_dir: Path, monkeypatch: pytest.MonkeyPatch) -> Iterator[Rig]:
    store = WatchedStore(dim=DIM, path=str(tmp_path / "qdrant"))
    monkeypatch.setattr(mod, "store", lambda: store)
    yield Rig(store=store)
    store.close()


def test_removing_deletes_the_chunks_and_every_record_of_that_repo_only(rig: Rig) -> None:
    rig.synced(REPO)
    rig.synced(OTHER)

    rig.remove(yes=True)

    assert rig.still_there(REPO) == ALL_GONE
    assert rig.still_there(OTHER) == ALL_THERE
    assert rig.said == [
        f"Removed {REPO}. Work already queued for it will find it gone and do nothing."
    ]


def test_it_asks_first_and_says_what_will_go(rig: Rig) -> None:
    rig.synced(REPO)
    rig.answers = ["y"]

    rig.remove()

    (question,) = rig.asked
    assert "3 chunks (2 issues)" in question
    assert question.rstrip().endswith("[y/N]")
    assert rig.still_there() == ALL_GONE


@pytest.mark.parametrize("answer", ["", "n", "no", "nope"])
def test_anything_but_yes_keeps_the_repo(rig: Rig, answer: str) -> None:
    rig.synced(REPO)
    rig.answers = [answer]

    rig.remove()

    assert rig.still_there() == ALL_THERE
    assert rig.said == ["Nothing removed."]


def test_with_no_terminal_to_ask_on_it_refuses_and_says_how(rig: Rig) -> None:
    """A script without --yes: input() hits the end of stdin at once."""
    rig.synced(REPO)

    with pytest.raises(SystemExit, match="--yes"):
        rig.remove()

    assert rig.still_there() == ALL_THERE


def test_yes_removes_without_asking(rig: Rig) -> None:
    rig.synced(REPO)

    rig.remove(yes=True)

    assert rig.asked == []


def test_a_repo_not_synced_here_is_refused_before_asking(rig: Rig) -> None:
    rig.synced(OTHER)

    with pytest.raises(SystemExit, match=f"{REPO} is not synced here"):
        rig.remove()

    assert rig.asked == []
    assert rig.still_there(OTHER) == ALL_THERE


def test_the_records_go_before_the_chunks(rig: Rig) -> None:
    """Each function checks the records before doing anything, so once they
    are gone nothing queued for the repo writes chunks again. Deleting the
    chunks first would leave a window where a run could put some back."""
    rig.synced(REPO)

    rig.remove(yes=True)

    assert rig.store.synced_when_deleted is False


def test_chunks_left_with_no_records_can_still_be_removed(rig: Rig) -> None:
    """One gap is left: a write already under way as the records go can land
    after the chunks are deleted. `status` then still lists the repo, and
    running remove again clears it."""
    rig.store.upsert([chunk(REPO, 1)])

    rig.remove(yes=True)

    assert rig.still_there() == ALL_GONE
