"""Turning an issue into chunks. Which chunks a run re-embeds, deletes or
refreshes is tested through the run itself, in test_index_issue.py."""

from gh_issue.github.types import Comment, Issue
from gh_issue.index.chunking import chunks_for, issue_meta, metadata_outdated

REPO = "owner/name"
ISSUE: Issue = {
    "number": 412,
    "title": "App freezes when I scroll fast",
    "body": "Happens on long lists.",
    "state": "closed",
    "labels": [{"name": "bug"}, {"name": "perf"}],
    "user": {"login": "reporter"},
    "created_at": "2026-01-01T00:00:00Z",
    "comments": 1,
}
COMMENTS: list[Comment] = [
    {
        "id": 9981,
        "body": "IntersectionObserver race condition.",
        "user": {"login": "maintainer"},
        "created_at": "2026-01-02T00:00:00Z",
    },
]


def test_body_and_each_comment_become_separate_chunks() -> None:
    chunks = chunks_for(REPO, ISSUE, COMMENTS)
    assert [c["id"] for c in chunks] == [
        "owner/name#412:body",
        "owner/name#412:comment:9981",
    ]


def test_body_chunk_text_includes_the_title() -> None:
    body = chunks_for(REPO, ISSUE, COMMENTS)[0]
    assert body["text"].startswith("App freezes when I scroll fast")
    assert "Happens on long lists." in body["text"]


def test_every_chunk_carries_filterable_metadata() -> None:
    for chunk in chunks_for(REPO, ISSUE, COMMENTS):
        meta = chunk["meta"]
        assert meta["repo"] == REPO
        assert meta["issue_number"] == 412
        assert meta["state"] == "closed"
        assert meta["labels"] == ["bug", "perf"]
        assert meta["kind"] in ("body", "comment")
        assert meta["author"]
        assert meta["created_at"]


def test_every_chunk_carries_the_issue_title() -> None:
    """A search hit on a comment is unreadable without knowing which issue it
    belongs to, and Qdrant has no joins to look it up at query time."""
    for chunk in chunks_for(REPO, ISSUE, COMMENTS):
        assert chunk["meta"]["title"] == "App freezes when I scroll fast"


def test_labels_are_stored_sorted() -> None:
    """GitHub does not promise an order, and comparing stored labels with
    current ones must not flag a reorder as a change."""
    chunks = chunks_for(REPO, {**ISSUE, "labels": [{"name": "perf"}, {"name": "bug"}]}, [])
    assert chunks[0]["meta"]["labels"] == ["bug", "perf"]


def test_issue_meta_is_the_part_every_chunk_shares() -> None:
    assert issue_meta(ISSUE) == {
        "title": "App freezes when I scroll fast",
        "state": "closed",
        "labels": ["bug", "perf"],
    }


def test_comment_chunk_carries_its_own_author_and_date() -> None:
    comment = chunks_for(REPO, ISSUE, COMMENTS)[1]
    assert comment["meta"]["author"] == "maintainer"
    assert comment["meta"]["created_at"] == "2026-01-02T00:00:00Z"
    assert comment["meta"]["kind"] == "comment"


def test_id_is_stable_across_edits_but_hash_is_not() -> None:
    before = chunks_for(REPO, ISSUE, COMMENTS)[1]
    edited: list[Comment] = [{**COMMENTS[0], "body": "Actually it was a debounce bug."}]
    after = chunks_for(REPO, ISSUE, edited)[1]
    assert before["id"] == after["id"]
    assert before["content_hash"] != after["content_hash"]


def test_missing_body_does_not_crash() -> None:
    chunks = chunks_for(REPO, {**ISSUE, "body": None}, [])
    assert chunks[0]["text"].strip() == "App freezes when I scroll fast"


def test_an_issue_with_no_labels_still_produces_a_labels_field() -> None:
    chunks = chunks_for(REPO, {**ISSUE, "labels": []}, [])
    assert chunks[0]["meta"]["labels"] == []


def test_closing_relabelling_or_retitling_is_outdated() -> None:
    """None of these change any chunk's text, so the content hash cannot see them.
    Each on its own, which the indexing run's tests do not do: they close and
    relabel together."""
    stored = issue_meta(ISSUE)
    assert metadata_outdated({**stored, "state": "open"}, [stored])
    assert metadata_outdated({**stored, "labels": ["bug"]}, [stored])
    assert metadata_outdated({**stored, "title": "Scrolling freezes"}, [stored])


def test_one_stale_copy_is_enough_to_be_outdated() -> None:
    """A run that died halfway can leave a thread with mixed copies."""
    current = issue_meta(ISSUE)
    assert metadata_outdated(current, [current, {**current, "state": "open"}])
