from gh_issue.chunking import chunks_for, needs_embedding, orphaned_ids
from gh_issue.github.types import Comment, Issue

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


def test_identical_text_hashes_identically() -> None:
    assert (
        chunks_for(REPO, ISSUE, COMMENTS)[0]["content_hash"]
        == (chunks_for(REPO, ISSUE, COMMENTS)[0]["content_hash"])
    )


def test_missing_body_does_not_crash() -> None:
    chunks = chunks_for(REPO, {**ISSUE, "body": None}, [])
    assert chunks[0]["text"].strip() == "App freezes when I scroll fast"


def test_an_issue_with_no_labels_still_produces_a_labels_field() -> None:
    chunks = chunks_for(REPO, {**ISSUE, "labels": []}, [])
    assert chunks[0]["meta"]["labels"] == []


def test_needs_embedding_skips_unchanged_chunks() -> None:
    chunks = chunks_for(REPO, ISSUE, COMMENTS)
    existing = {c["id"]: c["content_hash"] for c in chunks}
    assert needs_embedding(chunks, existing) == []


def test_needs_embedding_returns_only_the_changed_chunk() -> None:
    chunks = chunks_for(REPO, ISSUE, COMMENTS)
    existing = {
        chunks[0]["id"]: chunks[0]["content_hash"],
        chunks[1]["id"]: "stale-hash",
    }
    assert [c["id"] for c in needs_embedding(chunks, existing)] == [chunks[1]["id"]]


def test_needs_embedding_returns_everything_when_nothing_stored() -> None:
    chunks = chunks_for(REPO, ISSUE, COMMENTS)
    assert len(needs_embedding(chunks, {})) == 2


def test_orphaned_ids_finds_a_deleted_comment() -> None:
    chunks = chunks_for(REPO, {**ISSUE, "comments": 0}, [])
    existing = {"owner/name#412:body": "h1", "owner/name#412:comment:9981": "h2"}
    assert orphaned_ids(chunks, existing) == ["owner/name#412:comment:9981"]


def test_orphaned_ids_is_empty_when_nothing_was_removed() -> None:
    chunks = chunks_for(REPO, ISSUE, COMMENTS)
    existing = {c["id"]: c["content_hash"] for c in chunks}
    assert orphaned_ids(chunks, existing) == []


def test_an_edited_comment_is_re_embedded_but_never_orphaned() -> None:
    """The two ids do different jobs: edits change the hash, not the id."""
    chunks = chunks_for(REPO, ISSUE, COMMENTS)
    existing = {c["id"]: c["content_hash"] for c in chunks}

    edited: list[Comment] = [{**COMMENTS[0], "body": "Actually it was a debounce bug."}]
    after = chunks_for(REPO, ISSUE, edited)

    assert [c["id"] for c in needs_embedding(after, existing)] == ["owner/name#412:comment:9981"]
    assert orphaned_ids(after, existing) == []
