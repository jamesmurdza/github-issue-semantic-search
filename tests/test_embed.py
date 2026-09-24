from gh_issue.chunking import Chunk
from gh_issue.embed import DIM, embed_chunks, embed_texts


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b, strict=True))


def _chunk(chunk_id: str, text: str) -> Chunk:
    return {
        "id": chunk_id,
        "text": text,
        "content_hash": "h",
        "meta": {
            "title": "App freezes when I scroll fast",
            "repo": "owner/name",
            "issue_number": 412,
            "state": "open",
            "labels": [],
            "kind": "comment",
            "author": "someone",
            "created_at": "2026-01-01T00:00:00Z",
        },
    }


def test_embeds_to_the_expected_dimension() -> None:
    vectors = embed_texts(["app freezes when I scroll fast"])
    assert len(vectors) == 1
    assert len(vectors[0]) == DIM


def test_empty_input_returns_empty() -> None:
    """Called on every run that has nothing new, so it must not load the model."""
    assert embed_texts([]) == []


def test_vectors_come_back_in_input_order() -> None:
    first, second = embed_texts(["alpha text here", "completely different subject"])
    assert first != second
    assert embed_texts(["alpha text here"])[0] == first


def test_the_same_text_always_embeds_identically() -> None:
    """The hash dedupe is only worth anything if this holds."""
    assert embed_texts(["a repeated sentence"]) == embed_texts(["a repeated sentence"])


def test_similar_text_scores_closer_than_unrelated_text() -> None:
    """The whole premise: reporters write symptoms, maintainers write causes."""
    symptom, cause, unrelated = embed_texts(
        [
            "app freezes when I scroll fast through a long list",
            "IntersectionObserver race condition in the virtualized list",
            "how do I configure the billing webhook secret",
        ]
    )
    assert _dot(symptom, cause) > _dot(symptom, unrelated)


def test_embed_chunks_attaches_a_vector_and_keeps_everything_else() -> None:
    chunks = [_chunk("a", "first chunk of text"), _chunk("b", "second chunk of text")]
    embedded = embed_chunks(chunks)

    assert [c["id"] for c in embedded] == ["a", "b"]
    assert [c["text"] for c in embedded] == ["first chunk of text", "second chunk of text"]
    assert embedded[0]["meta"] == chunks[0]["meta"]
    assert len(embedded[0]["vector"]) == DIM
    assert embedded[0]["vector"] != embedded[1]["vector"]


def test_embed_chunks_of_nothing_is_empty() -> None:
    assert embed_chunks([]) == []
