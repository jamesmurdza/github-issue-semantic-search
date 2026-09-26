from gh_issue.chunking import Chunk
from gh_issue.embed import DIM, QUERY_INSTRUCTION, embed_chunks, embed_query, embed_texts


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


# Keywords, for the BM25 half of the search ------------------------------------


def test_embed_chunks_attaches_a_keyword_vector_too() -> None:
    [embedded] = embed_chunks([_chunk("a", "the app freezes while scrolling a long list")])

    sparse = embedded["sparse"]
    assert sparse["indices"]
    assert len(sparse["indices"]) == len(sparse["values"])
    assert embed_chunks([_chunk("a", "the app freezes while scrolling a long list")]) == [embedded]


def test_keywords_match_other_forms_of_the_same_word() -> None:
    """BM25 compares word stems, so "freezes" in a query finds "freezing"."""
    [embedded] = embed_chunks([_chunk("a", "the app keeps freezing")])
    query = embed_query("freezes")

    assert set(query["sparse"]["indices"]) & set(embedded["sparse"]["indices"])


def test_a_query_is_embedded_with_the_instruction_bge_expects() -> None:
    """BGE was trained to see short queries with this instruction in front.
    Without it, on 620 benchmark queries, dense search scored 0.502 instead of
    0.561 (MRR@10)."""
    text = "app freezes when I scroll"

    dense = embed_query(text)["dense"]

    assert dense == embed_texts([QUERY_INSTRUCTION + text])[0]
    assert dense != embed_texts([text])[0]
