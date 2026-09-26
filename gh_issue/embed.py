"""Embeddings, computed locally.

fastembed runs the model through onnxruntime on the CPU, so there is no API key,
no per-token cost and no rate limit of its own. GitHub's 5,000 calls an hour
stays the only budget worth managing, which is the one Inngest's throttle covers.

Local does not mean free. Measured on one real thread from `inngest/inngest`
(36 chunks averaging 780 characters), this runs at about 5.6 chunks a second —
five minutes to embed that whole repo, and roughly six hours for a repo of
25,000 issues. That is the same order as the time GitHub's rate limit already
forces, which is why the hash comparison upstream matters: it is what stops an
edited comment from re-embedding the two hundred that did not change. It is also
why `embed` is its own checkpointed step, so a run that dies afterwards does not
pay for those vectors twice.

Each chunk gets a second, much cheaper vector too: BM25, which is word counting
and needs no model to run. Search uses both, because each finds what the other
misses. Measured on 620 queries across two repos (benchmark/), either one alone
scored about 0.57 (MRR@10) and the two fused scored 0.64.
"""

import os
from functools import lru_cache
from typing import TypedDict

from fastembed import SparseEmbedding, SparseTextEmbedding, TextEmbedding

from gh_issue.chunking import Chunk, EmbeddedChunk, SparseVector
from gh_issue.config import home

MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384
SPARSE_MODEL = "Qdrant/bm25"

# BGE was trained to see short queries with this in front of them, and stored
# text without it. fastembed does not add it, so without this every query was
# embedded as if it were a stored comment: on the benchmark, dense search
# scored 0.513 instead of 0.569.
QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "


# Inngest caps a step's return value at 4 MiB. One 384-dim vector is roughly 8 KB
# once it is JSON, so the caller embeds in batches of this size, leaving a run
# free to index a thread of any length.
BATCH = 200


class QueryVectors(TypedDict):
    dense: list[float]
    sparse: SparseVector


def cache_dir() -> str:
    """Where the models are downloaded to. fastembed defaults to the system temp
    directory, which gets cleaned, and then the next run pays for the 65 MB
    download again. GH_ISSUE_MODEL_CACHE moves it, which the tests use to share
    one download."""
    return os.environ.get("GH_ISSUE_MODEL_CACHE") or str(home() / "models")


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    """Loaded once per process, on first use.

    The first call ever downloads about 65 MB into the cache. Keeping it lazy
    means a run with nothing new to embed never pays for it.
    """
    return TextEmbedding(model_name=MODEL, cache_dir=cache_dir())


@lru_cache(maxsize=1)
def _sparse_model() -> SparseTextEmbedding:
    """BM25's tokenizer, stemmer and stopword list. A small download, no model."""
    return SparseTextEmbedding(model_name=SPARSE_MODEL, cache_dir=cache_dir())


def _sparse(embedding: SparseEmbedding) -> SparseVector:
    return {"indices": embedding.indices.tolist(), "values": embedding.values.tolist()}


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Vectors for each text, in the order given."""
    if not texts:
        return []
    return [vector.tolist() for vector in _model().embed(texts)]


def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    """Attach both vectors to each chunk, leaving everything else untouched."""
    if not chunks:
        return []
    texts = [chunk["text"] for chunk in chunks]
    vectors = embed_texts(texts)
    sparse = [_sparse(embedding) for embedding in _sparse_model().embed(texts)]
    return [
        {**chunk, "vector": vector, "sparse": keywords}
        for chunk, vector, keywords in zip(chunks, vectors, sparse, strict=True)
    ]


def embed_query(text: str) -> QueryVectors:
    """Both vectors for a search. The query side of each differs from the stored
    side: BGE wants its instruction, and BM25 weighs every query word once."""
    [dense] = embed_texts([QUERY_INSTRUCTION + text])
    [keywords] = _sparse_model().query_embed(text)
    return {"dense": dense, "sparse": _sparse(keywords)}
