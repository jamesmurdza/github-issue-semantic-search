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
"""

import os
from functools import lru_cache

from fastembed import TextEmbedding

from gh_issue.chunking import Chunk, EmbeddedChunk

MODEL = "BAAI/bge-small-en-v1.5"
DIM = 384

# fastembed defaults to the system temp directory, which gets cleaned — and then
# the next run pays for the 90 MB download again. Keep it next to the project.
DEFAULT_CACHE = "./.fastembed_cache"

# Inngest caps a step's return value at 4 MiB. One 384-dim vector is roughly 8 KB
# once it is JSON, so the caller embeds in batches of this size, leaving a run
# free to index a thread of any length.
BATCH = 200


@lru_cache(maxsize=1)
def _model() -> TextEmbedding:
    """Loaded once per process, on first use.

    The first call ever downloads about 90 MB into the cache. Keeping it lazy
    means a run with nothing new to embed never pays for it.
    """
    return TextEmbedding(
        model_name=MODEL,
        cache_dir=os.environ.get("GH_ISSUE_MODEL_CACHE", DEFAULT_CACHE),
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Vectors for each text, in the order given."""
    if not texts:
        return []
    return [vector.tolist() for vector in _model().embed(texts)]


def embed_chunks(chunks: list[Chunk]) -> list[EmbeddedChunk]:
    """Attach a vector to each chunk, leaving everything else untouched."""
    vectors = embed_texts([chunk["text"] for chunk in chunks])
    return [{**chunk, "vector": vector} for chunk, vector in zip(chunks, vectors, strict=True)]
