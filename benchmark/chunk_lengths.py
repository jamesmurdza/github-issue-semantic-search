"""How much of the index the dense model never reads.

  uv run python -m benchmark.chunk_lengths

bge-small reads the first 512 tokens of a text and ignores the rest, so a long
comment is embedded by its opening alone. BM25 counts every word. This is one
reason dense search trails BM25 on these repos (README.md).
"""

import statistics
from typing import Any

from tokenizers import Tokenizer

from benchmark.make_queries import REPOS, indexed_chunks
from gh_issue.index.embed import _model
from gh_issue.interface.cli import config

LIMIT = 512


def main() -> None:
    # A copy of the tokenizer fastembed feeds the model, with its cut-off at
    # LIMIT removed, and its padding: it pads a batch to its longest text.
    # fastembed's types do not mention that it holds one.
    loaded: Any = _model().model
    tokenizer = Tokenizer.from_str(loaded.tokenizer.to_str())
    tokenizer.no_truncation()
    tokenizer.no_padding()
    print("| repo | chunks | median tokens | longer than 512 | tokens past 512 |")
    print("|---|---:|---:|---:|---:|")
    for repo in REPOS:
        texts = [chunk["text"] for chunk in indexed_chunks(repo, ["text"])]
        lengths = [len(encoding.ids) for encoding in tokenizer.encode_batch(texts)]
        longer = sum(length > LIMIT for length in lengths) / len(lengths)
        unread = sum(max(length - LIMIT, 0) for length in lengths) / sum(lengths)
        print(
            f"| {repo} | {len(lengths):,} | {statistics.median(lengths):.0f} "
            f"| {longer:.0%} | {unread:.0%} |"
        )


if __name__ == "__main__":
    config.apply()
    main()
