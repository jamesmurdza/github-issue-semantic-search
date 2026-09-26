"""Run every search method on every benchmark query, and print how each did.

  uv run python -m benchmark.run [queries.json]

Reads benchmark/queries.json (see make_queries.py) and the index at the
qdrant.url setting, which must hold the repos those queries are about, synced
by `gh-issue sync`. Settings come from `gh-issue config` and the environment,
as for gh-issue itself.
Prints Markdown tables; README.md shows the last run's.

Each method is one Qdrant query per benchmark query: the query `gh-issue search`
sends, or that query with one thing changed. Two filters are added to every
one: the query's repo, so other repos in the index do not matter, and the
chunks the query hides. Before scoring anything, the shipped method is checked
against `Store.search` itself.
"""

import json
import sys
from collections import defaultdict
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict

from qdrant_client import QdrantClient, models

from benchmark.make_queries import QUERIES, Query
from benchmark.scoring import Rank, difference_interval, macro, rank_of, summarise
from gh_issue import config
from gh_issue.embed import embed_query, embed_texts
from gh_issue.index.store import CANDIDATES, COLLECTION, DENSE, KEYWORDS, best_first
from gh_issue.settings import qdrant_url, store


class Vectors(TypedDict):
    dense: list[float]
    # The dense vector as it was before the fix: the query embedded without
    # BGE's instruction in front.
    plain: list[float]
    sparse: models.SparseVector


# Each chunk found, best first, as its issue number and its score.
Found = list[tuple[int, float]]

# Takes a query's vectors and filter.
Search = Callable[[QdrantClient, Vectors, models.Filter], Found]


def in_order(points: list[models.ScoredPoint]) -> Found:
    """In the order `gh-issue search` lists them, equal scores included."""
    return [
        (point.payload["issue_number"], point.score)
        for point in best_first(points)
        if point.payload
    ]


def one_vector(using: str, pick: Callable[[Vectors], list[float] | models.SparseVector]) -> Search:
    """Search with one of the two vectors alone."""

    def search(client: QdrantClient, vectors: Vectors, where: models.Filter) -> Found:
        points = client.query_points(
            COLLECTION,
            query=pick(vectors),
            using=using,
            query_filter=where,
            limit=2 * CANDIDATES,
            with_payload=["repo", "issue_number"],
        ).points
        return in_order(points)

    return search


def fused(fusion: models.FusionQuery | models.RrfQuery, candidates: int = CANDIDATES) -> Search:
    """Search with both vectors, fused inside Qdrant: what `Store.search` does."""

    def search(client: QdrantClient, vectors: Vectors, where: models.Filter) -> Found:
        points = client.query_points(
            COLLECTION,
            prefetch=[
                models.Prefetch(
                    query=vectors["dense"], using=DENSE, limit=candidates, filter=where
                ),
                models.Prefetch(
                    query=vectors["sparse"], using=KEYWORDS, limit=candidates, filter=where
                ),
            ],
            query=fusion,
            limit=2 * candidates,
            with_payload=["repo", "issue_number"],
        ).points
        return in_order(points)

    return search


def rrf(k: int | None = None, dense: float = 1, bm25: float = 1) -> models.RrfQuery:
    """Reciprocal rank fusion: each list adds a score by rank alone. The weights
    let one list count for more."""
    return models.RrfQuery(rrf=models.Rrf(k=k, weights=[dense, bm25]))


DBSF = models.FusionQuery(fusion=models.Fusion.DBSF)
SHIPPED = "DBSF (shipped)"

METHODS: dict[str, Search] = {
    "dense, without the query instruction": one_vector(DENSE, lambda v: v["plain"]),
    "dense": one_vector(DENSE, lambda v: v["dense"]),
    "BM25": one_vector(KEYWORDS, lambda v: v["sparse"]),
    "RRF": fused(rrf()),
    "RRF, k=60": fused(rrf(k=60)),
    "RRF, dense counts 3x": fused(rrf(dense=3)),
    "RRF, dense counts 2x": fused(rrf(dense=2)),
    "RRF, BM25 counts 2x": fused(rrf(bm25=2)),
    "RRF, BM25 counts 3x": fused(rrf(bm25=3)),
    SHIPPED: fused(DBSF),
    "DBSF, 25 candidates a side": fused(DBSF, candidates=25),
    "DBSF, 50 candidates a side": fused(DBSF, candidates=50),
    "DBSF, 200 candidates a side": fused(DBSF, candidates=200),
}

# The methods shown group by group.
BY_GROUP = ["dense", "BM25", "RRF", SHIPPED]


def match(key: str, value: str | int) -> models.FieldCondition:
    return models.FieldCondition(key=key, match=models.MatchValue(value=value))


def where(query: Query) -> models.Filter:
    """Only the query's repo, less the chunks it hides."""
    hidden: list[models.Condition] = []
    if hide := query["hide"]:
        parts: list[models.Condition] = [match("issue_number", hide["issue"])]
        if "kind" in hide:
            parts.append(match("kind", hide["kind"]))
        hidden.append(models.Filter(must=parts))
    return models.Filter(must=[match("repo", query["repo"])], must_not=hidden)


def embed(queries: list[Query]) -> list[Vectors]:
    plain = embed_texts([query["text"] for query in queries])
    vectors: list[Vectors] = []
    for query, without in zip(queries, plain, strict=True):
        both = embed_query(query["text"])
        vectors.append(
            {
                "dense": both["dense"],
                "plain": without,
                "sparse": models.SparseVector(**both["sparse"]),
            }
        )
    return vectors


def check_shipped(client: QdrantClient, queries: list[Query], vectors: list[Vectors]) -> None:
    """The shipped method must rank issues exactly as `gh-issue search` does,
    with the same scores, or its row would describe some other search. Only
    queries that hide nothing can be checked: the product has no filter for
    hiding."""
    checked = 0
    for query, vector in zip(queries, vectors, strict=True):
        if query["hide"]:
            continue
        hits = store().search(
            vector["dense"],
            {"indices": vector["sparse"].indices, "values": vector["sparse"].values},
            filters={"repos": [query["repo"]]},
        )
        best: dict[int, float] = {}
        for issue, score in METHODS[SHIPPED](client, vector, where(query)):
            best.setdefault(issue, score)
        if list(best.items())[: len(hits)] != [(hit["issue_number"], hit["score"]) for hit in hits]:
            raise SystemExit(f"{SHIPPED} ranks {query['text']!r} differently from Store.search")
        checked += 1
    print(f"{SHIPPED} ranks all {checked} queries that hide nothing as `gh-issue search` does.\n")


def main(path: Path = QUERIES) -> None:
    queries: list[Query] = json.loads(path.read_text(encoding="utf-8"))
    client = QdrantClient(url=qdrant_url(), timeout=60)
    vectors = embed(queries)
    check_shipped(client, queries, vectors)

    # ranks[method][group] lists each of the group's queries' ranks, in order.
    ranks: dict[str, dict[str, list[Rank]]] = {name: defaultdict(list) for name in METHODS}
    for query, vector in zip(queries, vectors, strict=True):
        group = f"{query['repo']} | {query['kind']}"
        for name, search in METHODS.items():
            found = search(client, vector, where(query))
            ranks[name][group].append(rank_of(query["target"], [issue for issue, _ in found]))

    print("| method | MRR@10 | right issue first | in the top 10 | vs DBSF, 95% interval |")
    print("|---|---:|---:|---:|---|")
    for name, groups in ranks.items():
        overall = macro(groups)
        if name == SHIPPED:
            versus = ""
        else:
            low, high = difference_interval(groups, ranks[SHIPPED])
            gap = overall["mrr"] - macro(ranks[SHIPPED])["mrr"]
            versus = f"{gap:+.3f} [{low:+.3f}, {high:+.3f}]"
        print(
            f"| {name} | {overall['mrr']:.3f} | {overall['first']:.0%} | {overall['top10']:.0%} "
            f"| {versus} |"
        )

    print("\nMRR@10 by group\n")
    print("| repo | kind | queries | " + " | ".join(BY_GROUP) + " |")
    print("|---|---|---:|" + "---:|" * len(BY_GROUP))
    for group in sorted(ranks[SHIPPED]):
        repo, kind = group.split(" | ")
        # Bold the best, and anything that ties with it as printed.
        scores = [f"{summarise(ranks[name][group])['mrr']:.3f}" for name in BY_GROUP]
        best = max(scores)
        cells = [f"**{score}**" if score == best else score for score in scores]
        print(f"| {repo} | {kind} | {len(ranks[SHIPPED][group])} | " + " | ".join(cells) + " |")


if __name__ == "__main__":
    config.apply()
    main(*map(Path, sys.argv[1:2]))
