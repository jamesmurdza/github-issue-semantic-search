"""How one search method's results are scored.

Every benchmark query has one right answer, an issue. Search shows one hit per
issue, so a method's rank for a query is where the right issue lands once each
issue is counted once, at its best chunk.

The score is MRR@10, the mean reciprocal rank: 1 when the right issue comes
first, 1/2 when second, down to 1/10 when tenth, and 0 when it is not in the top
ten. Averaged over queries, 0.5 is roughly "second place on average", though a
few misses pull it down more than a few second places do.

Queries come in groups (a repo and a kind of query), and the groups differ in
size and difficulty. The headline number is the mean of the groups' scores, so
that 150 title queries do not outweigh 50 paraphrases.
"""

import random
import statistics
from collections.abc import Iterable, Mapping
from typing import TypedDict

DEPTH = 10

# A query's rank, or None when the right issue was not in the top DEPTH.
Rank = int | None


class Summary(TypedDict):
    mrr: float
    # How often the right issue came first, and how often it was in the top ten.
    first: float
    top10: float


def rank_of(target: int, chunk_issues: Iterable[int], depth: int = DEPTH) -> Rank:
    """Where `target` lands among the issues of a ranked list of chunks, given
    as each chunk's issue number."""
    seen: list[int] = []
    for issue in chunk_issues:
        if issue in seen:
            continue
        seen.append(issue)
        if issue == target:
            return len(seen)
        if len(seen) == depth:
            return None
    return None


def reciprocal(rank: Rank) -> float:
    return 1 / rank if rank else 0.0


def summarise(ranks: list[Rank]) -> Summary:
    return {
        "mrr": statistics.fmean(reciprocal(rank) for rank in ranks),
        "first": statistics.fmean(rank == 1 for rank in ranks),
        "top10": statistics.fmean(rank is not None for rank in ranks),
    }


def macro(groups: Mapping[str, list[Rank]]) -> Summary:
    """The mean of each group's scores."""
    summaries = [summarise(ranks) for ranks in groups.values()]
    return {
        "mrr": statistics.fmean(summary["mrr"] for summary in summaries),
        "first": statistics.fmean(summary["first"] for summary in summaries),
        "top10": statistics.fmean(summary["top10"] for summary in summaries),
    }


def macro_mrr(groups: Mapping[str, list[Rank]]) -> float:
    return macro(groups)["mrr"]


def difference_interval(
    method: Mapping[str, list[Rank]],
    baseline: Mapping[str, list[Rank]],
    rounds: int = 2000,
    seed: int = 0,
) -> tuple[float, float]:
    """A 95% interval for `macro_mrr(method) - macro_mrr(baseline)`.

    The queries are a sample; another sample would give other scores. This
    redraws each group's queries at random, with repeats, `rounds` times, and
    scores both methods on the same redrawn queries each time. If the middle 95%
    of the differences stays on one side of zero, which queries happened to be
    asked does not explain the gap.
    """
    rng = random.Random(seed)
    sizes = {group: len(ranks) for group, ranks in method.items()}
    differences = []
    for _ in range(rounds):
        picks = {group: rng.choices(range(size), k=size) for group, size in sizes.items()}
        differences.append(macro_mrr(_redraw(method, picks)) - macro_mrr(_redraw(baseline, picks)))
    differences.sort()
    return differences[int(rounds * 0.025)], differences[int(rounds * 0.975) - 1]


def _redraw(groups: Mapping[str, list[Rank]], picks: dict[str, list[int]]) -> dict[str, list[Rank]]:
    return {group: [ranks[i] for i in picks[group]] for group, ranks in groups.items()}
