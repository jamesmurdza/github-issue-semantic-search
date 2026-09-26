"""Tests for how the search benchmark scores a method's results."""

from benchmark.scoring import Rank, difference_interval, macro, rank_of, summarise


def test_rank_counts_each_issue_once_at_its_best_chunk() -> None:
    # Issue 5 took the first two places; issue 3 is the third issue shown.
    assert rank_of(3, [5, 5, 7, 5, 3]) == 3


def test_rank_is_none_when_the_issue_is_not_in_the_top_ten() -> None:
    assert rank_of(99, [*range(1, 11), 99]) is None
    assert rank_of(99, [1, 2, 3]) is None


def test_summarise() -> None:
    summary = summarise([1, 2, None, 4])
    assert summary["mrr"] == (1 + 1 / 2 + 0 + 1 / 4) / 4
    assert summary["first"] == 1 / 4
    assert summary["top10"] == 3 / 4


def test_every_group_counts_the_same_however_many_queries_it_has() -> None:
    assert macro({"big": [1] * 10, "small": [None]}) == {"mrr": 0.5, "first": 0.5, "top10": 0.5}


def test_difference_interval_of_identical_results_is_zero() -> None:
    ranks: dict[str, list[Rank]] = {"a": [1, None, 3], "b": [2, 2]}
    assert difference_interval(ranks, ranks) == (0.0, 0.0)


def test_difference_interval_covers_which_queries_happened_to_be_asked() -> None:
    # One win in four queries: resamples that leave the win out show no
    # difference; ones that draw it three times show 0.75 (drawing it three or
    # more times happens 5% of the time, four times 0.4%).
    better: dict[str, list[Rank]] = {"a": [1, None, None, None]}
    worse: dict[str, list[Rank]] = {"a": [None, None, None, None]}
    assert difference_interval(better, worse) == (0.0, 0.75)
