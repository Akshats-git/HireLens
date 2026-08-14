"""Unit tests for the ranking and skill-extraction metrics."""

import pytest

from src.evaluation.metrics import (
    average_precision_at_k,
    dcg_at_k,
    ndcg_at_k,
    precision_at_k,
    reciprocal_rank,
    token_level_f1,
)


# ── Precision@K ───────────────────────────────────────────────────────────────


def test_precision_at_k_counts_hits_in_the_top_k():
    assert precision_at_k({0, 1}, [0, 1, 2, 3], k=2) == 1.0
    assert precision_at_k({0, 1}, [0, 2, 3, 1], k=2) == 0.5
    assert precision_at_k({5}, [0, 1, 2], k=3) == 0.0


def test_precision_at_k_divides_by_k_not_by_the_result_count():
    """With one relevant item, P@5 is capped at 0.2 by construction."""
    assert precision_at_k({0}, [0, 1, 2, 3, 4], k=5) == pytest.approx(0.2)


def test_precision_at_k_is_zero_for_a_non_positive_cutoff():
    assert precision_at_k({0}, [0, 1], k=0) == 0.0


# ── DCG / NDCG ────────────────────────────────────────────────────────────────


def test_dcg_discounts_by_rank():
    assert dcg_at_k([1, 0], 2) > dcg_at_k([0, 1], 2)


def test_dcg_honours_the_cutoff():
    assert dcg_at_k([0, 0, 1], 2) == 0.0
    assert dcg_at_k([0, 0, 1], 3) > 0.0


def test_ndcg_is_one_for_a_perfect_ranking():
    assert ndcg_at_k([1, 1, 0, 0], 4) == pytest.approx(1.0)


def test_ndcg_is_below_one_for_an_imperfect_ranking():
    assert ndcg_at_k([0, 1, 1, 0], 4) < 1.0


def test_ndcg_is_zero_when_nothing_is_relevant():
    assert ndcg_at_k([0, 0, 0], 3) == 0.0


def test_ndcg_stays_within_bounds():
    for relevances in ([1, 0, 1, 0], [0, 0, 1], [1, 1, 1]):
        assert 0.0 <= ndcg_at_k(relevances, 10) <= 1.0


# ── MRR ───────────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "retrieved, expected",
    [([0, 1, 2], 1.0), ([1, 0, 2], 0.5), ([1, 2, 0], pytest.approx(1 / 3))],
)
def test_reciprocal_rank_tracks_the_first_hit(retrieved, expected):
    assert reciprocal_rank({0}, retrieved) == expected


def test_reciprocal_rank_is_zero_without_a_hit():
    assert reciprocal_rank({9}, [0, 1, 2]) == 0.0


# ── MAP@K ─────────────────────────────────────────────────────────────────────


def test_average_precision_is_one_when_hits_lead_the_ranking():
    assert average_precision_at_k({0, 1}, [0, 1, 2, 3], k=4) == pytest.approx(1.0)


def test_average_precision_penalises_late_hits():
    leading = average_precision_at_k({0, 1}, [0, 1, 2, 3], k=4)
    trailing = average_precision_at_k({0, 1}, [2, 3, 0, 1], k=4)
    assert trailing < leading


def test_average_precision_is_zero_without_relevant_items():
    assert average_precision_at_k(set(), [0, 1, 2], k=3) == 0.0


def test_average_precision_normalises_by_the_reachable_hits():
    """With three relevant items but a cutoff of 2, two hits is a perfect score."""
    assert average_precision_at_k({0, 1, 2}, [0, 1, 2], k=2) == pytest.approx(1.0)


# ── Token-level F1 ────────────────────────────────────────────────────────────


def test_f1_is_one_for_an_exact_match():
    scores = token_level_f1([{"python", "sql"}], [{"python", "sql"}])
    assert scores == {"precision": 1.0, "recall": 1.0, "f1": 1.0}


def test_f1_is_zero_when_nothing_overlaps():
    scores = token_level_f1([{"chef"}], [{"python"}])
    assert scores == {"precision": 0.0, "recall": 0.0, "f1": 0.0}


def test_precision_and_recall_capture_over_and_under_prediction():
    # Scores are rounded to four decimal places by token_level_f1.
    over = token_level_f1([{"python", "sql", "go"}], [{"python"}])
    assert over["recall"] == 1.0
    assert over["precision"] == pytest.approx(1 / 3, abs=1e-4)

    under = token_level_f1([{"python"}], [{"python", "sql", "go"}])
    assert under["precision"] == 1.0
    assert under["recall"] == pytest.approx(1 / 3, abs=1e-4)


def test_f1_is_micro_averaged_across_documents():
    scores = token_level_f1(
        [{"python"}, {"sql", "go"}],
        [{"python"}, {"sql"}],
    )
    # 3 predicted, 2 ground truth, 2 correct.
    assert scores["precision"] == pytest.approx(2 / 3, abs=1e-4)
    assert scores["recall"] == 1.0


def test_f1_handles_empty_predictions():
    assert token_level_f1([set()], [{"python"}])["f1"] == 0.0
    assert token_level_f1([], [])["f1"] == 0.0
