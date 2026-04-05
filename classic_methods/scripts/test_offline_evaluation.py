"""Regression tests for offline recommendation evaluation helpers."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "classic_methods" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from evaluation import (
    mrr_at_k,
    ndcg_at_k,
    recall_at_k,
    remove_last_purchase,
    remove_whole_item,
)


def _build_purchase_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CustomerID": [1, 1, 1, 2, 2, 3, 4, 4, 5, 5, 5],
            "item_idx": [10, 10, 20, 30, 40, 50, 60, 60, 70, 80, 80],
            "Order Date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-02-01",
                "2024-02-01",
                "2024-03-01",
                "2024-04-01",
                "2024-04-02",
                "2024-05-01",
                "2024-05-02",
                "2024-05-02",
            ],
        }
    )


def test_remove_last_purchase_removes_exactly_one_row_per_evaluated_user() -> None:
    df = _build_purchase_df()

    result = remove_last_purchase(df, [1, 2, 3, 99])

    assert result.evaluated_user_ids == [1, 2]
    assert result.skipped_user_ids == [3, 99]
    assert result.ground_truth == {1: 20, 2: 40}
    assert len(result.masked_df) == len(df) - 2
    assert len(result.masked_df[result.masked_df["CustomerID"] == 1]) == 2
    assert len(result.masked_df[result.masked_df["CustomerID"] == 2]) == 1


def test_remove_last_purchase_uses_original_row_order_for_latest_date_ties() -> None:
    df = pd.DataFrame(
        {
            "CustomerID": [7, 7, 7],
            "item_idx": [100, 200, 300],
            "Order Date": ["2024-06-01", "2024-06-02", "2024-06-02"],
        }
    )

    result = remove_last_purchase(df, [7])

    assert result.ground_truth == {7: 300}
    assert result.masked_df["item_idx"].tolist() == [100, 200]


def test_remove_whole_item_removes_all_rows_for_selected_user_item_pair() -> None:
    df = _build_purchase_df()

    result = remove_whole_item(df, [1], seed=7)

    assert result.evaluated_user_ids == [1]
    assert result.skipped_user_ids == []
    hidden_item = result.ground_truth[1]
    remaining_user_items = result.masked_df.loc[result.masked_df["CustomerID"] == 1, "item_idx"].tolist()
    assert hidden_item in {10, 20}
    assert hidden_item not in remaining_user_items
    assert len(result.masked_df) == len(df) - int((df["CustomerID"].eq(1) & df["item_idx"].eq(hidden_item)).sum())


def test_remove_whole_item_skips_users_without_remaining_history() -> None:
    df = _build_purchase_df()

    result = remove_whole_item(df, [3, 4])

    assert result.evaluated_user_ids == []
    assert result.skipped_user_ids == [3, 4]
    assert result.ground_truth == {}
    assert result.masked_df.equals(df)


def test_remove_whole_item_is_deterministic_with_fixed_seed() -> None:
    df = _build_purchase_df()

    first = remove_whole_item(df, [1, 5], seed=123)
    second = remove_whole_item(df, [1, 5], seed=123)

    assert first.ground_truth == second.ground_truth
    assert first.masked_df.equals(second.masked_df)
    assert first.evaluated_user_ids == second.evaluated_user_ids
    assert first.skipped_user_ids == second.skipped_user_ids


def test_metrics_cover_rank_boundaries_misses_and_missing_predictions() -> None:
    predictions = {
        1: [10, 99, 98],
        2: [91, 92, 93, 94, 95, 96, 97, 98, 99, 20],
        3: [31, 32, 33],
    }
    ground_truth = {1: 10, 2: 20, 3: 30, 4: 40}

    assert math.isclose(recall_at_k(predictions, ground_truth, k=10), 0.5)
    assert math.isclose(mrr_at_k(predictions, ground_truth, k=10), (1.0 + 0.1) / 4.0)
    assert math.isclose(
        ndcg_at_k(predictions, ground_truth, k=10),
        (1.0 + (1.0 / math.log2(11.0))) / 4.0,
    )


def test_metrics_return_zero_for_empty_ground_truth_and_zero_k() -> None:
    predictions = {1: [1, 2, 3]}

    assert recall_at_k(predictions, {}, k=10) == 0.0
    assert mrr_at_k(predictions, {}, k=10) == 0.0
    assert ndcg_at_k(predictions, {}, k=10) == 0.0
    assert recall_at_k(predictions, {1: 1}, k=0) == 0.0
    assert mrr_at_k(predictions, {1: 1}, k=0) == 0.0
    assert ndcg_at_k(predictions, {1: 1}, k=0) == 0.0


def test_metrics_are_perfect_for_perfect_predictions() -> None:
    predictions = {1: [10, 20], 2: [30, 40]}
    ground_truth = {1: 10, 2: 30}

    assert recall_at_k(predictions, ground_truth, k=10) == 1.0
    assert mrr_at_k(predictions, ground_truth, k=10) == 1.0
    assert ndcg_at_k(predictions, ground_truth, k=10) == 1.0


def test_integration_mask_then_score_with_synthetic_predictions() -> None:
    df = pd.DataFrame(
        {
            "CustomerID": [1, 1, 2, 2, 3, 3],
            "item_idx": [10, 20, 30, 40, 50, 60],
            "Order Date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-01",
                "2024-01-03",
                "2024-01-01",
                "2024-01-04",
            ],
        }
    )

    split = remove_last_purchase(df, [1, 2, 3])
    predictions = {
        1: [20, 10, 30],
        2: [99, 98, 40, 97],
        3: [70, 71, 72],
    }

    assert split.ground_truth == {1: 20, 2: 40, 3: 60}
    assert math.isclose(recall_at_k(predictions, split.ground_truth, k=3), 2.0 / 3.0)
    assert math.isclose(mrr_at_k(predictions, split.ground_truth, k=3), (1.0 + (1.0 / 3.0)) / 3.0)
    assert math.isclose(
        ndcg_at_k(predictions, split.ground_truth, k=3),
        (1.0 + (1.0 / math.log2(4.0))) / 3.0,
    )


def main() -> None:
    """Execute the regression tests."""

    test_remove_last_purchase_removes_exactly_one_row_per_evaluated_user()
    test_remove_last_purchase_uses_original_row_order_for_latest_date_ties()
    test_remove_whole_item_removes_all_rows_for_selected_user_item_pair()
    test_remove_whole_item_skips_users_without_remaining_history()
    test_remove_whole_item_is_deterministic_with_fixed_seed()
    test_metrics_cover_rank_boundaries_misses_and_missing_predictions()
    test_metrics_return_zero_for_empty_ground_truth_and_zero_k()
    test_metrics_are_perfect_for_perfect_predictions()
    test_integration_mask_then_score_with_synthetic_predictions()
    print("offline evaluation tests passed")


if __name__ == "__main__":
    main()
