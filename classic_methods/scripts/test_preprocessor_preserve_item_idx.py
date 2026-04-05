"""Regression tests for preprocessor preserve-existing-item-index mode."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "classic_methods" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_processing.preprocessing import Preprocessor


def test_preprocessor_can_impute_without_remapping_item_idx() -> None:
    df = pd.DataFrame(
        {
            "CustomerID": [1, 1, 2],
            "Item Code": ["A", "B", "A"],
            "item_idx": [7, 9, 7],
            "Order Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "Invoiced price": [10.0, None, 12.0],
            "Invoiced price (TX)": [10.0, None, 12.0],
            "Ordered qty": [1, 2, 1],
            "Invoiced qty (shipped)": [1, 2, 1],
            "GM%": [0.1, None, 0.3],
        }
    )

    preprocessor = Preprocessor(
        map_item_codes=False,
        impute=True,
        date_columns=[],
        drop_price_gt_tx=False,
        drop_missing_customer_id=False,
        drop_missing_item_code=False,
        reindex_items_at_end=False,
        min_num_purchases_per_customer=None,
        min_num_items_per_customer=None,
        min_num_purchases_per_item_rows=None,
        min_num_unique_customers_per_item=None,
        null_rules=[],
    )

    processed, idx2item = preprocessor.fit_transform(df)

    assert processed["item_idx"].tolist() == [7, 9, 7]
    assert idx2item == {7: "A", 9: "B"}
    assert processed["Invoiced price"].isna().sum() == 0


def main() -> None:
    test_preprocessor_can_impute_without_remapping_item_idx()
    print("preprocessor preserve item_idx tests passed")


if __name__ == "__main__":
    main()
