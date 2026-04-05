"""Regression tests for preprocessor dataframe and CSV input modes."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path
from textwrap import dedent

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "classic_methods" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from data_processing.preprocessing import Preprocessor
from pipeline import Pipeline


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).lstrip(), encoding="utf-8")


def _build_df() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "CustomerID": [1, 1, 2],
            "Item Code": ["A", "B", "A"],
            "Order Date": ["2024-01-01", "2024-01-02", "2024-01-03"],
            "Invoiced price": [10.0, 12.0, 11.0],
            "Invoiced price (TX)": [10.0, 12.0, 11.0],
            "Ordered qty": [1, 2, 1],
            "Invoiced qty (shipped)": [1, 2, 1],
            "GM%": [0.1, 0.2, 0.3],
        }
    )


def test_preprocessor_accepts_dataframe_or_csv_path() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        csv_path = Path(tmpdir) / "database.csv"
        df = _build_df()
        df.to_csv(csv_path, index=False)

        from_df = Preprocessor(
            drop_price_gt_tx=False,
            min_num_purchases_per_customer=None,
            min_num_items_per_customer=None,
            min_num_purchases_per_item_rows=None,
            min_num_unique_customers_per_item=None,
            null_rules=[],
        )
        from_path = Preprocessor(
            csv_path=str(csv_path),
            drop_price_gt_tx=False,
            min_num_purchases_per_customer=None,
            min_num_items_per_customer=None,
            min_num_purchases_per_item_rows=None,
            min_num_unique_customers_per_item=None,
            null_rules=[],
        )

        df_result, df_idx2item = from_df.fit_transform(df, impute=False)
        path_result, path_idx2item = from_path.fit_transform(impute=False)

        assert df_result.equals(path_result)
        assert df_idx2item == path_idx2item


def test_pipeline_can_use_preprocessor_without_dataframe_input_mapping() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        project_dir = root / "classic_methods"
        data_dir = project_dir / "data"
        config_dir = project_dir / "config"
        data_dir.mkdir(parents=True, exist_ok=True)
        config_dir.mkdir(parents=True, exist_ok=True)

        _build_df().to_csv(data_dir / "database.csv", index=False)

        _write_text(
            config_dir / "pipeline.yaml",
            f"""
            data:
              csv_path: {str((data_dir / "database.csv").resolve())}

            save_intermediates: false

            stage1:
              config_path: config/preprocessing_from_path.yaml
            """,
        )
        _write_text(
            config_dir / "preprocessing_from_path.yaml",
            """
            _target_: data_processing.preprocessing.Preprocessor
            method: fit_transform
            params:
              csv_path: ${data.csv_path}
              drop_price_gt_tx: false
              min_num_purchases_per_customer: null
              min_num_items_per_customer: null
              min_num_purchases_per_item_rows: null
              min_num_unique_customers_per_item: null
              null_rules: []
            """,
        )

        pipeline = Pipeline(config_dir / "pipeline.yaml")
        artifacts = pipeline.run(save_artifacts=False)

        assert "dataframe" in artifacts
        assert "idx2item" in artifacts
        assert artifacts["dataframe"].value.shape[0] == 3
        assert artifacts["idx2item"].value == {0: "A", 1: "B"}


def main() -> None:
    test_preprocessor_accepts_dataframe_or_csv_path()
    test_pipeline_can_use_preprocessor_without_dataframe_input_mapping()
    print("preprocessor input mode tests passed")


if __name__ == "__main__":
    main()
