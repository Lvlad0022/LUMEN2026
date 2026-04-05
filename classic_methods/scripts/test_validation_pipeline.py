"""Regression tests for the validation pipeline wrapper."""

from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from textwrap import dedent

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "classic_methods" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from validation import ValidationPipeline


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).lstrip(), encoding="utf-8")


def _build_project(root: Path) -> Path:
    project_dir = root / "classic_methods"
    data_dir = project_dir / "data"
    output_dir = project_dir / "output"
    config_dir = project_dir / "config"
    data_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    helper_module = root / "validation_test_support.py"
    _write_text(
        helper_module,
        """
        from __future__ import annotations

        import numpy as np


        class PassThroughStage:
            def fit(self, df):
                customer_idx = np.array(sorted(set(int(value) for value in df["CustomerID"].tolist())), dtype=int)
                idx2item = {
                    int(item_idx): item_code
                    for item_idx, item_code in sorted(
                        {(int(item_idx), item_code) for item_idx, item_code in zip(df["item_idx"], df["Item Code"])}
                    )
                }
                return {
                    "dataframe": df.copy(),
                    "customer_idx": customer_idx,
                    "idx2item": idx2item,
                }


        class DummyRecommendationModel:
            def __init__(self, seen_items_by_user):
                self.seen_items_by_user = seen_items_by_user

            def predict(self, customer_idx: int, num_prediction: int = 10):
                seen_items = self.seen_items_by_user.get(int(customer_idx), [])
                unseen_candidates = [item for item in [10, 20, 30, 40, 50, 60] if item not in seen_items]
                return unseen_candidates[: int(num_prediction)]


        class RecommendationStage:
            def fit(self, dataframe, customer_idx):
                seen_items_by_user = {}
                for customer_id in customer_idx.tolist():
                    seen_items_by_user[int(customer_id)] = dataframe.loc[
                        dataframe["CustomerID"] == int(customer_id),
                        "item_idx",
                    ].astype(int).tolist()
                return {
                    "model": DummyRecommendationModel(seen_items_by_user),
                    "customer_idx": customer_idx,
                }
        """,
    )

    df = pd.DataFrame(
        {
            "CustomerID": [1, 1, 1, 2, 2, 2, 3, 3, 3],
            "Item Code": ["A", "B", "C", "A", "D", "E", "B", "E", "F"],
            "item_idx": [10, 20, 30, 10, 40, 50, 20, 50, 60],
            "Order Date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
            ],
        }
    )
    df.to_csv(data_dir / "processed.csv", index=False)

    _write_text(
        config_dir / "paths.yaml",
        f"""
        paths:
          data_csv: {str((data_dir / 'processed.csv').resolve())}
          processed_data_csv: {str((data_dir / 'processed.csv').resolve())}
          output_dir: {str(output_dir.resolve())}
        """,
    )
    _write_text(
        config_dir / "recommendation_pipeline.yaml",
        """
        data:
          csv_path: ${paths.processed_data_csv}

        output:
          dir: ${paths.output_dir}

        save_intermediates: false

        stage1:
          config_path: config/preprocess_stage.yaml

        stage2:
          config_path: config/recommend_stage.yaml
        """,
    )
    _write_text(
        config_dir / "preprocess_stage.yaml",
        """
        _target_: validation_test_support.PassThroughStage
        method: fit
        params: {}
        inputs:
          df: dataframe
        """,
    )
    _write_text(
        config_dir / "recommend_stage.yaml",
        """
        _target_: validation_test_support.RecommendationStage
        method: fit
        params: {}
        inputs:
          dataframe: dataframe
          customer_idx: customer_idx
        """,
    )
    _write_text(
        config_dir / "validation.yaml",
        """
        seed: ${project.seed}

        data:
          csv_path: ${paths.processed_data_csv}
          user_col: CustomerID
          item_col: item_idx

        pipeline:
          recommendation_config_path: config/recommendation_pipeline.yaml
          recommendation_k: 3

        sampling:
          users_per_repetition: 2

        masking:
          strategy: remove_whole_item

        metrics:
          k: 3

        output:
          summary_path: ${paths.output_dir}/validation_summary.json
        """,
    )
    _write_text(
        config_dir / "base.yaml",
        """
        project:
          name: validation_test
          seed: 7

        validation_bool: true
        validation_repetitions: 3

        paths:
          config_path: config/paths.yaml

        pipeline:
          config_path: config/recommendation_pipeline.yaml

        validation:
          config_path: config/validation.yaml
        """,
    )
    return config_dir / "base.yaml"


def test_validation_pipeline_runs_and_writes_summary() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sys.path.insert(0, str(root))
        try:
            base_config = _build_project(root)
            validation_pipeline = ValidationPipeline(base_config)
            summary = validation_pipeline.run()

            summary_path = validation_pipeline.config.summary_path
            assert summary_path.exists()
            saved = json.loads(summary_path.read_text(encoding="utf-8"))
            assert saved["repetitions_run"] == 3
            assert saved["users_per_repetition"] == 2
            assert saved["masking_strategy"] == "remove_whole_item"
            assert len(saved["repetition_results"]) == 3
            assert all("ndcg_at_k" in result for result in saved["repetition_results"])
            assert all("recall_at_k" in result for result in saved["repetition_results"])
            assert all("mrr_at_k" in result for result in saved["repetition_results"])
            assert saved == summary
            assert isinstance(saved["per_user_test_counts"], dict)
            assert sum(saved["per_user_test_counts"].values()) >= 1
        finally:
            if str(root) in sys.path:
                sys.path.remove(str(root))


def main() -> None:
    test_validation_pipeline_runs_and_writes_summary()
    print("validation pipeline tests passed")


if __name__ == "__main__":
    main()
