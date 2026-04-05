"""Regression tests for pipeline runtime options."""

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

from pipeline import Pipeline


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(dedent(text).lstrip(), encoding="utf-8")


def _build_project(root: Path) -> Path:
    project_dir = root / "classic_methods"
    data_dir = project_dir / "data"
    config_dir = project_dir / "config"
    data_dir.mkdir(parents=True, exist_ok=True)
    config_dir.mkdir(parents=True, exist_ok=True)

    helper_module = root / "test_runtime_support.py"
    _write_text(
        helper_module,
        """
        from __future__ import annotations

        import numpy as np


        class CustomerIndexStage:
            def fit(self, df):
                customer_idx = np.array(sorted(set(int(value) for value in df["CustomerID"].tolist())), dtype=int)
                return {
                    "dataframe": df.copy(),
                    "customer_idx": customer_idx,
                }


        class DummyRecommendationModel:
            def predict(self, customer_idx: int, num_prediction: int = 5):
                start = int(customer_idx) * 100
                return list(range(start, start + int(num_prediction)))


        class ModelStage:
            def fit(self, customer_idx):
                return {
                    "model": DummyRecommendationModel(),
                    "customer_idx": customer_idx,
                }
        """,
    )

    df = pd.DataFrame(
        {
            "CustomerID": [1, 1, 2, 3],
            "item_idx": [10, 20, 30, 40],
            "Order Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
        }
    )
    df.to_csv(data_dir / "database.csv", index=False)

    _write_text(
        config_dir / "pipeline.yaml",
        """
        data:
          csv_path: data/database.csv

        save_intermediates: true

        stage1:
          config_path: config/stage1.yaml

        stage2:
          config_path: config/stage2.yaml
        """,
    )
    _write_text(
        config_dir / "stage1.yaml",
        """
        _target_: test_runtime_support.CustomerIndexStage
        method: fit
        params: {}
        inputs:
          df: dataframe
        """,
    )
    _write_text(
        config_dir / "stage2.yaml",
        """
        _target_: test_runtime_support.ModelStage
        method: fit
        params: {}
        inputs:
          customer_idx: customer_idx
        """,
    )

    return config_dir / "pipeline.yaml"


def test_pipeline_runtime_options() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir)
        sys.path.insert(0, str(root))
        try:
            config_path = _build_project(root)

            pipeline = Pipeline(config_path)
            assert pipeline.output_dir.name == "run1"
            assert not pipeline.output_dir.exists()

            artifacts = pipeline.run(save_artifacts=False, return_recommendations=True, recommendation_k=3)
            recommendations = artifacts["recommendations"].value
            assert recommendations == {
                1: [100, 101, 102],
                2: [200, 201, 202],
                3: [300, 301, 302],
            }
            assert not pipeline.output_dir.exists()

            saved_pipeline = Pipeline(config_path)
            saved_artifacts = saved_pipeline.run(
                save_artifacts=True,
                return_recommendations=True,
                recommendation_k=2,
            )
            assert saved_pipeline.output_dir.name == "run1"
            assert saved_pipeline.output_dir.exists()
            assert (saved_pipeline.output_dir / "manifest.json").exists()
            assert (saved_pipeline.output_dir / "recommendations.json").exists()
            manifest = json.loads((saved_pipeline.output_dir / "manifest.json").read_text(encoding="utf-8"))
            assert "recommendations" in manifest
            assert saved_artifacts["recommendations"].value == {
                1: [100, 101],
                2: [200, 201],
                3: [300, 301],
            }

            next_pipeline = Pipeline(config_path)
            assert next_pipeline.output_dir.name == "run2"
        finally:
            if str(root) in sys.path:
                sys.path.remove(str(root))


def main() -> None:
    test_pipeline_runtime_options()
    print("pipeline runtime option tests passed")


if __name__ == "__main__":
    main()
