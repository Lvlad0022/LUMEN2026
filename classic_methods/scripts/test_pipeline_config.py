"""Regression tests for the numbered-stage config pipeline."""

from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "classic_methods" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from pipeline import Pipeline


def _write_text(path: Path, text: str) -> None:
    """Write text to a file, creating parent directories as needed."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _build_project(root: Path) -> Path:
    """Create a temporary mini-project with config files and a CSV database."""

    project_dir = root / "classic_methods"
    data_dir = project_dir / "data" / "raw"
    config_dir = project_dir / "config"
    data_processing_dir = config_dir / "data_processing"
    embeddings_dir = config_dir / "embeddings"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_processing_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(
        {
            "CustomerID": [1, 1, 2, 2],
            "Item Code": ["A", "B", "A", "C"],
            "Order Date": ["2024-01-01", "2024-01-02", "2024-01-03", "2024-01-04"],
            "Invoiced price": [10.0, 12.0, 9.0, 11.0],
            "Invoiced price (TX)": [11.0, 13.0, 10.0, 12.0],
        }
    ).to_csv(data_dir / "database.csv", index=False)

    _write_text(
        config_dir / "pipeline.yaml",
        """
        data:
          csv_path: data/raw/database.csv

        save_intermediates: true

        stage1:
          config_path: config/data_processing/preprocessing.yaml

        stage2:
          config_path: config/embeddings/function1.yaml
        """,
    )

    _write_text(
        data_processing_dir / "preprocessing.yaml",
        """
        _target_: data_processing.preprocessing.Preprocessor
        method: fit_transform
        params:
          drop_price_gt_tx: true
          drop_missing_customer_id: true
          drop_missing_item_code: true
          reindex_items_at_end: true
          min_num_purchases_per_customer: null
          min_num_items_per_customer: null
          min_num_purchases_per_item_rows: null
          min_num_unique_customers_per_item: null
          null_rules: []
        inputs:
          df: dataframe
        """,
    )

    _write_text(
        embeddings_dir / "function1.yaml",
        """
        _target_: embeddings.simple_embeddings.Function1Embedding
        method: fit
        params: {}
        inputs:
          df: dataframe
        """,
    )

    return config_dir / "pipeline.yaml"


def test_sequential_pipeline() -> None:
    """Verify that numbered stages run in order and preserve intermediates."""

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = _build_project(Path(tmpdir))
        pipeline = Pipeline(config_path)
        artifacts = pipeline.run()

        assert len(pipeline.stage_configs) == 2
        assert "dataframe" in artifacts
        assert "idx2item" in artifacts
        assert "stage1.dataframe" in artifacts
        assert "stage2.embedding" in artifacts
        assert isinstance(artifacts["dataframe"].value, pd.DataFrame)
        assert "item_idx" in artifacts["dataframe"].value.columns
        assert isinstance(artifacts["idx2item"].value, dict)


def test_gap_detection() -> None:
    """Verify that missing stage numbers fail fast."""

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "classic_methods"
        data_dir = root / "data" / "raw"
        config_dir = root / "config"
        data_processing_dir = config_dir / "data_processing"
        embeddings_dir = config_dir / "embeddings"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_processing_dir.mkdir(parents=True, exist_ok=True)
        embeddings_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            {
                "CustomerID": [1],
                "Item Code": ["A"],
                "Order Date": ["2024-01-01"],
                "Invoiced price": [10.0],
                "Invoiced price (TX)": [11.0],
            }
        ).to_csv(data_dir / "database.csv", index=False)

        _write_text(
            config_dir / "pipeline.yaml",
            """
            data:
              csv_path: data/raw/database.csv

            stage1:
              config_path: config/data_processing/preprocessing.yaml

            stage3:
              config_path: config/embeddings/function1.yaml
            """,
        )
        _write_text(
            data_processing_dir / "preprocessing.yaml",
            """
            _target_: data_processing.preprocessing.Preprocessor
            method: fit_transform
            params:
              drop_price_gt_tx: true
              drop_missing_customer_id: true
              drop_missing_item_code: true
              reindex_items_at_end: true
              min_num_purchases_per_customer: null
              min_num_items_per_customer: null
              min_num_purchases_per_item_rows: null
              min_num_unique_customers_per_item: null
              null_rules: []
            inputs:
              df: dataframe
            """,
        )
        _write_text(
            embeddings_dir / "function1.yaml",
            """
            _target_: embeddings.simple_embeddings.Function1Embedding
            method: fit
            params: {}
            inputs:
              df: dataframe
            """,
        )

        try:
            Pipeline(config_dir / "pipeline.yaml")
        except ValueError as exc:
            assert "stage2" in str(exc)
        else:
            raise AssertionError("Expected a gap detection error for missing stage2.")


def test_katz_incompatibility() -> None:
    """Verify that a dataframe stage cannot feed a similarity-matrix model."""

    with tempfile.TemporaryDirectory() as tmpdir:
        root = Path(tmpdir) / "classic_methods"
        data_dir = root / "data" / "raw"
        config_dir = root / "config"
        data_processing_dir = config_dir / "data_processing"
        models_similarity_dir = config_dir / "models" / "similarity_matrix"
        data_dir.mkdir(parents=True, exist_ok=True)
        data_processing_dir.mkdir(parents=True, exist_ok=True)
        models_similarity_dir.mkdir(parents=True, exist_ok=True)

        pd.DataFrame(
            {
                "CustomerID": [1, 1],
                "Item Code": ["A", "B"],
                "Order Date": ["2024-01-01", "2024-01-02"],
                "Invoiced price": [10.0, 12.0],
                "Invoiced price (TX)": [11.0, 13.0],
            }
        ).to_csv(data_dir / "database.csv", index=False)

        _write_text(
            config_dir / "pipeline.yaml",
            """
            data:
              csv_path: data/raw/database.csv

            stage1:
              config_path: config/data_processing/preprocessing.yaml

            stage2:
              config_path: config/models/similarity_matrix/katz.yaml
            """,
        )
        _write_text(
            data_processing_dir / "preprocessing.yaml",
            """
            _target_: data_processing.preprocessing.Preprocessor
            method: fit_transform
            params:
              drop_price_gt_tx: true
              drop_missing_customer_id: true
              drop_missing_item_code: true
              reindex_items_at_end: true
              min_num_purchases_per_customer: null
              min_num_items_per_customer: null
              min_num_purchases_per_item_rows: null
              min_num_unique_customers_per_item: null
              null_rules: []
            inputs:
              df: dataframe
            """,
        )
        _write_text(
            models_similarity_dir / "katz.yaml",
            """
            _target_: models.similarity_matrix.katz.KatzRecommender
            method: fit
            params:
              beta: 0.001
              max_iter: 10
              tol: 1.0e-10
              include_self: false
            inputs:
              user_user_matrix: user_user_matrix
              user_item_matrix: user_item_matrix
              customer_idx: customer_idx
              item_idx: item_idx
            """,
        )

        try:
            Pipeline(config_dir / "pipeline.yaml")
        except ValueError as exc:
            message = str(exc)
            assert "expects kind 'matrix'" in message or "expects input type" in message
        else:
            raise AssertionError("Expected Katz compatibility validation to fail.")


def main() -> None:
    """Execute all pipeline regression tests."""

    test_sequential_pipeline()
    test_gap_detection()
    test_katz_incompatibility()
    print("pipeline config tests passed")


if __name__ == "__main__":
    main()
