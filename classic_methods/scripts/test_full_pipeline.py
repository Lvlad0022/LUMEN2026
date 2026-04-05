"""End-to-end regression test for preprocessing -> embedding -> clustering -> similarity -> Katz."""

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
    """Write a config file with parent directories created on demand."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.strip() + "\n", encoding="utf-8")


def _build_project(root: Path) -> Path:
    """Create a minimal project tree with a CSV database and stage configs."""

    project_dir = root / "classic_methods"
    data_dir = project_dir / "data" / "raw"
    config_dir = project_dir / "config"
    data_processing_dir = config_dir / "data_processing"
    embeddings_dir = config_dir / "embeddings"
    clustering_dir = config_dir / "clustering"
    models_similarity_dir = config_dir / "models" / "similarity_matrix"
    data_dir.mkdir(parents=True, exist_ok=True)
    data_processing_dir.mkdir(parents=True, exist_ok=True)
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    clustering_dir.mkdir(parents=True, exist_ok=True)
    models_similarity_dir.mkdir(parents=True, exist_ok=True)

    df = pd.DataFrame(
        {
            "CustomerID": [1, 1, 2, 2, 3, 3],
            "Item Code": ["A", "B", "A", "C", "B", "D"],
            "Order Date": [
                "2024-01-01",
                "2024-01-02",
                "2024-01-03",
                "2024-01-04",
                "2024-01-05",
                "2024-01-06",
            ],
            "Invoiced price": [10.0, 11.0, 12.0, 13.0, 14.0, 15.0],
            "Invoiced price (TX)": [11.0, 12.0, 13.0, 14.0, 15.0, 16.0],
            "Ordered qty": [1, 2, 1, 2, 1, 1],
            "Invoiced qty (shipped)": [1, 2, 1, 2, 1, 1],
            "GM%": [0.1, 0.2, 0.15, 0.18, 0.22, 0.25],
        }
    )
    df.to_csv(data_dir / "database.csv", index=False)

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

        stage3:
          config_path: config/clustering/adjusted_kmeans.yaml

        stage4:
          config_path: config/models/similarity_matrix/from_clusters.yaml

        stage5:
          config_path: config/models/similarity_matrix/katz.yaml
        """,
    )

    _write_text(
        data_processing_dir / "preprocessing.yaml",
        """
        _target_: data_processing.preprocessing.Preprocessor
        method: fit_transform
        params:
          drop_price_gt_tx: false
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

    _write_text(
        clustering_dir / "adjusted_kmeans.yaml",
        """
        _target_: clustering.methods.AdjustedKMeansClustering
        method: fit
        params:
          k: 2
          min_size: 1
        inputs:
          data_points: embedding
        """,
    )

    _write_text(
        models_similarity_dir / "from_clusters.yaml",
        """
        _target_: models.similarity_matrix.from_clusters.clusters_to_similarity
        method: fit
        params:
          customer_col: CustomerID
          item_col: Item Code
          item_idx_col: item_idx
          drop_missing_customer_id: true
          drop_missing_item_code: true
        inputs:
          dataframe: dataframe
          clusters: clusters
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

    return config_dir / "pipeline.yaml"


def test_full_pipeline() -> None:
    """Verify the full numbered-stage chain executes and stores artifacts."""

    with tempfile.TemporaryDirectory() as tmpdir:
        config_path = _build_project(Path(tmpdir))
        pipeline = Pipeline(config_path)
        artifacts = pipeline.run()
        output_dir = pipeline.output_dir

        assert output_dir.name == "run1"
        assert output_dir.parent.name == "output"
        assert "dataframe" in artifacts
        assert "embedding" in artifacts
        assert "clusters" in artifacts
        assert "user_user_matrix" in artifacts
        assert "user_item_matrix" in artifacts
        assert "similarity_matrix" in artifacts
        assert "model" in artifacts
        assert "stage5.similarity_matrix" in artifacts
        assert "stage5.model" in artifacts
        assert (output_dir / "manifest.json").exists()
        assert (output_dir / "dataframe.csv").exists()
        assert (output_dir / "embedding.npy").exists()
        assert (output_dir / "clusters.npy").exists()
        assert (output_dir / "user_user_matrix.npz").exists()
        assert (output_dir / "user_item_matrix.npz").exists()
        assert (output_dir / "similarity_matrix.npz").exists()
        assert (output_dir / "model.pkl").exists()

        second_pipeline = Pipeline(config_path)
        second_output_dir = second_pipeline.output_dir
        assert second_output_dir.name == "run2"
        assert second_output_dir.parent == output_dir.parent


def main() -> None:
    """Execute the end-to-end regression test."""

    test_full_pipeline()
    print("full pipeline test passed")


if __name__ == "__main__":
    main()
