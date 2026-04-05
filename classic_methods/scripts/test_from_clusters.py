"""Regression tests for the cluster-to-similarity constructor."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "classic_methods" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.similarity_matrix.from_clusters import clusters_to_similarity


def _sample_dataframe() -> pd.DataFrame:
    """Build a small dataframe with repeated customer-item purchases."""

    return pd.DataFrame(
        {
            "CustomerID": [1, 1, 2, 3, 3],
            "Item Code": ["A", "B", "A", "C", "A"],
            "item_idx": [0, 1, 0, 2, 0],
        }
    )


def test_sequence_clusters() -> None:
    """Verify sequence-based cluster labels build the expected matrices."""

    builder = clusters_to_similarity()
    builder.fit(_sample_dataframe(), [7, 7, 9])

    user_user = builder.user_user_matrix_.toarray()
    user_item = builder.user_item_matrix_.toarray()

    assert user_user.shape == (3, 3)
    assert user_item.shape == (3, 3)
    assert user_user.tolist() == [
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]
    assert user_item.tolist() == [
        [1.0, 1.0, 0.0],
        [1.0, 0.0, 0.0],
        [1.0, 0.0, 1.0],
    ]
    assert builder.customer_idx_.tolist() == [1, 2, 3]
    assert builder.item_idx_.tolist() == [0, 1, 2]


def test_mapping_clusters() -> None:
    """Verify mapping-based cluster labels align by customer id."""

    builder = clusters_to_similarity()
    builder.fit(_sample_dataframe(), {1: 3, 2: 3, 3: 8})

    user_user = builder.user_user_matrix_.toarray()
    assert user_user.tolist() == [
        [1.0, 1.0, 0.0],
        [1.0, 1.0, 0.0],
        [0.0, 0.0, 1.0],
    ]


def test_missing_cluster_raises() -> None:
    """Verify incomplete mappings fail fast."""

    builder = clusters_to_similarity()
    try:
        builder.fit(_sample_dataframe(), {1: 0, 2: 0})
    except KeyError as exc:
        assert "Missing cluster assignments" in str(exc)
    else:
        raise AssertionError("Expected a KeyError for missing customer clusters.")


def main() -> None:
    """Execute the cluster similarity regression tests."""

    test_sequence_clusters()
    test_mapping_clusters()
    test_missing_cluster_raises()
    print("from_clusters tests passed")


if __name__ == "__main__":
    main()
