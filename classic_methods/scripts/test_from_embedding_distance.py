"""Regression tests for embedding-distance graph construction."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "classic_methods" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from models.similarity_matrix.from_embedding_distance import embedding_distance_to_similarity


def _sample_dataframe() -> pd.DataFrame:
    # Inputs: none. Outputs: toy customer-item dataframe with stable item_idx values.
    return pd.DataFrame(
        {
            "CustomerID": [1, 1, 2, 3, 3],
            "Item Code": ["A", "B", "A", "C", "A"],
            "item_idx": [0, 1, 0, 2, 0],
        }
    )


def test_embedding_similarity_combined_matrix() -> None:
    # Inputs: toy dataframe and embedding. Outputs: aligned combined graph with expected customer/item blocks.
    builder = embedding_distance_to_similarity(metric="euclidean", similarity_kernel="inverse", n_neighbors=1)
    embedding = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
        ],
        dtype=float,
    )
    builder.fit(_sample_dataframe(), embedding)

    user_user = builder.user_user_matrix_.toarray()
    user_item = builder.user_item_matrix_.toarray()
    combined = builder.combined_matrix_.toarray()

    assert user_user.shape == (3, 3)
    assert user_item.shape == (3, 3)
    assert combined.shape == (6, 6)
    assert builder.customer_idx_.tolist() == [1, 2, 3]
    assert builder.item_idx_.tolist() == [0, 1, 2]
    assert combined[:3, :3].tolist() == user_user.tolist()
    assert combined[:3, 3:].tolist() == user_item.tolist()


def test_embedding_similarity_weights_scale_combined_blocks() -> None:
    # Inputs: toy dataframe, embedding, and relation weights. Outputs: weighted combined graph blocks.
    builder = embedding_distance_to_similarity(
        metric="euclidean",
        similarity_kernel="inverse",
        n_neighbors=1,
        customer_customer_weight=2.0,
        customer_item_weight=3.0,
    )
    embedding = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
        ],
        dtype=float,
    )
    builder.fit(_sample_dataframe(), embedding)

    combined = builder.combined_matrix_.toarray()
    assert combined[0, 1] == 2.0 * builder.user_user_matrix_.toarray()[0, 1]
    assert combined[0, 3] == 3.0
    assert combined[0, 4] == 3.0


def test_embedding_similarity_normalizes_customer_blocks_separately() -> None:
    # Inputs: toy dataframe, embedding, and enabled block normalization. Outputs: customer row sums equal the configured block weights.
    builder = embedding_distance_to_similarity(
        metric="euclidean",
        similarity_kernel="inverse",
        n_neighbors=2,
        customer_customer_weight=2.0,
        customer_item_weight=3.0,
        normalize_customer_customer=True,
        normalize_customer_item=True,
    )
    embedding = np.asarray(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
        ],
        dtype=float,
    )
    builder.fit(_sample_dataframe(), embedding)

    combined = builder.combined_matrix_.toarray()
    customer_customer_block = combined[:3, :3]
    customer_item_block = combined[:3, 3:]

    np.testing.assert_allclose(customer_customer_block.sum(axis=1), np.full(3, 2.0), atol=1e-6)
    np.testing.assert_allclose(customer_item_block.sum(axis=1), np.full(3, 3.0), atol=1e-6)


def main() -> None:
    # Inputs: none. Outputs: executed regression checks.
    test_embedding_similarity_combined_matrix()
    test_embedding_similarity_weights_scale_combined_blocks()
    test_embedding_similarity_normalizes_customer_blocks_separately()
    print("from_embedding_distance tests passed")


if __name__ == "__main__":
    main()
