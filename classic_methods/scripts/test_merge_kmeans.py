"""Regression tests for the merge-KMeans clustering wrapper."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = REPO_ROOT / "classic_methods" / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from clustering.methods import MergeKMeansClustering


def test_merge_kmeans_wrapper() -> None:
    """Verify the wrapper fits and exports cluster labels."""

    data_points = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [5.0, 5.0],
            [5.1, 4.9],
        ]
    )

    model = MergeKMeansClustering(k=2, min_size=1)
    fitted = model.fit(data_points)
    artifacts = fitted.export_artifacts()

    assert fitted is model
    assert "clusters" in artifacts
    assert "model" in artifacts
    assert len(artifacts["clusters"]) == 4
    assert artifacts["model"] is model


def main() -> None:
    """Execute the regression test."""

    test_merge_kmeans_wrapper()
    print("merge_kmeans tests passed")


if __name__ == "__main__":
    main()
