"""Katz-based recommender.

This implementation mirrors the notebook logic:
- build a combined block matrix from user-user and user-item relations
- compute Katz scores on the combined graph
- recommend only items for a given customer
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import scipy.sparse as sp

try:
    from ..pipeline_contracts import (
        ARTIFACT_INDEX_ARRAY,
        ARTIFACT_MATRIX,
        ARTIFACT_MODEL,
        ARTIFACT_RELATION_MATRICES,
        ARTIFACT_SIMILARITY_MATRIX,
        ArtifactSpec,
        StageContract,
    )
except ImportError:  # pragma: no cover - supports direct imports from src/
    from pipeline_contracts import (  # type: ignore
        ARTIFACT_INDEX_ARRAY,
        ARTIFACT_MATRIX,
        ARTIFACT_MODEL,
        ARTIFACT_RELATION_MATRICES,
        ARTIFACT_SIMILARITY_MATRIX,
        ArtifactSpec,
        StageContract,
    )


def _as_csr(matrix: sp.spmatrix | np.ndarray) -> sp.csr_matrix:
    # Inputs: sparse or dense matrix. Outputs: CSR sparse matrix.
    if sp.issparse(matrix):
        return matrix.tocsr()
    return sp.csr_matrix(matrix)


def _build_index_lookup(indices: Sequence[int] | np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    # Inputs: ordered id sequence. Outputs: NumPy array plus id-to-position lookup.
    values = np.asarray(indices)
    lookup = {int(value): pos for pos, value in enumerate(values.tolist())}
    return values, lookup


def _symmetric_graph_normalize(matrix: sp.csr_matrix) -> sp.csr_matrix:
    # Inputs: graph adjacency matrix. Outputs: symmetrically degree-normalized adjacency matrix.
    degrees = np.asarray(matrix.sum(axis=1)).ravel().astype(float)
    inv_sqrt = np.zeros_like(degrees)
    positive = degrees > 0.0
    inv_sqrt[positive] = 1.0 / np.sqrt(degrees[positive])
    scale = sp.diags(inv_sqrt, format="csr")
    return (scale @ matrix @ scale).tocsr()


@dataclass
class KatzConfig:
    beta: float = 0.001
    max_iter: int = 100
    tol: float = 1e-10
    include_self: bool = False
    symmetric_graph_normalization: bool = False


class katz_recommender:
    """Katz recommender for customer-item recommendation."""

    input_type = ARTIFACT_RELATION_MATRICES
    output_type = ARTIFACT_SIMILARITY_MATRIX
    input_artifacts = {
        "combined_matrix": ArtifactSpec(
            name="combined_matrix",
            kind=ARTIFACT_MATRIX,
            dense=False,
            description="Combined graph adjacency matrix over customers and items.",
        ),
        "customer_idx": ArtifactSpec(
            name="customer_idx",
            kind=ARTIFACT_INDEX_ARRAY,
            dense=True,
            description="Customer ordering used by the matrices.",
        ),
        "item_idx": ArtifactSpec(
            name="item_idx",
            kind=ARTIFACT_INDEX_ARRAY,
            dense=True,
            description="Item ordering used by the matrices.",
        ),
    }
    output_artifacts = {
        "similarity_matrix": ArtifactSpec(
            name="similarity_matrix",
            kind=ARTIFACT_SIMILARITY_MATRIX,
            dense=False,
            description="Katz similarity matrix over the combined graph.",
        ),
        "model": ArtifactSpec(
            name="model",
            kind=ARTIFACT_MODEL,
            dense=False,
            description="Fitted Katz recommender instance.",
        ),
    }
    contract = StageContract(
        input_type=input_type,
        output_type=output_type,
        input_artifacts=input_artifacts,
        output_artifacts=output_artifacts,
        dense=False,
        description="Katz similarity model for customer-item recommendation.",
    )

    def __init__(
        self,
        beta: float = 0.001,
        max_iter: int = 100,
        tol: float = 1e-10,
        include_self: bool = False,
        symmetric_graph_normalization: bool = False,
    ) -> None:
        self.config = KatzConfig(
            beta=beta,
            max_iter=max_iter,
            tol=tol,
            include_self=include_self,
            symmetric_graph_normalization=symmetric_graph_normalization,
        )
        self.customer_idx_: np.ndarray | None = None
        self.item_idx_: np.ndarray | None = None
        self.customer_lookup_: dict[int, int] = {}
        self.item_lookup_: dict[int, int] = {}
        self.combined_matrix_: sp.csr_matrix | None = None
        self.graph_matrix_: sp.csr_matrix | None = None
        self.katz_matrix_: sp.spmatrix | None = None
        self.item_scores_: sp.spmatrix | None = None
        self._seen_items_by_customer: dict[int, set[int]] = {}

    def fit(
        self,
        combined_matrix: sp.spmatrix | np.ndarray,
        customer_idx: Sequence[int] | np.ndarray,
        item_idx: Sequence[int] | np.ndarray,
    ) -> "katz_recommender":
        # Inputs: combined adjacency matrix plus customer/item orderings. Outputs: fitted Katz recommender.
        """Fit the recommender and compute the Katz matrix."""

        combined = _as_csr(combined_matrix)
        customer_idx_arr, customer_lookup = _build_index_lookup(customer_idx)
        item_idx_arr, item_lookup = _build_index_lookup(item_idx)

        n_customers = len(customer_idx_arr)
        n_items = len(item_idx_arr)
        minimum_nodes = n_customers + n_items
        if combined.shape[0] != combined.shape[1]:
            raise ValueError(
                f"combined_matrix must be square; got {combined.shape}"
            )
        if combined.shape[0] < minimum_nodes:
            raise ValueError(
                "combined_matrix must contain at least customer and item nodes with shape "
                f">= ({minimum_nodes}, {minimum_nodes}); got {combined.shape}"
            )

        self.customer_idx_ = customer_idx_arr
        self.item_idx_ = item_idx_arr
        self.customer_lookup_ = customer_lookup
        self.item_lookup_ = item_lookup
        self.combined_matrix_ = combined
        self.graph_matrix_ = (
            _symmetric_graph_normalize(combined)
            if self.config.symmetric_graph_normalization
            else combined.copy()
        )

        user_item = combined[:n_customers, n_customers : n_customers + n_items].tocsr()
        self._seen_items_by_customer = {
            int(customer_id): set(user_item.getrow(pos).indices.tolist())
            for customer_id, pos in customer_lookup.items()
        }

        self.katz_matrix_ = self._compute_katz_matrix(self.graph_matrix_)
        self.item_scores_ = self._extract_item_scores(self.katz_matrix_, n_customers, n_items)
        return self

    def export_artifacts(self) -> dict[str, object]:
        # Inputs: fitted recommender state. Outputs: exported Katz similarity matrix and model.
        """Export the fitted similarity matrix and the fitted model."""

        if self.katz_matrix_ is None:
            raise RuntimeError("The recommender must be fitted before exporting artifacts.")
        return {
            "similarity_matrix": self.katz_matrix_,
            "model": self,
        }

    def predict(self, customer_idx: int, num_prediction: int = 5) -> list[int]:
        # Inputs: customer id and desired recommendation count. Outputs: ranked unseen item ids.
        """Return the top item ids for a given customer id."""

        if self.item_scores_ is None or self.customer_lookup_ is None or self.item_idx_ is None:
            raise RuntimeError("The recommender must be fitted before calling predict().")

        if customer_idx not in self.customer_lookup_:
            raise KeyError(f"Unknown customer_idx: {customer_idx}")

        customer_pos = self.customer_lookup_[int(customer_idx)]
        scores = np.asarray(self.item_scores_.getrow(customer_pos).toarray()).ravel().copy()

        seen_items = self._seen_items_by_customer.get(int(customer_idx), set())
        if seen_items:
            scores[list(seen_items)] = -np.inf

        top_k = min(int(num_prediction), scores.shape[0])
        if top_k <= 0:
            return []

        candidate_positions = np.argpartition(-scores, kth=top_k - 1)[:top_k]
        candidate_positions = candidate_positions[np.argsort(-scores[candidate_positions])]

        return [int(self.item_idx_[pos]) for pos in candidate_positions if np.isfinite(scores[pos])]

    def _compute_katz_matrix(self, matrix: sp.csr_matrix) -> sp.spmatrix:
        # Inputs: graph adjacency matrix. Outputs: truncated Katz similarity matrix over all graph nodes.
        beta = self.config.beta
        current = matrix.copy().astype(float)
        katz = current.copy()
        factor = 1.0

        for _ in range(self.config.max_iter):
            if factor < self.config.tol:
                break
            current = matrix @ current
            factor *= beta
            katz = katz + factor * current

        if self.config.include_self:
            katz = katz + sp.eye(matrix.shape[0], format="csr")

        return katz.tocsr()

    def _extract_item_scores(self, katz_matrix: sp.spmatrix, n_customers: int, n_items: int) -> sp.csr_matrix:
        # Inputs: full Katz matrix plus customer/item counts. Outputs: customer-item score block.
        return katz_matrix[:n_customers, n_customers : n_customers + n_items].tocsr()


KatzRecommender = katz_recommender


__all__ = ["KatzRecommender", "KatzConfig", "katz_recommender"]
