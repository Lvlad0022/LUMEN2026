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
    if sp.issparse(matrix):
        return matrix.tocsr()
    return sp.csr_matrix(matrix)


def _build_index_lookup(indices: Sequence[int] | np.ndarray) -> tuple[np.ndarray, dict[int, int]]:
    values = np.asarray(indices)
    lookup = {int(value): pos for pos, value in enumerate(values.tolist())}
    return values, lookup


@dataclass
class KatzConfig:
    beta: float = 0.001
    max_iter: int = 100
    tol: float = 1e-10
    include_self: bool = False


class katz_recommender:
    """Katz recommender for customer-item recommendation."""

    input_type = ARTIFACT_RELATION_MATRICES
    output_type = ARTIFACT_SIMILARITY_MATRIX
    input_artifacts = {
        "user_user_matrix": ArtifactSpec(
            name="user_user_matrix",
            kind=ARTIFACT_MATRIX,
            dense=False,
            description="Customer-customer relation matrix.",
        ),
        "user_item_matrix": ArtifactSpec(
            name="user_item_matrix",
            kind=ARTIFACT_MATRIX,
            dense=False,
            description="Customer-item relation matrix.",
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
    ) -> None:
        self.config = KatzConfig(
            beta=beta,
            max_iter=max_iter,
            tol=tol,
            include_self=include_self,
        )
        self.customer_idx_: np.ndarray | None = None
        self.item_idx_: np.ndarray | None = None
        self.customer_lookup_: dict[int, int] = {}
        self.item_lookup_: dict[int, int] = {}
        self.user_user_matrix_: sp.csr_matrix | None = None
        self.user_item_matrix_: sp.csr_matrix | None = None
        self.combined_matrix_: sp.csr_matrix | None = None
        self.katz_matrix_: sp.spmatrix | None = None
        self.item_scores_: sp.spmatrix | None = None
        self._seen_items_by_customer: dict[int, set[int]] = {}

    def fit(
        self,
        user_user_matrix: sp.spmatrix | np.ndarray,
        user_item_matrix: sp.spmatrix | np.ndarray,
        customer_idx: Sequence[int] | np.ndarray,
        item_idx: Sequence[int] | np.ndarray,
    ) -> "katz_recommender":
        """Fit the recommender and compute the Katz matrix."""

        user_user = _as_csr(user_user_matrix)
        user_item = _as_csr(user_item_matrix)
        customer_idx_arr, customer_lookup = _build_index_lookup(customer_idx)
        item_idx_arr, item_lookup = _build_index_lookup(item_idx)

        n_customers = user_user.shape[0]
        n_items = user_item.shape[1]

        if user_user.shape != (n_customers, n_customers):
            raise ValueError(
                f"user_user_matrix must be square with shape (n, n); got {user_user.shape}"
            )
        if user_item.shape[0] != n_customers:
            raise ValueError(
                "user_item_matrix must have the same number of rows as user_user_matrix."
            )
        if len(customer_idx_arr) != n_customers:
            raise ValueError(
                "customer_idx length must match the number of rows in user_user_matrix."
            )
        if len(item_idx_arr) != n_items:
            raise ValueError(
                "item_idx length must match the number of columns in user_item_matrix."
            )

        zero_items = sp.csr_matrix((n_items, n_items), dtype=user_user.dtype)
        combined = sp.bmat(
            [[user_user, user_item], [user_item.T, zero_items]],
            format="csr",
        )

        self.customer_idx_ = customer_idx_arr
        self.item_idx_ = item_idx_arr
        self.customer_lookup_ = customer_lookup
        self.item_lookup_ = item_lookup
        self.user_user_matrix_ = user_user
        self.user_item_matrix_ = user_item
        self.combined_matrix_ = combined

        self._seen_items_by_customer = {
            int(customer_id): set(user_item.getrow(pos).indices.tolist())
            for customer_id, pos in customer_lookup.items()
        }

        self.katz_matrix_ = self._compute_katz_matrix(combined)
        self.item_scores_ = self._extract_item_scores(self.katz_matrix_, n_customers, n_items)
        return self

    def export_artifacts(self) -> dict[str, object]:
        """Export the fitted similarity matrix and the fitted model."""

        if self.katz_matrix_ is None:
            raise RuntimeError("The recommender must be fitted before exporting artifacts.")
        return {
            "similarity_matrix": self.katz_matrix_,
            "model": self,
        }

    def predict(self, customer_idx: int, num_prediction: int = 5) -> list[int]:
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
        return katz_matrix[:n_customers, n_customers : n_customers + n_items].tocsr()


KatzRecommender = katz_recommender


__all__ = ["KatzRecommender", "KatzConfig", "katz_recommender"]
