"""Construct similarity matrices from cluster assignments and transaction data."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp

try:
    from ...pipeline_contracts import (
        ARTIFACT_CLUSTER_LABELS,
        ARTIFACT_DATAFRAME,
        ARTIFACT_INDEX_ARRAY,
        ARTIFACT_MATRIX,
        ARTIFACT_RELATION_MATRICES,
        ArtifactSpec,
        StageContract,
    )
except ImportError:  # pragma: no cover - supports direct imports from src/
    from pipeline_contracts import (  # type: ignore
        ARTIFACT_CLUSTER_LABELS,
        ARTIFACT_DATAFRAME,
        ARTIFACT_INDEX_ARRAY,
        ARTIFACT_MATRIX,
        ARTIFACT_RELATION_MATRICES,
        ArtifactSpec,
        StageContract,
    )


def _as_array(values: Sequence[Any] | np.ndarray | pd.Series) -> np.ndarray:
    """Convert a sequence-like object to a NumPy array."""

    if isinstance(values, pd.Series):
        return values.to_numpy()
    return np.asarray(values)


def _to_csr(matrix: sp.spmatrix | np.ndarray) -> sp.csr_matrix:
    """Convert a sparse or dense matrix into CSR format."""

    if sp.issparse(matrix):
        return matrix.tocsr()
    return sp.csr_matrix(matrix)


@dataclass
class ClusterSimilarityConfig:
    """Configuration for cluster-driven similarity construction."""

    customer_col: str = "CustomerID"
    item_col: str = "Item Code"
    item_idx_col: str = "item_idx"
    drop_missing_customer_id: bool = True
    drop_missing_item_code: bool = True


class clusters_to_similarity:
    """Build customer-customer and customer-item similarity matrices from clusters."""

    input_type = None
    output_type = ARTIFACT_RELATION_MATRICES
    input_artifacts = {
        "dataframe": ArtifactSpec(
            name="dataframe",
            kind=ARTIFACT_DATAFRAME,
            dense=True,
            description="Transaction dataframe with customer and item columns.",
        ),
        "clusters": ArtifactSpec(
            name="clusters",
            kind=ARTIFACT_CLUSTER_LABELS,
            dense=True,
            description="Cluster labels aligned with customers or keyed by customer id.",
        ),
    }
    output_artifacts = {
        "user_user_matrix": ArtifactSpec(
            name="user_user_matrix",
            kind=ARTIFACT_MATRIX,
            dense=False,
            description="Customer-customer similarity matrix derived from clusters.",
        ),
        "user_item_matrix": ArtifactSpec(
            name="user_item_matrix",
            kind=ARTIFACT_MATRIX,
            dense=False,
            description="Binary customer-item matrix from purchase history.",
        ),
        "customer_idx": ArtifactSpec(
            name="customer_idx",
            kind=ARTIFACT_INDEX_ARRAY,
            dense=True,
            description="Customer ids in the same order as the similarity matrix rows.",
        ),
        "item_idx": ArtifactSpec(
            name="item_idx",
            kind=ARTIFACT_INDEX_ARRAY,
            dense=True,
            description="Item ids in the same order as the similarity matrix columns.",
        ),
    }
    contract = StageContract(
        input_type=input_type,
        output_type=output_type,
        input_artifacts=input_artifacts,
        output_artifacts=output_artifacts,
        dense=False,
        description="Convert customer clusters into similarity matrices.",
    )

    def __init__(
        self,
        customer_col: str = "CustomerID",
        item_col: str = "Item Code",
        item_idx_col: str = "item_idx",
        drop_missing_customer_id: bool = True,
        drop_missing_item_code: bool = True,
    ) -> None:
        self.config = ClusterSimilarityConfig(
            customer_col=customer_col,
            item_col=item_col,
            item_idx_col=item_idx_col,
            drop_missing_customer_id=drop_missing_customer_id,
            drop_missing_item_code=drop_missing_item_code,
        )
        self.customer_idx_: np.ndarray | None = None
        self.item_idx_: np.ndarray | None = None
        self.customer_lookup_: dict[Any, int] = {}
        self.item_lookup_: dict[Any, int] = {}
        self.cluster_labels_: np.ndarray | None = None
        self.user_user_matrix_: sp.csr_matrix | None = None
        self.user_item_matrix_: sp.csr_matrix | None = None

    def fit(
        self,
        dataframe: pd.DataFrame,
        clusters: Sequence[Any] | Mapping[Any, Any] | pd.Series,
    ) -> "clusters_to_similarity":
        """Fit the constructor and build both similarity matrices."""

        if self.config.customer_col not in dataframe.columns:
            raise KeyError(f"Missing required customer column: {self.config.customer_col}")
        if self.config.item_col not in dataframe.columns and self.config.item_idx_col not in dataframe.columns:
            raise KeyError(
                f"Missing required item column: {self.config.item_col} or {self.config.item_idx_col}"
            )

        work = dataframe.copy()
        if self.config.drop_missing_customer_id:
            work = work[~work[self.config.customer_col].isna()].copy()
        if self.config.drop_missing_item_code:
            item_source = self.config.item_idx_col if self.config.item_idx_col in work.columns else self.config.item_col
            work = work[~work[item_source].isna()].copy()

        customer_idx = pd.unique(work[self.config.customer_col].dropna())
        customer_lookup = {customer: pos for pos, customer in enumerate(customer_idx.tolist())}
        cluster_labels = self._align_clusters(customer_idx, clusters)

        if self.config.item_idx_col in work.columns:
            item_idx = pd.unique(work[self.config.item_idx_col].dropna())
            item_lookup = {item: pos for pos, item in enumerate(item_idx.tolist())}
            item_positions = work[self.config.item_idx_col].map(item_lookup).to_numpy()
        else:
            item_idx = pd.unique(work[self.config.item_col].dropna())
            item_lookup = {item: pos for pos, item in enumerate(item_idx.tolist())}
            item_positions = work[self.config.item_col].map(item_lookup).to_numpy()

        customer_positions = work[self.config.customer_col].map(customer_lookup).to_numpy()

        if np.any(pd.isna(customer_positions)):
            raise ValueError("Some customer ids in the dataframe were not aligned to customer_idx.")
        if np.any(pd.isna(item_positions)):
            raise ValueError("Some item ids in the dataframe were not aligned to item_idx.")

        customer_positions = customer_positions.astype(int)
        item_positions = item_positions.astype(int)

        self.customer_idx_ = _as_array(customer_idx)
        self.item_idx_ = _as_array(item_idx)
        self.customer_lookup_ = customer_lookup
        self.item_lookup_ = item_lookup
        self.cluster_labels_ = _as_array(cluster_labels)

        self.user_user_matrix_ = self._build_customer_similarity(self.cluster_labels_)
        self.user_item_matrix_ = self._build_customer_item_matrix(
            customer_positions=customer_positions,
            item_positions=item_positions,
            n_customers=len(customer_idx),
            n_items=len(item_idx),
        )
        return self

    def export_artifacts(self) -> dict[str, object]:
        """Export the matrices and ordering arrays for downstream stages."""

        if (
            self.user_user_matrix_ is None
            or self.user_item_matrix_ is None
            or self.customer_idx_ is None
            or self.item_idx_ is None
        ):
            raise RuntimeError("The stage must be fitted before exporting artifacts.")
        return {
            "user_user_matrix": self.user_user_matrix_,
            "user_item_matrix": self.user_item_matrix_,
            "customer_idx": self.customer_idx_,
            "item_idx": self.item_idx_,
        }

    def _align_clusters(
        self,
        customer_idx: np.ndarray,
        clusters: Sequence[Any] | Mapping[Any, Any] | pd.Series,
    ) -> np.ndarray:
        """Align cluster labels with the customer ordering used by the dataframe."""

        if isinstance(clusters, pd.Series):
            lookup = clusters.to_dict()
            missing = [customer for customer in customer_idx.tolist() if customer not in lookup]
            if missing:
                raise KeyError(f"Missing cluster assignments for customers: {missing[:5]}")
            return np.asarray([lookup[customer] for customer in customer_idx.tolist()])

        if isinstance(clusters, Mapping):
            missing = [customer for customer in customer_idx.tolist() if customer not in clusters]
            if missing:
                raise KeyError(f"Missing cluster assignments for customers: {missing[:5]}")
            return np.asarray([clusters[customer] for customer in customer_idx.tolist()])

        cluster_array = np.asarray(clusters)
        if cluster_array.ndim != 1:
            raise ValueError("clusters must be a 1D sequence, Series, or mapping.")
        if cluster_array.shape[0] != customer_idx.shape[0]:
            raise ValueError(
                "clusters length must match the number of unique customers in the dataframe."
            )
        return cluster_array

    def _build_customer_similarity(self, cluster_labels: np.ndarray) -> sp.csr_matrix:
        """Build a binary customer-customer similarity matrix from cluster labels."""

        _, inverse = np.unique(cluster_labels, return_inverse=True)
        n_customers = cluster_labels.shape[0]
        membership = sp.csr_matrix(
            (
                np.ones(n_customers, dtype=np.float32),
                (np.arange(n_customers), inverse),
            ),
            shape=(n_customers, int(inverse.max()) + 1 if inverse.size else 0),
        )
        similarity = membership @ membership.T
        similarity.data[:] = 1.0
        return similarity.tocsr()

    def _build_customer_item_matrix(
        self,
        *,
        customer_positions: np.ndarray,
        item_positions: np.ndarray,
        n_customers: int,
        n_items: int,
    ) -> sp.csr_matrix:
        """Build a binary customer-item matrix from observed purchases."""

        pairs = np.column_stack([customer_positions, item_positions])
        if pairs.size == 0:
            return sp.csr_matrix((n_customers, n_items), dtype=np.float32)

        unique_pairs = np.unique(pairs, axis=0)
        data = np.ones(unique_pairs.shape[0], dtype=np.float32)
        return sp.csr_matrix(
            (data, (unique_pairs[:, 0], unique_pairs[:, 1])),
            shape=(n_customers, n_items),
        )


ClustersToSimilarity = clusters_to_similarity


__all__ = ["ClusterSimilarityConfig", "ClustersToSimilarity", "clusters_to_similarity"]
