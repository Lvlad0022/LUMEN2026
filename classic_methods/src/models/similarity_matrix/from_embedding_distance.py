"""Construct customer-customer similarity matrices from embedding distances."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.neighbors import NearestNeighbors

try:
    from ...pipeline_contracts import (
        ARTIFACT_DATAFRAME,
        ARTIFACT_INDEX_ARRAY,
        ARTIFACT_MATRIX,
        ARTIFACT_RELATION_MATRICES,
        ArtifactSpec,
        StageContract,
    )
except ImportError:  # pragma: no cover - supports direct imports from src/
    from pipeline_contracts import (  # type: ignore
        ARTIFACT_DATAFRAME,
        ARTIFACT_INDEX_ARRAY,
        ARTIFACT_MATRIX,
        ARTIFACT_RELATION_MATRICES,
        ArtifactSpec,
        StageContract,
    )


def _as_array(values: Sequence[Any] | np.ndarray | pd.Series) -> np.ndarray:
    # Inputs: sequence-like values. Outputs: NumPy array view/copy.
    """Convert a sequence-like object to a NumPy array."""

    if isinstance(values, pd.Series):
        return values.to_numpy()
    return np.asarray(values)


def _resolve_customer_order(
    dataframe: pd.DataFrame,
    *,
    customer_col: str,
    customer_idx: Sequence[Any] | np.ndarray | None,
) -> np.ndarray:
    # Inputs: dataframe, customer column name, and optional explicit order. Outputs: resolved customer ordering.
    """Resolve customer ordering from an explicit array or from first appearance in the dataframe."""

    if customer_idx is not None:
        resolved = _as_array(customer_idx)
        if resolved.ndim != 1:
            raise ValueError("customer_idx must be a 1D sequence when provided.")
        return resolved
    return pd.unique(dataframe[customer_col].dropna())


def _build_customer_item_matrix(
    dataframe: pd.DataFrame,
    *,
    customer_col: str,
    item_col: str,
    item_idx_col: str,
    customer_idx: np.ndarray,
) -> tuple[sp.csr_matrix, np.ndarray]:
    # Inputs: transaction dataframe plus column names and customer order. Outputs: aligned customer-item matrix and item ids.
    """Build a binary customer-item matrix aligned to the provided customer ordering."""

    customer_lookup = {customer: pos for pos, customer in enumerate(customer_idx.tolist())}
    if item_idx_col in dataframe.columns:
        item_values = pd.unique(dataframe[item_idx_col].dropna())
        item_positions = dataframe[item_idx_col]
    else:
        item_values = pd.unique(dataframe[item_col].dropna())
        item_positions = dataframe[item_col]

    item_lookup = {item: pos for pos, item in enumerate(item_values.tolist())}
    customer_positions = dataframe[customer_col].map(customer_lookup).to_numpy()
    item_positions_array = item_positions.map(item_lookup).to_numpy()

    if np.any(pd.isna(customer_positions)):
        raise ValueError("Some dataframe customers were not aligned to the resolved customer_idx order.")
    if np.any(pd.isna(item_positions_array)):
        raise ValueError("Some dataframe items were not aligned to the resolved item_idx order.")

    pairs = np.column_stack([customer_positions.astype(int), item_positions_array.astype(int)])
    if pairs.size == 0:
        return sp.csr_matrix((len(customer_idx), len(item_values)), dtype=np.float32), item_values

    unique_pairs = np.unique(pairs, axis=0)
    data = np.ones(unique_pairs.shape[0], dtype=np.float32)
    matrix = sp.csr_matrix(
        (data, (unique_pairs[:, 0], unique_pairs[:, 1])),
        shape=(len(customer_idx), len(item_values)),
    )
    return matrix, item_values


def _distance_to_similarity(
    distances: np.ndarray,
    *,
    kernel: str,
    bandwidth: float,
) -> np.ndarray:
    # Inputs: distance values plus kernel options. Outputs: bounded similarity weights.
    """Convert non-negative distances into bounded similarity weights."""

    if kernel == "rbf":
        safe_bandwidth = max(float(bandwidth), 1e-12)
        return np.exp(-(distances ** 2) / (2.0 * safe_bandwidth ** 2))
    if kernel == "inverse":
        return 1.0 / (1.0 + distances)
    if kernel == "cosine":
        return np.clip(1.0 - distances, 0.0, 1.0)
    raise ValueError(f"Unsupported similarity kernel: {kernel}")


@dataclass
class EmbeddingDistanceConfig:
    """Configuration for embedding-distance similarity construction."""

    customer_col: str = "CustomerID"
    item_col: str = "Item Code"
    item_idx_col: str = "item_idx"
    metric: str = "euclidean"
    similarity_kernel: str = "rbf"
    n_neighbors: int = 25
    bandwidth: float | None = None
    include_self: bool = True
    symmetrize: str = "max"
    customer_customer_weight: float = 1.0
    customer_item_weight: float = 1.0
    normalize_customer_customer: bool = False
    normalize_customer_item: bool = False
    drop_missing_customer_id: bool = True
    drop_missing_item_code: bool = True


class embedding_distance_to_similarity:
    """Build customer-customer similarity from embedding space and customer-item matrix from history."""

    input_type = None
    output_type = ARTIFACT_RELATION_MATRICES
    input_artifacts = {
        "dataframe": ArtifactSpec(
            name="dataframe",
            kind=ARTIFACT_DATAFRAME,
            dense=True,
            description="Transaction dataframe with customer and item columns.",
        ),
        "embedding": ArtifactSpec(
            name="embedding",
            kind=ARTIFACT_MATRIX,
            dense=True,
            description="Customer embedding matrix aligned with customer order.",
        ),
    }
    output_artifacts = {
        "user_user_matrix": ArtifactSpec(
            name="user_user_matrix",
            kind=ARTIFACT_MATRIX,
            dense=False,
            description="Customer-customer similarity matrix derived from embedding distances.",
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
            description="Customer ids aligned with embedding and user-user matrix rows.",
        ),
        "item_idx": ArtifactSpec(
            name="item_idx",
            kind=ARTIFACT_INDEX_ARRAY,
            dense=True,
            description="Item ids aligned with the user-item matrix columns.",
        ),
        "combined_matrix": ArtifactSpec(
            name="combined_matrix",
            kind=ARTIFACT_MATRIX,
            dense=False,
            description="Block adjacency matrix combining customer-customer and customer-item relations.",
        ),
    }
    contract = StageContract(
        input_type=input_type,
        output_type=output_type,
        input_artifacts=input_artifacts,
        output_artifacts=output_artifacts,
        dense=False,
        description="Convert customer embeddings into a similarity graph plus purchase matrix.",
    )

    def __init__(
        self,
        customer_col: str = "CustomerID",
        item_col: str = "Item Code",
        item_idx_col: str = "item_idx",
        metric: str = "euclidean",
        similarity_kernel: str = "rbf",
        n_neighbors: int = 25,
        bandwidth: float | None = None,
        include_self: bool = True,
        symmetrize: str = "max",
        customer_customer_weight: float = 1.0,
        customer_item_weight: float = 1.0,
        normalize_customer_customer: bool = False,
        normalize_customer_item: bool = False,
        drop_missing_customer_id: bool = True,
        drop_missing_item_code: bool = True,
    ) -> None:
        """Store runtime options for embedding-distance similarity construction."""

        self.config = EmbeddingDistanceConfig(
            customer_col=customer_col,
            item_col=item_col,
            item_idx_col=item_idx_col,
            metric=metric,
            similarity_kernel=similarity_kernel,
            n_neighbors=int(n_neighbors),
            bandwidth=bandwidth,
            include_self=bool(include_self),
            symmetrize=symmetrize,
            customer_customer_weight=float(customer_customer_weight),
            customer_item_weight=float(customer_item_weight),
            normalize_customer_customer=bool(normalize_customer_customer),
            normalize_customer_item=bool(normalize_customer_item),
            drop_missing_customer_id=bool(drop_missing_customer_id),
            drop_missing_item_code=bool(drop_missing_item_code),
        )
        self.customer_idx_: np.ndarray | None = None
        self.item_idx_: np.ndarray | None = None
        self.user_user_matrix_: sp.csr_matrix | None = None
        self.user_item_matrix_: sp.csr_matrix | None = None
        self.combined_matrix_: sp.csr_matrix | None = None

    def fit(
        self,
        dataframe: pd.DataFrame,
        embedding: np.ndarray,
        customer_idx: Sequence[Any] | np.ndarray | None = None,
    ) -> "embedding_distance_to_similarity":
        # Inputs: transaction dataframe, customer embedding matrix, and optional customer order. Outputs: fitted relation builder.
        """Build embedding-distance customer similarity and purchase matrix artifacts."""

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

        resolved_customer_idx = _resolve_customer_order(
            work,
            customer_col=self.config.customer_col,
            customer_idx=customer_idx,
        )
        embedding_array = np.asarray(embedding, dtype=float)
        if embedding_array.ndim != 2:
            raise ValueError("embedding must be a 2D matrix.")
        if embedding_array.shape[0] != resolved_customer_idx.shape[0]:
            raise ValueError(
                "embedding row count must match the resolved number of unique customers."
            )

        self.customer_idx_ = resolved_customer_idx
        self.user_user_matrix_ = self._build_embedding_similarity(embedding_array)
        self.user_item_matrix_, self.item_idx_ = _build_customer_item_matrix(
            work,
            customer_col=self.config.customer_col,
            item_col=self.config.item_col,
            item_idx_col=self.config.item_idx_col,
            customer_idx=resolved_customer_idx,
        )
        self.combined_matrix_ = self._build_combined_matrix(
            self.user_user_matrix_,
            self.user_item_matrix_,
        )
        return self

    def export_artifacts(self) -> dict[str, object]:
        # Inputs: fitted builder state. Outputs: exported matrices and ordering arrays.
        """Export similarity graph, purchase matrix, and ordering arrays."""

        if (
            self.user_user_matrix_ is None
            or self.user_item_matrix_ is None
            or self.customer_idx_ is None
            or self.item_idx_ is None
            or self.combined_matrix_ is None
        ):
            raise RuntimeError("The stage must be fitted before exporting artifacts.")
        return {
            "user_user_matrix": self.user_user_matrix_,
            "user_item_matrix": self.user_item_matrix_,
            "customer_idx": self.customer_idx_,
            "item_idx": self.item_idx_,
            "combined_matrix": self.combined_matrix_,
        }

    def _build_embedding_similarity(self, embedding: np.ndarray) -> sp.csr_matrix:
        # Inputs: dense customer embedding matrix. Outputs: sparse customer-customer similarity graph.
        """Build a sparse customer-customer similarity graph from nearest neighbors in embedding space."""

        n_customers = embedding.shape[0]
        if n_customers == 0:
            return sp.csr_matrix((0, 0), dtype=np.float32)

        requested_neighbors = max(int(self.config.n_neighbors), 1)
        effective_neighbors = min(requested_neighbors + int(self.config.include_self), n_customers)
        neighbors = NearestNeighbors(n_neighbors=effective_neighbors, metric=self.config.metric)
        neighbors.fit(embedding)
        distances, indices = neighbors.kneighbors(embedding)

        if self.config.include_self:
            rows = np.repeat(np.arange(n_customers), effective_neighbors)
            cols = indices.reshape(-1)
            values = distances.reshape(-1)
        else:
            rows = np.repeat(np.arange(n_customers), effective_neighbors - 1)
            cols = indices[:, 1:].reshape(-1)
            values = distances[:, 1:].reshape(-1)

        non_self = values[values > 0]
        if self.config.bandwidth is None:
            bandwidth = float(np.median(non_self)) if non_self.size else 1.0
        else:
            bandwidth = float(self.config.bandwidth)
        similarities = _distance_to_similarity(values, kernel=self.config.similarity_kernel, bandwidth=bandwidth)

        graph = sp.csr_matrix((similarities.astype(np.float32), (rows, cols)), shape=(n_customers, n_customers))
        if self.config.include_self:
            graph.setdiag(1.0)

        if self.config.symmetrize == "max":
            graph = graph.maximum(graph.T)
        elif self.config.symmetrize == "mean":
            graph = ((graph + graph.T) * 0.5).tocsr()
        elif self.config.symmetrize != "none":
            raise ValueError(f"Unsupported symmetrize mode: {self.config.symmetrize}")

        return graph.tocsr()

    def _build_combined_matrix(
        self,
        user_user_matrix: sp.csr_matrix,
        user_item_matrix: sp.csr_matrix,
    ) -> sp.csr_matrix:
        # Inputs: customer-customer and customer-item matrices. Outputs: combined block adjacency matrix.
        if self.config.normalize_customer_customer:
            user_user_matrix = self._row_normalize(user_user_matrix)
        if self.config.normalize_customer_item:
            user_item_matrix = self._row_normalize(user_item_matrix)
        weighted_user_user = user_user_matrix.multiply(self.config.customer_customer_weight)
        weighted_user_item = user_item_matrix.multiply(self.config.customer_item_weight)
        n_items = user_item_matrix.shape[1]
        zero_items = sp.csr_matrix((n_items, n_items), dtype=weighted_user_item.dtype)
        return sp.bmat(
            [[weighted_user_user, weighted_user_item], [weighted_user_item.T, zero_items]],
            format="csr",
        )

    @staticmethod
    def _row_normalize(matrix: sp.csr_matrix) -> sp.csr_matrix:
        # Inputs: sparse matrix. Outputs: row-normalized sparse matrix with zero rows preserved.
        matrix = matrix.tocsr(copy=True)
        row_sums = np.asarray(matrix.sum(axis=1)).ravel()
        non_zero = row_sums > 0
        if not np.any(non_zero):
            return matrix
        inv_row_sums = np.zeros_like(row_sums, dtype=np.float32)
        inv_row_sums[non_zero] = 1.0 / row_sums[non_zero]
        return sp.diags(inv_row_sums).dot(matrix).tocsr()


EmbeddingDistanceToSimilarity = embedding_distance_to_similarity


__all__ = [
    "EmbeddingDistanceConfig",
    "EmbeddingDistanceToSimilarity",
    "embedding_distance_to_similarity",
]
