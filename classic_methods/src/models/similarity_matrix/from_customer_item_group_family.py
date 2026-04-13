"""Construct a heterogeneous customer-item-group-family graph."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd
import scipy.sparse as sp

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


def _as_ordered_unique(series: pd.Series) -> np.ndarray:
    # Inputs: pandas Series. Outputs: unique non-null values in first-seen order.
    return pd.unique(series.dropna())


def _binary_incidence(
    left_positions: np.ndarray,
    right_positions: np.ndarray,
    *,
    n_left: int,
    n_right: int,
) -> sp.csr_matrix:
    # Inputs: aligned left/right positions and target sizes. Outputs: binary sparse incidence matrix.
    if left_positions.size == 0 or right_positions.size == 0:
        return sp.csr_matrix((n_left, n_right), dtype=np.float32)
    pairs = np.column_stack([left_positions.astype(int), right_positions.astype(int)])
    unique_pairs = np.unique(pairs, axis=0)
    data = np.ones(unique_pairs.shape[0], dtype=np.float32)
    return sp.csr_matrix(
        (data, (unique_pairs[:, 0], unique_pairs[:, 1])),
        shape=(n_left, n_right),
    )


@dataclass
class CustomerItemGroupFamilyConfig:
    """Configuration for heterogeneous graph construction."""

    customer_col: str = "CustomerID"
    item_col: str = "Item Code"
    item_idx_col: str = "item_idx"
    group_col: str = "Product group"
    family_col: str = "Product family"
    customer_item_weight: float = 1.0
    item_group_weight: float = 1.0
    item_family_weight: float = 1.0
    group_family_weight: float = 1.0
    drop_missing_customer_id: bool = True
    drop_missing_item_code: bool = True
    drop_missing_group: bool = True
    drop_missing_family: bool = True


class customer_item_group_family_to_similarity:
    """Build a combined heterogeneous adjacency for customer-item-group-family Katz."""

    input_type = None
    output_type = ARTIFACT_RELATION_MATRICES
    input_artifacts = {
        "dataframe": ArtifactSpec(
            name="dataframe",
            kind=ARTIFACT_DATAFRAME,
            dense=True,
            description="Transaction dataframe with customer, item, group, and family columns.",
        ),
    }
    output_artifacts = {
        "customer_idx": ArtifactSpec(
            name="customer_idx",
            kind=ARTIFACT_INDEX_ARRAY,
            dense=True,
            description="Customer ids aligned with the first block of the heterogeneous graph.",
        ),
        "item_idx": ArtifactSpec(
            name="item_idx",
            kind=ARTIFACT_INDEX_ARRAY,
            dense=True,
            description="Item ids aligned with the second block of the heterogeneous graph.",
        ),
        "combined_matrix": ArtifactSpec(
            name="combined_matrix",
            kind=ARTIFACT_MATRIX,
            dense=False,
            description="Combined adjacency over customers, items, groups, and families.",
        ),
    }
    contract = StageContract(
        input_type=input_type,
        output_type=output_type,
        input_artifacts=input_artifacts,
        output_artifacts=output_artifacts,
        dense=False,
        description="Convert transactional hierarchy columns into a heterogeneous graph adjacency.",
    )

    def __init__(
        self,
        customer_col: str = "CustomerID",
        item_col: str = "Item Code",
        item_idx_col: str = "item_idx",
        group_col: str = "Product group",
        family_col: str = "Product family",
        customer_item_weight: float = 1.0,
        item_group_weight: float = 1.0,
        item_family_weight: float = 1.0,
        group_family_weight: float = 1.0,
        drop_missing_customer_id: bool = True,
        drop_missing_item_code: bool = True,
        drop_missing_group: bool = True,
        drop_missing_family: bool = True,
    ) -> None:
        # Inputs: column names, relation weights, and filtering flags. Outputs: initialized heterogeneous graph builder.
        self.config = CustomerItemGroupFamilyConfig(
            customer_col=customer_col,
            item_col=item_col,
            item_idx_col=item_idx_col,
            group_col=group_col,
            family_col=family_col,
            customer_item_weight=float(customer_item_weight),
            item_group_weight=float(item_group_weight),
            item_family_weight=float(item_family_weight),
            group_family_weight=float(group_family_weight),
            drop_missing_customer_id=bool(drop_missing_customer_id),
            drop_missing_item_code=bool(drop_missing_item_code),
            drop_missing_group=bool(drop_missing_group),
            drop_missing_family=bool(drop_missing_family),
        )
        self.customer_idx_: np.ndarray | None = None
        self.item_idx_: np.ndarray | None = None
        self.group_idx_: np.ndarray | None = None
        self.family_idx_: np.ndarray | None = None
        self.combined_matrix_: sp.csr_matrix | None = None

    def fit(self, dataframe: pd.DataFrame) -> "customer_item_group_family_to_similarity":
        # Inputs: transaction dataframe. Outputs: fitted heterogeneous graph builder.
        required = {
            self.config.customer_col,
            self.config.group_col,
            self.config.family_col,
        }
        item_source = self.config.item_idx_col if self.config.item_idx_col in dataframe.columns else self.config.item_col
        required.add(item_source)
        missing = [column for column in required if column not in dataframe.columns]
        if missing:
            raise KeyError(f"Missing required columns for heterogeneous graph stage: {missing}")

        work = dataframe.copy()
        if self.config.drop_missing_customer_id:
            work = work[~work[self.config.customer_col].isna()].copy()
        if self.config.drop_missing_item_code:
            work = work[~work[item_source].isna()].copy()
        if self.config.drop_missing_group:
            work = work[~work[self.config.group_col].isna()].copy()
        if self.config.drop_missing_family:
            work = work[~work[self.config.family_col].isna()].copy()

        customer_idx = _as_ordered_unique(work[self.config.customer_col])
        item_idx = _as_ordered_unique(work[item_source])
        group_idx = _as_ordered_unique(work[self.config.group_col].astype(str))
        family_idx = _as_ordered_unique(work[self.config.family_col].astype(str))

        customer_lookup = {value: pos for pos, value in enumerate(customer_idx.tolist())}
        item_lookup = {value: pos for pos, value in enumerate(item_idx.tolist())}
        group_lookup = {value: pos for pos, value in enumerate(group_idx.tolist())}
        family_lookup = {value: pos for pos, value in enumerate(family_idx.tolist())}

        customer_positions = work[self.config.customer_col].map(customer_lookup).to_numpy()
        item_positions = work[item_source].map(item_lookup).to_numpy()
        group_positions = work[self.config.group_col].astype(str).map(group_lookup).to_numpy()
        family_positions = work[self.config.family_col].astype(str).map(family_lookup).to_numpy()

        if np.any(pd.isna(customer_positions)) or np.any(pd.isna(item_positions)):
            raise ValueError("Customer-item relations could not be aligned to the resolved index order.")
        if np.any(pd.isna(group_positions)) or np.any(pd.isna(family_positions)):
            raise ValueError("Group-family relations could not be aligned to the resolved index order.")

        customer_item = _binary_incidence(
            customer_positions,
            item_positions,
            n_left=len(customer_idx),
            n_right=len(item_idx),
        )

        item_group_pairs = (
            work[[item_source, self.config.group_col]]
            .drop_duplicates()
            .assign(
                _item=lambda df: df[item_source].map(item_lookup).astype(int),
                _group=lambda df: df[self.config.group_col].astype(str).map(group_lookup).astype(int),
            )
        )
        item_group = _binary_incidence(
            item_group_pairs["_item"].to_numpy(),
            item_group_pairs["_group"].to_numpy(),
            n_left=len(item_idx),
            n_right=len(group_idx),
        )

        item_family_pairs = (
            work[[item_source, self.config.family_col]]
            .drop_duplicates()
            .assign(
                _item=lambda df: df[item_source].map(item_lookup).astype(int),
                _family=lambda df: df[self.config.family_col].astype(str).map(family_lookup).astype(int),
            )
        )
        item_family = _binary_incidence(
            item_family_pairs["_item"].to_numpy(),
            item_family_pairs["_family"].to_numpy(),
            n_left=len(item_idx),
            n_right=len(family_idx),
        )

        group_family_pairs = (
            work[[self.config.group_col, self.config.family_col]]
            .drop_duplicates()
            .assign(
                _group=lambda df: df[self.config.group_col].astype(str).map(group_lookup).astype(int),
                _family=lambda df: df[self.config.family_col].astype(str).map(family_lookup).astype(int),
            )
        )
        group_family = _binary_incidence(
            group_family_pairs["_group"].to_numpy(),
            group_family_pairs["_family"].to_numpy(),
            n_left=len(group_idx),
            n_right=len(family_idx),
        )

        weighted_customer_item = customer_item.multiply(self.config.customer_item_weight)
        weighted_item_group = item_group.multiply(self.config.item_group_weight)
        weighted_item_family = item_family.multiply(self.config.item_family_weight)
        weighted_group_family = group_family.multiply(self.config.group_family_weight)

        zero_cc = sp.csr_matrix((len(customer_idx), len(customer_idx)), dtype=np.float32)
        zero_ii = sp.csr_matrix((len(item_idx), len(item_idx)), dtype=np.float32)
        zero_gg = sp.csr_matrix((len(group_idx), len(group_idx)), dtype=np.float32)
        zero_ff = sp.csr_matrix((len(family_idx), len(family_idx)), dtype=np.float32)
        zero_ci_g = sp.csr_matrix((len(customer_idx), len(group_idx)), dtype=np.float32)
        zero_ci_f = sp.csr_matrix((len(customer_idx), len(family_idx)), dtype=np.float32)
        zero_i_c = sp.csr_matrix((len(item_idx), len(customer_idx)), dtype=np.float32)
        zero_g_c = sp.csr_matrix((len(group_idx), len(customer_idx)), dtype=np.float32)
        zero_f_c = sp.csr_matrix((len(family_idx), len(customer_idx)), dtype=np.float32)
        zero_g_i = sp.csr_matrix((len(group_idx), len(item_idx)), dtype=np.float32)
        zero_f_i = sp.csr_matrix((len(family_idx), len(item_idx)), dtype=np.float32)

        combined = sp.bmat(
            [
                [zero_cc, weighted_customer_item, zero_ci_g, zero_ci_f],
                [zero_i_c, zero_ii, weighted_item_group, weighted_item_family],
                [zero_g_c, weighted_item_group.T, zero_gg, weighted_group_family],
                [zero_f_c, weighted_item_family.T, weighted_group_family.T, zero_ff],
            ],
            format="csr",
        )

        self.customer_idx_ = np.asarray(customer_idx)
        self.item_idx_ = np.asarray(item_idx)
        self.group_idx_ = np.asarray(group_idx)
        self.family_idx_ = np.asarray(family_idx)
        self.combined_matrix_ = combined
        return self

    def export_artifacts(self) -> dict[str, object]:
        # Inputs: fitted builder state. Outputs: exported heterogeneous graph artifacts.
        if self.customer_idx_ is None or self.item_idx_ is None or self.combined_matrix_ is None:
            raise RuntimeError("The stage must be fitted before exporting artifacts.")
        return {
            "customer_idx": self.customer_idx_,
            "item_idx": self.item_idx_,
            "combined_matrix": self.combined_matrix_,
        }


CustomerItemGroupFamilyToSimilarity = customer_item_group_family_to_similarity


__all__ = [
    "CustomerItemGroupFamilyConfig",
    "CustomerItemGroupFamilyToSimilarity",
    "customer_item_group_family_to_similarity",
]
