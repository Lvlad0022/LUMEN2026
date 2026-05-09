"""Group-share popularity recommender baseline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

try:
    from ...pipeline_contracts import (
        ARTIFACT_DATAFRAME,
        ARTIFACT_INDEX_ARRAY,
        ARTIFACT_MODEL,
        ArtifactSpec,
        StageContract,
    )
except ImportError:  # pragma: no cover - supports direct imports from src/
    from pipeline_contracts import (  # type: ignore
        ARTIFACT_DATAFRAME,
        ARTIFACT_INDEX_ARRAY,
        ARTIFACT_MODEL,
        ArtifactSpec,
        StageContract,
    )


@dataclass(frozen=True)
class GroupSharePopularityConfig:
    """Configuration for the group-share popularity baseline."""

    customer_col: str = "CustomerID"
    item_col: str = "Item Code"
    item_idx_col: str = "item_idx"
    group_col: str = "Product group"
    drop_missing_customer_id: bool = True
    drop_missing_item: bool = True
    drop_missing_group: bool = True
    fallback_to_global_popularity: bool = True


class GroupSharePopularityRecommender:
    """Recommend unseen items from a customer's group mix and in-group popularity."""

    input_type = None
    output_type = ARTIFACT_MODEL
    input_artifacts = {
        "dataframe": ArtifactSpec(
            name="dataframe",
            kind=ARTIFACT_DATAFRAME,
            dense=True,
            description="Transaction dataframe with customer, item, and product-group columns.",
        ),
    }
    output_artifacts = {
        "customer_idx": ArtifactSpec(
            name="customer_idx",
            kind=ARTIFACT_INDEX_ARRAY,
            dense=True,
            description="Customer ids seen during fitting.",
        ),
        "item_idx": ArtifactSpec(
            name="item_idx",
            kind=ARTIFACT_INDEX_ARRAY,
            dense=True,
            description="Item ids seen during fitting.",
        ),
        "model": ArtifactSpec(
            name="model",
            kind=ARTIFACT_MODEL,
            dense=False,
            description="Fitted group-share popularity recommender.",
        ),
    }
    contract = StageContract(
        input_type=input_type,
        output_type=output_type,
        input_artifacts=input_artifacts,
        output_artifacts=output_artifacts,
        dense=False,
        description="Baseline recommender using customer group shares and filtered in-group item popularity.",
    )

    def __init__(
        self,
        customer_col: str = "CustomerID",
        item_col: str = "Item Code",
        item_idx_col: str = "item_idx",
        group_col: str = "Product group",
        drop_missing_customer_id: bool = True,
        drop_missing_item: bool = True,
        drop_missing_group: bool = True,
        fallback_to_global_popularity: bool = True,
    ) -> None:
        self.config = GroupSharePopularityConfig(
            customer_col=customer_col,
            item_col=item_col,
            item_idx_col=item_idx_col,
            group_col=group_col,
            drop_missing_customer_id=bool(drop_missing_customer_id),
            drop_missing_item=bool(drop_missing_item),
            drop_missing_group=bool(drop_missing_group),
            fallback_to_global_popularity=bool(fallback_to_global_popularity),
        )
        self.customer_idx_: np.ndarray | None = None
        self.item_idx_: np.ndarray | None = None
        self.customer_group_share_: dict[int, dict[Any, float]] = {}
        self.customer_seen_items_: dict[int, set[int]] = {}
        self.group_item_counts_: dict[Any, dict[int, float]] = {}
        self.global_item_counts_: dict[int, float] = {}

    def fit(self, dataframe: pd.DataFrame) -> "GroupSharePopularityRecommender":
        """Fit all count tables needed for recommendation."""

        item_source = self.config.item_idx_col if self.config.item_idx_col in dataframe.columns else self.config.item_col
        required = [self.config.customer_col, item_source, self.config.group_col]
        missing = [column for column in required if column not in dataframe.columns]
        if missing:
            raise KeyError(f"Missing required columns for group-share popularity baseline: {missing}")

        work = dataframe[[self.config.customer_col, item_source, self.config.group_col]].copy()
        if self.config.drop_missing_customer_id:
            work = work[~work[self.config.customer_col].isna()].copy()
        if self.config.drop_missing_item:
            work = work[~work[item_source].isna()].copy()
        if self.config.drop_missing_group:
            work = work[~work[self.config.group_col].isna()].copy()
        if work.empty:
            raise ValueError("No rows remain after filtering missing ids/groups.")

        work["_customer"] = pd.to_numeric(work[self.config.customer_col], errors="raise").astype(int)
        work["_item"] = pd.to_numeric(work[item_source], errors="raise").astype(int)
        work["_group"] = work[self.config.group_col].astype(str)

        self.customer_idx_ = pd.unique(work["_customer"]).astype(int)
        self.item_idx_ = pd.unique(work["_item"]).astype(int)

        customer_group_counts = work.groupby(["_customer", "_group"], sort=False).size()
        customer_totals = work.groupby("_customer", sort=False).size()
        self.customer_group_share_ = {}
        for (customer_id, group_id), count in customer_group_counts.items():
            total = float(customer_totals.loc[customer_id])
            self.customer_group_share_.setdefault(int(customer_id), {})[group_id] = float(count) / total

        self.customer_seen_items_ = {
            int(customer_id): set(group["_item"].astype(int).tolist())
            for customer_id, group in work.groupby("_customer", sort=False)
        }

        group_item_counts = work.groupby(["_group", "_item"], sort=False).size()
        self.group_item_counts_ = {}
        for (group_id, item_id), count in group_item_counts.items():
            self.group_item_counts_.setdefault(group_id, {})[int(item_id)] = float(count)

        self.global_item_counts_ = {
            int(item_id): float(count)
            for item_id, count in work.groupby("_item", sort=False).size().items()
        }
        return self

    def export_artifacts(self) -> dict[str, object]:
        """Export artifacts consumed by the validation pipeline."""

        if self.customer_idx_ is None or self.item_idx_ is None:
            raise RuntimeError("The recommender must be fitted before exporting artifacts.")
        return {
            "customer_idx": self.customer_idx_,
            "item_idx": self.item_idx_,
            "model": self,
        }

    def predict(self, customer_idx: int, num_prediction: int = 5) -> list[int]:
        """Return top unseen item ids for one customer."""

        if self.customer_idx_ is None:
            raise RuntimeError("The recommender must be fitted before calling predict().")
        customer_id = int(customer_idx)
        if customer_id not in self.customer_group_share_:
            raise KeyError(f"Unknown customer_idx: {customer_id}")

        seen_items = self.customer_seen_items_.get(customer_id, set())
        scores: dict[int, float] = {}

        for group_id, group_share in self.customer_group_share_[customer_id].items():
            item_counts = self.group_item_counts_.get(group_id, {})
            candidate_counts = {
                item_id: count
                for item_id, count in item_counts.items()
                if item_id not in seen_items
            }
            denominator = float(sum(candidate_counts.values()))
            if denominator <= 0.0:
                continue
            for item_id, count in candidate_counts.items():
                scores[item_id] = scores.get(item_id, 0.0) + float(group_share) * (float(count) / denominator)

        if len(scores) < int(num_prediction) and self.config.fallback_to_global_popularity:
            self._add_global_fallback(scores, seen_items)

        if not scores:
            return []

        ranked = sorted(scores.items(), key=lambda pair: (-pair[1], pair[0]))
        return [int(item_id) for item_id, score in ranked[: int(num_prediction)] if np.isfinite(score)]

    def _add_global_fallback(self, scores: dict[int, float], seen_items: set[int]) -> None:
        """Fill sparse recommendations with globally popular unseen items."""

        if not self.global_item_counts_:
            return
        max_existing = max(scores.values(), default=0.0)
        total = float(sum(count for item_id, count in self.global_item_counts_.items() if item_id not in seen_items))
        if total <= 0.0:
            return
        fallback_scale = max_existing * 1e-6 if max_existing > 0.0 else 1e-12
        for item_id, count in self.global_item_counts_.items():
            if item_id in seen_items or item_id in scores:
                continue
            scores[item_id] = fallback_scale * (float(count) / total)


GroupSharePopularityBaseline = GroupSharePopularityRecommender


__all__ = [
    "GroupSharePopularityBaseline",
    "GroupSharePopularityConfig",
    "GroupSharePopularityRecommender",
]
