"""Offline evaluation utilities for recommendation experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Hashable, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd


UserId = Hashable
ItemId = Hashable


@dataclass(frozen=True)
class EvaluationSplitResult:
    """Masked evaluation dataframe plus held-out single-item targets."""

    masked_df: pd.DataFrame
    ground_truth: dict[UserId, ItemId]
    evaluated_user_ids: list[UserId]
    skipped_user_ids: list[UserId]
    split_details: dict[str, Any] | None = None


def _normalize_user_ids(user_ids: Iterable[UserId]) -> list[UserId]:
    seen: set[UserId] = set()
    ordered: list[UserId] = []
    for user_id in user_ids:
        if user_id in seen:
            continue
        seen.add(user_id)
        ordered.append(user_id)
    return ordered


def _validate_required_columns(df: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in df.columns]
    if missing:
        raise KeyError(f"Missing required dataframe columns: {missing}")


def _ordered_selected_users(df: pd.DataFrame, user_ids: Iterable[UserId], user_col: str) -> tuple[list[UserId], list[UserId]]:
    ordered_users = _normalize_user_ids(user_ids)
    present_users = set(df[user_col].dropna().tolist())
    eligible = [user_id for user_id in ordered_users if user_id in present_users]
    return ordered_users, eligible


def _build_working_frame(df: pd.DataFrame) -> pd.DataFrame:
    work = df.copy()
    work["_original_order"] = np.arange(len(work))
    return work


def remove_last_purchase(
    df: pd.DataFrame,
    user_ids: Iterable[UserId],
    user_col: str = "CustomerID",
    item_col: str = "item_idx",
    date_col: str = "Order Date",
) -> EvaluationSplitResult:
    """Hide the last chronological purchase row for each selected user."""

    _validate_required_columns(df, (user_col, item_col, date_col))
    work = _build_working_frame(df)
    ordered_users, selected_users = _ordered_selected_users(work, user_ids, user_col)

    drop_indices: list[Any] = []
    ground_truth: dict[UserId, ItemId] = {}
    evaluated_user_ids: list[UserId] = []

    for user_id in selected_users:
        user_rows = work[work[user_col] == user_id]
        if len(user_rows) < 2:
            continue

        ordered_rows = user_rows.assign(
            _parsed_date=pd.to_datetime(user_rows[date_col], errors="coerce")
        ).sort_values(
            by=["_parsed_date", "_original_order"],
            kind="stable",
        )
        hidden_row = ordered_rows.iloc[-1]
        drop_indices.append(hidden_row.name)
        ground_truth[user_id] = hidden_row[item_col]
        evaluated_user_ids.append(user_id)

    masked_df = work.drop(index=drop_indices).drop(columns="_original_order")
    skipped_user_ids = [user_id for user_id in ordered_users if user_id not in ground_truth]
    return EvaluationSplitResult(
        masked_df=masked_df,
        ground_truth=ground_truth,
        evaluated_user_ids=evaluated_user_ids,
        skipped_user_ids=skipped_user_ids,
    )


def remove_whole_item(
    df: pd.DataFrame,
    user_ids: Iterable[UserId],
    user_col: str = "CustomerID",
    item_col: str = "item_idx",
    seed: int | None = None,
    rng: np.random.Generator | None = None,
) -> EvaluationSplitResult:
    """Hide one randomly chosen user-item pair by removing all matching rows."""

    _validate_required_columns(df, (user_col, item_col))
    work = df.copy()
    ordered_users, selected_users = _ordered_selected_users(work, user_ids, user_col)
    random_state = rng if rng is not None else np.random.default_rng(seed)

    drop_indices: list[Any] = []
    ground_truth: dict[UserId, ItemId] = {}
    evaluated_user_ids: list[UserId] = []

    for user_id in selected_users:
        user_rows = work[work[user_col] == user_id]
        item_counts = user_rows[item_col].value_counts(dropna=False)
        eligible_items = [
            item_id
            for item_id, count in item_counts.items()
            if len(user_rows) - int(count) > 0
        ]
        if len(eligible_items) < 2:
            continue

        chosen_item = eligible_items[int(random_state.integers(0, len(eligible_items)))]
        pair_mask = (work[user_col] == user_id) & (work[item_col] == chosen_item)
        pair_indices = work.index[pair_mask].tolist()
        if not pair_indices or len(user_rows) == len(pair_indices):
            continue

        drop_indices.extend(pair_indices)
        ground_truth[user_id] = chosen_item
        evaluated_user_ids.append(user_id)

    masked_df = work.drop(index=drop_indices)
    skipped_user_ids = [user_id for user_id in ordered_users if user_id not in ground_truth]
    return EvaluationSplitResult(
        masked_df=masked_df,
        ground_truth=ground_truth,
        evaluated_user_ids=evaluated_user_ids,
        skipped_user_ids=skipped_user_ids,
    )


def hold_out_last_novel_tail_item(
    df: pd.DataFrame,
    user_col: str = "CustomerID",
    item_col: str = "item_idx",
    date_col: str = "Order Date",
    tail_fraction: float = 0.2,
    min_training_customer_support: int = 0,
) -> EvaluationSplitResult:
    """Hold out each eligible user's last novel item in the final tail of their history.

    Eligibility rule:
    - order each user's rows chronologically using ``date_col`` and original row order
    - find the last row where the item has not appeared earlier for that user
    - keep the user for evaluation only when that row lies in the final ``tail_fraction`` of the row sequence
    - training history for that user becomes only the prefix before that held-out row
    - all rows from the held-out row onward are removed from training for that user
    """

    _validate_required_columns(df, (user_col, item_col, date_col))
    if not (0.0 < float(tail_fraction) <= 1.0):
        raise ValueError("tail_fraction must be in the interval (0, 1].")
    if int(min_training_customer_support) < 0:
        raise ValueError("min_training_customer_support must be non-negative.")

    work = _build_working_frame(df)
    work["_parsed_date"] = pd.to_datetime(work[date_col], errors="coerce")

    drop_indices: list[Any] = []
    ground_truth: dict[UserId, ItemId] = {}
    evaluated_user_ids: list[UserId] = []
    skipped_user_ids: list[UserId] = []
    eligible_records: list[dict[str, Any]] = []

    total_rows = int(len(work))
    all_user_ids = pd.unique(work[user_col].dropna()).tolist()
    for user_id, user_rows in work.groupby(user_col, sort=False):
        ordered_rows = user_rows.sort_values(
            by=["_parsed_date", "_original_order"],
            kind="stable",
        ).reset_index()
        n_rows = int(len(ordered_rows))
        if n_rows < 2:
            skipped_user_ids.append(user_id)
            continue

        seen_items: set[ItemId] = set()
        novel_positions: list[int] = []
        for pos, item_id in enumerate(ordered_rows[item_col].tolist()):
            if item_id in seen_items:
                continue
            seen_items.add(item_id)
            novel_positions.append(pos)

        if not novel_positions:
            skipped_user_ids.append(user_id)
            continue

        heldout_pos = int(novel_positions[-1])
        tail_start_pos = int(np.floor((1.0 - float(tail_fraction)) * n_rows))
        if heldout_pos < tail_start_pos:
            skipped_user_ids.append(user_id)
            continue

        heldout_row = ordered_rows.iloc[heldout_pos]
        removed_rows = ordered_rows.iloc[heldout_pos:]
        drop_indices.extend(removed_rows["index"].tolist())
        ground_truth[user_id] = heldout_row[item_col]
        evaluated_user_ids.append(user_id)
        eligible_records.append(
            {
                "user_id": user_id,
                "history_length": n_rows,
                "heldout_position": heldout_pos,
                "tail_start_position": tail_start_pos,
                "heldout_item": heldout_row[item_col],
                "heldout_item_code": heldout_row.get("Item Code"),
                "removed_rows": int(len(removed_rows)),
                "prefix_rows_kept": heldout_pos,
            }
        )

    masked_df = work.drop(index=drop_indices).drop(columns=["_original_order", "_parsed_date"])
    removed_purchase_count = int(len(drop_indices))

    support_by_item = (
        masked_df.groupby(item_col)[user_col]
        .nunique(dropna=True)
        .to_dict()
    )
    if int(min_training_customer_support) > 0:
        filtered_ground_truth = {
            user_id: heldout_item
            for user_id, heldout_item in ground_truth.items()
            if int(support_by_item.get(heldout_item, 0)) >= int(min_training_customer_support)
        }
        removed_by_support = [user_id for user_id in ground_truth if user_id not in filtered_ground_truth]
        skipped_user_ids.extend(removed_by_support)
        skipped_user_ids = _normalize_user_ids(skipped_user_ids)
        evaluated_user_ids = [user_id for user_id in evaluated_user_ids if user_id in filtered_ground_truth]
        ground_truth = filtered_ground_truth
        eligible_records = [
            record
            for record in eligible_records
            if int(support_by_item.get(record["heldout_item"], 0)) >= int(min_training_customer_support)
        ]
    else:
        removed_by_support = []

    unique_heldout_items = {item_id for item_id in ground_truth.values()}
    skipped_initial = [user_id for user_id in all_user_ids if user_id not in set(evaluated_user_ids) and user_id not in set(removed_by_support)]
    details = {
        "split_mode": "hold_out_last_novel_tail_item",
        "tail_fraction": float(tail_fraction),
        "min_training_customer_support": int(min_training_customer_support),
        "original_purchase_count": total_rows,
        "training_purchase_count": int(len(masked_df)),
        "removed_purchase_count": removed_purchase_count,
        "validation_customer_count": int(len(evaluated_user_ids)),
        "support_filtered_customer_count": int(len(removed_by_support)),
        "eligible_customer_count_before_support_filter": int(len(evaluated_user_ids) + len(removed_by_support)),
        "unique_heldout_item_count": int(len(unique_heldout_items)),
        "skipped_customer_count": int(len(skipped_initial)),
        "eligible_records": eligible_records,
    }
    return EvaluationSplitResult(
        masked_df=masked_df,
        ground_truth=ground_truth,
        evaluated_user_ids=evaluated_user_ids,
        skipped_user_ids=skipped_user_ids,
        split_details=details,
    )


def _top_k_items(
    predictions: Mapping[UserId, Sequence[ItemId]],
    user_id: UserId,
    k: int,
) -> Sequence[ItemId]:
    if k <= 0:
        return ()
    ranked_items = predictions.get(user_id, ())
    return ranked_items[:k]


def recall_at_k(
    predictions: Mapping[UserId, Sequence[ItemId]],
    ground_truth: Mapping[UserId, ItemId],
    k: int = 10,
) -> float:
    """Average single-target recall at ``k`` over users in ``ground_truth``."""

    if not ground_truth or k <= 0:
        return 0.0

    total = 0.0
    for user_id, target_item in ground_truth.items():
        total += 1.0 if target_item in _top_k_items(predictions, user_id, k) else 0.0
    return total / len(ground_truth)


def mrr_at_k(
    predictions: Mapping[UserId, Sequence[ItemId]],
    ground_truth: Mapping[UserId, ItemId],
    k: int = 10,
) -> float:
    """Average reciprocal rank at ``k`` over users in ``ground_truth``."""

    if not ground_truth or k <= 0:
        return 0.0

    total = 0.0
    for user_id, target_item in ground_truth.items():
        top_items = _top_k_items(predictions, user_id, k)
        for rank, item_id in enumerate(top_items, start=1):
            if item_id == target_item:
                total += 1.0 / rank
                break
    return total / len(ground_truth)


def ndcg_at_k(
    predictions: Mapping[UserId, Sequence[ItemId]],
    ground_truth: Mapping[UserId, ItemId],
    k: int = 10,
) -> float:
    """Average NDCG at ``k`` for a single held-out relevant item per user."""

    if not ground_truth or k <= 0:
        return 0.0

    total = 0.0
    for user_id, target_item in ground_truth.items():
        top_items = _top_k_items(predictions, user_id, k)
        for rank, item_id in enumerate(top_items, start=1):
            if item_id == target_item:
                total += 1.0 / math.log2(rank + 1.0)
                break
    return total / len(ground_truth)


__all__ = [
    "EvaluationSplitResult",
    "hold_out_last_novel_tail_item",
    "mrr_at_k",
    "ndcg_at_k",
    "recall_at_k",
    "remove_last_purchase",
    "remove_whole_item",
]
