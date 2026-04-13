from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import math

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer

try:
    from ..pipeline_contracts import ARTIFACT_DATAFRAME, ARTIFACT_MATRIX, ArtifactSpec, StageContract
except ImportError:  # pragma: no cover - supports direct imports from src/
    from pipeline_contracts import ARTIFACT_DATAFRAME, ARTIFACT_MATRIX, ArtifactSpec, StageContract  # type: ignore


NUMERIC_COLUMNS = [
    "Invoiced qty (shipped)",
    "Ordered qty",
    "Invoiced price",
    "Invoiced price (TX)",
    "Cost of part",
    "Material cost of part",
    "Labor cost of part",
    "Overhead cost of part",
    "GM%",
    "# of unique products on a quote",
]

DATE_COLUMNS = [
    "Customer First Invoice Date",
    "Price last modified date in the ERP",
    "Born on date",
    "Invoice Date",
    "Order Date",
]

FULL_SHARE_COLUMNS = [
    "Product family",
    "Product group",
    "Customer industry",
    "Customer Region",
    "Top Customer Group",
    "Manufacturing Region",
]

TOP_SHARE_COLUMNS = [
    "Sales Channel - Internal",
    "Sales Channel - External",
]


@dataclass(frozen=True)
class CustomerFeatureConfig:
    customer_col: str = "CustomerID"
    item_col: str = "Item Code"
    product_group_col: str = "Product group"
    product_family_col: str = "Product family"
    revenue_col: str = "Invoiced price"
    bucket_size: int = 5
    top_n_channel_values: int = 12
    min_channel_count: int = 100
    feature_max_missing_ratio: float = 0.85
    feature_imputation_strategy: str = "median"


def read_lumen_csv(path: str | Path, nrows: int | None = None) -> pd.DataFrame:
    # Inputs: CSV path and optional row limit. Outputs: dataframe loaded with the repository encoding fallback chain.
    csv_path = Path(path)
    attempts = (
        {"encoding": "utf-8", "sep": ",", "low_memory": False},
        {"encoding": "utf-8-sig", "sep": ",", "low_memory": False},
        {"encoding": "utf-16", "sep": "|", "low_memory": False},
        {"encoding": "utf-16", "sep": ",", "low_memory": False},
        {"encoding": "cp1250", "sep": ",", "low_memory": False},
        {"encoding": "latin1", "sep": ",", "low_memory": False},
    )
    last_error: Exception | None = None
    for kwargs in attempts:
        try:
            return pd.read_csv(csv_path, nrows=nrows, **kwargs)
        except Exception as exc:  # pragma: no cover - fallback logic
            last_error = exc
    if last_error is not None:
        raise last_error
    return pd.read_csv(csv_path, nrows=nrows)


def schema_report(df: pd.DataFrame) -> pd.DataFrame:
    # Inputs: dataframe. Outputs: schema summary with dtype, non-null count, null ratio, and cardinality.
    return pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "non_null": df.notna().sum(),
            "null_ratio": df.isna().mean().round(4),
            "n_unique": df.nunique(dropna=True),
        }
    ).sort_values(["null_ratio", "n_unique"], ascending=[False, False])


def safe_numeric(series: pd.Series) -> pd.Series:
    # Inputs: a series. Outputs: numeric series with invalid values coerced to NaN.
    return pd.to_numeric(series, errors="coerce")


def safe_datetime(series: pd.Series) -> pd.Series:
    # Inputs: a series. Outputs: datetime series with invalid values coerced to NaT and 9999-prefix values nulled.
    text = series.astype("string")
    parsed = pd.to_datetime(text, errors="coerce")
    return parsed.mask(text.str.startswith("9999-", na=False), pd.NaT)


def entropy_from_series(series: pd.Series) -> float:
    # Inputs: categorical series. Outputs: normalized Shannon entropy over non-null values.
    values = series.dropna().astype(str)
    if values.empty:
        return float("nan")
    probabilities = values.value_counts(normalize=True).to_numpy(dtype=float)
    entropy = -(probabilities * np.log(probabilities + 1e-12)).sum()
    max_entropy = math.log(len(probabilities)) if len(probabilities) > 1 else 1.0
    return float(entropy / max_entropy) if max_entropy else 0.0


def top_share(series: pd.Series) -> float:
    # Inputs: categorical series. Outputs: the largest category share among non-null values.
    values = series.dropna()
    if values.empty:
        return float("nan")
    return float(values.value_counts(normalize=True).iloc[0])


def _prepare_base_dataframe(df: pd.DataFrame, config: CustomerFeatureConfig) -> pd.DataFrame:
    # Inputs: raw transaction dataframe and feature config. Outputs: cleaned working dataframe with derived columns.
    work = df.copy()
    for column in NUMERIC_COLUMNS:
        if column in work.columns:
            work[column] = safe_numeric(work[column])
    for column in DATE_COLUMNS:
        if column in work.columns:
            work[column] = safe_datetime(work[column])

    work = work[work[config.customer_col].notna()].copy()
    work[config.customer_col] = work[config.customer_col].astype(int)
    work["price_gap"] = work["Invoiced price (TX)"] - work["Invoiced price"]
    work["ship_ratio"] = work["Invoiced qty (shipped)"] / work["Ordered qty"].replace(0, np.nan)
    work["ship_gap"] = work["Ordered qty"] - work["Invoiced qty (shipped)"]
    work["under_shipped_flag"] = (work["ship_gap"] > 0).astype(float)
    work["margin_value"] = work["Invoiced price"] - work["Cost of part"]
    work["invoice_month"] = work["Invoice Date"].dt.month
    work["invoice_weekday"] = work["Invoice Date"].dt.weekday
    return work


def build_full_share_block(df: pd.DataFrame, customer_col: str, value_col: str) -> pd.DataFrame:
    # Inputs: dataframe plus customer and categorical columns. Outputs: per-customer shares for all observed values.
    if value_col not in df.columns:
        return pd.DataFrame()
    work = df[[customer_col, value_col]].dropna().copy()
    if work.empty:
        return pd.DataFrame()
    work[value_col] = work[value_col].astype(str)
    counts = work.groupby([customer_col, value_col]).size().unstack(fill_value=0)
    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    shares.columns = [f"share__{value_col}__{column}" for column in shares.columns]
    return shares


def build_top_value_share_block(
    df: pd.DataFrame,
    customer_col: str,
    value_col: str,
    top_n: int,
    min_global_count: int,
) -> pd.DataFrame:
    # Inputs: dataframe plus customer and categorical columns. Outputs: per-customer shares for globally frequent values.
    if value_col not in df.columns:
        return pd.DataFrame()
    work = df[[customer_col, value_col]].dropna().copy()
    if work.empty:
        return pd.DataFrame()
    work[value_col] = work[value_col].astype(str)
    top_values = (
        work[value_col].value_counts().loc[lambda series: series >= min_global_count].head(top_n).index.tolist()
    )
    if not top_values:
        return pd.DataFrame(index=pd.Index(sorted(work[customer_col].unique()), name=customer_col))
    counts = work[work[value_col].isin(top_values)].groupby([customer_col, value_col]).size().unstack(fill_value=0)
    counts = counts.reindex(columns=top_values, fill_value=0)
    shares = counts.div(counts.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    shares.columns = [f"share__{value_col}__{column}" for column in shares.columns]
    return shares


def build_product_group_bucket_map(
    df: pd.DataFrame,
    *,
    product_group_col: str,
    bucket_size: int,
) -> pd.DataFrame:
    # Inputs: dataframe, product-group column, and bucket size. Outputs: global ranking table with per-group bucket assignment.
    if product_group_col not in df.columns:
        return pd.DataFrame(columns=[product_group_col, "global_rank", "bucket_id", "bucket_label"])
    counts = (
        df[product_group_col]
        .dropna()
        .astype(str)
        .value_counts()
        .rename_axis(product_group_col)
        .reset_index(name="global_count")
    )
    counts["global_rank"] = np.arange(1, len(counts) + 1)
    counts["bucket_id"] = ((counts["global_rank"] - 1) // int(bucket_size)) + 1
    counts["bucket_start_rank"] = (counts["bucket_id"] - 1) * int(bucket_size) + 1
    counts["bucket_end_rank"] = counts["bucket_start_rank"] + int(bucket_size) - 1
    counts["bucket_label"] = [
        f"bucket_{start}_{end}"
        for start, end in zip(counts["bucket_start_rank"].tolist(), counts["bucket_end_rank"].tolist())
    ]
    return counts


def build_product_group_bucket_features(
    df: pd.DataFrame,
    *,
    customer_col: str,
    product_group_col: str,
    revenue_col: str,
    bucket_size: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    # Inputs: dataframe and bucket settings. Outputs: bucket-local share features plus bucket metadata table.
    bucket_map = build_product_group_bucket_map(df, product_group_col=product_group_col, bucket_size=bucket_size)
    if bucket_map.empty:
        return pd.DataFrame(), bucket_map

    work = df[[customer_col, product_group_col, revenue_col]].dropna(subset=[customer_col, product_group_col]).copy()
    work[product_group_col] = work[product_group_col].astype(str)
    work = work.merge(bucket_map[[product_group_col, "bucket_id", "bucket_label"]], on=product_group_col, how="left")

    group_counts = work.groupby([customer_col, product_group_col]).size().rename("group_count").reset_index()
    group_counts = group_counts.merge(bucket_map[[product_group_col, "bucket_label"]], on=product_group_col, how="left")
    bucket_totals = group_counts.groupby([customer_col, "bucket_label"])["group_count"].sum().rename("bucket_total")
    group_counts = group_counts.merge(bucket_totals.reset_index(), on=[customer_col, "bucket_label"], how="left")
    group_counts["bucket_local_share"] = group_counts["group_count"] / group_counts["bucket_total"].replace(0, np.nan)
    share_features = group_counts.pivot_table(
        index=customer_col,
        columns=product_group_col,
        values="bucket_local_share",
        fill_value=0.0,
    )
    share_features.columns = [f"bucketshare__{product_group_col}__{column}" for column in share_features.columns]

    magnitude = work.groupby([customer_col, "bucket_label"]).agg(
        total_count=(product_group_col, "size"),
        total_revenue=(revenue_col, "sum"),
    )
    count_features = magnitude["total_count"].unstack(fill_value=0.0)
    revenue_features = magnitude["total_revenue"].unstack(fill_value=0.0)
    count_features.columns = [f"{column}_total_count" for column in count_features.columns]
    revenue_features.columns = [f"{column}_total_revenue" for column in revenue_features.columns]

    features = pd.concat([share_features, count_features, revenue_features], axis=1).fillna(0.0)
    return features, bucket_map


def build_family_hierarchy_features(
    df: pd.DataFrame,
    *,
    customer_col: str,
    product_family_col: str,
    product_group_col: str,
    revenue_col: str,
) -> pd.DataFrame:
    # Inputs: dataframe and hierarchy columns. Outputs: family-level totals and within-family product-group share features.
    required = {customer_col, product_family_col, product_group_col, revenue_col}
    if not required.issubset(df.columns):
        return pd.DataFrame()

    work = df[[customer_col, product_family_col, product_group_col, revenue_col]].dropna(
        subset=[customer_col, product_family_col, product_group_col]
    ).copy()
    if work.empty:
        return pd.DataFrame()

    work[product_family_col] = work[product_family_col].astype(str)
    work[product_group_col] = work[product_group_col].astype(str)

    family_totals = work.groupby([customer_col, product_family_col]).agg(
        family_count=(product_group_col, "size"),
        family_revenue=(revenue_col, "sum"),
    )
    family_count = family_totals["family_count"].unstack(fill_value=0.0)
    family_revenue = family_totals["family_revenue"].unstack(fill_value=0.0)
    family_count.columns = [f"family__{column}__total_count" for column in family_count.columns]
    family_revenue.columns = [f"family__{column}__total_revenue" for column in family_revenue.columns]

    family_group = work.groupby([customer_col, product_family_col, product_group_col]).size().rename("count").reset_index()
    family_sums = family_group.groupby([customer_col, product_family_col])["count"].sum().rename("family_total").reset_index()
    family_group = family_group.merge(family_sums, on=[customer_col, product_family_col], how="left")
    family_group["share_in_family"] = family_group["count"] / family_group["family_total"].replace(0, np.nan)
    family_group["feature_name"] = [
        f"familygroup__{family}__{group}"
        for family, group in zip(
            family_group[product_family_col].tolist(),
            family_group[product_group_col].tolist(),
        )
    ]
    features = family_group.pivot_table(
        index=customer_col,
        columns="feature_name",
        values="share_in_family",
        fill_value=0.0,
    )
    return pd.concat([family_count, family_revenue, features], axis=1).fillna(0.0)


def describe_feature_groups(feature_df: pd.DataFrame) -> dict[str, list[str]]:
    # Inputs: engineered feature dataframe. Outputs: mapping from feature-group labels to matching columns.
    return {
        "activity": [column for column in feature_df.columns if column.startswith("activity__")],
        "time": [column for column in feature_df.columns if column.startswith("time__")],
        "aggregates": [column for column in feature_df.columns if column.startswith("agg__")],
        "full_shares": [column for column in feature_df.columns if column.startswith("share__")],
        "bucket_shares": [column for column in feature_df.columns if column.startswith("bucketshare__")],
        "bucket_magnitudes": [column for column in feature_df.columns if column.startswith("bucket_")],
        "family_hierarchy": [column for column in feature_df.columns if column.startswith("family__") or column.startswith("familygroup__")],
    }


def finalize_customer_feature_matrix(
    feature_df: pd.DataFrame,
    *,
    max_missing_ratio: float = 0.85,
    imputation_strategy: str = "median",
) -> tuple[pd.DataFrame, SimpleImputer]:
    # Inputs: raw engineered feature dataframe and imputation settings. Outputs: dense imputed feature dataframe and fitted imputer.
    cleaned = feature_df.replace([np.inf, -np.inf], np.nan).copy()
    keep_columns = cleaned.columns[cleaned.isna().mean() <= float(max_missing_ratio)].tolist()
    cleaned = cleaned[keep_columns]
    imputer = SimpleImputer(strategy=imputation_strategy)
    matrix = imputer.fit_transform(cleaned)
    finalized = pd.DataFrame(matrix, index=cleaned.index, columns=cleaned.columns)
    return finalized.astype(float), imputer


def build_customer_feature_matrix(
    df: pd.DataFrame,
    config: CustomerFeatureConfig | None = None,
    *,
    return_metadata: bool = False,
) -> pd.DataFrame | tuple[pd.DataFrame, dict[str, Any]]:
    # Inputs: transaction dataframe and optional config. Outputs: wide customer-level feature matrix, optionally with metadata.
    resolved_config = config or CustomerFeatureConfig()
    work = _prepare_base_dataframe(df, resolved_config)
    grouped = work.groupby(resolved_config.customer_col, sort=True)
    latest_invoice = work["Invoice Date"].max()

    agg_map = {
        resolved_config.item_col: ["count", pd.Series.nunique],
        "Invoice #": [pd.Series.nunique],
        "Order #": [pd.Series.nunique],
        "Invoiced qty (shipped)": ["sum", "mean", "median", "std", "max"],
        "Ordered qty": ["sum", "mean", "median", "std", "max"],
        "Invoiced price": ["sum", "mean", "median", "std", "max"],
        "Invoiced price (TX)": ["sum", "mean", "median", "std", "max"],
        "Cost of part": ["sum", "mean", "median", "std", "max"],
        "Material cost of part": ["sum", "mean", "median", "std"],
        "Labor cost of part": ["sum", "mean", "median", "std"],
        "Overhead cost of part": ["sum", "mean", "median", "std"],
        "GM%": ["mean", "median", "std", "min", "max"],
        "# of unique products on a quote": ["mean", "median", "max"],
        "price_gap": ["mean", "median", "std", "max"],
        "ship_ratio": ["mean", "median", "std", "min"],
        "ship_gap": ["mean", "median", "std", "max"],
        "margin_value": ["sum", "mean", "median", "std", "max"],
        "under_shipped_flag": ["mean", "sum"],
    }
    base = grouped.agg(agg_map)
    base.columns = [f"agg__{column}__{agg}" for column, agg in base.columns]

    activity = pd.DataFrame(index=base.index)
    activity["activity__row_count"] = grouped.size()
    activity["activity__unique_items"] = grouped[resolved_config.item_col].nunique(dropna=True)
    activity["activity__unique_product_groups"] = grouped[resolved_config.product_group_col].nunique(dropna=True)
    activity["activity__top_item_share"] = grouped[resolved_config.item_col].apply(top_share)
    activity["activity__product_group_entropy"] = grouped[resolved_config.product_group_col].apply(entropy_from_series)
    activity["activity__product_family_entropy"] = grouped[resolved_config.product_family_col].apply(entropy_from_series)
    activity["activity__internal_channel_entropy"] = grouped["Sales Channel - Internal"].apply(entropy_from_series)
    activity["activity__external_channel_entropy"] = grouped["Sales Channel - External"].apply(entropy_from_series)
    activity["activity__invoice_month_entropy"] = grouped["invoice_month"].apply(entropy_from_series)
    activity["activity__invoice_weekday_entropy"] = grouped["invoice_weekday"].apply(entropy_from_series)

    first_invoice = grouped["Invoice Date"].min()
    last_invoice = grouped["Invoice Date"].max()
    activity["time__invoice_recency_days"] = (latest_invoice - last_invoice).dt.days
    activity["time__invoice_tenure_days"] = (last_invoice - first_invoice).dt.days

    blocks: list[pd.DataFrame] = [base, activity]
    for column in FULL_SHARE_COLUMNS:
        block = build_full_share_block(work, customer_col=resolved_config.customer_col, value_col=column)
        if not block.empty:
            blocks.append(block)
    for column in TOP_SHARE_COLUMNS:
        block = build_top_value_share_block(
            work,
            customer_col=resolved_config.customer_col,
            value_col=column,
            top_n=resolved_config.top_n_channel_values,
            min_global_count=resolved_config.min_channel_count,
        )
        if not block.empty:
            blocks.append(block)

    bucket_features, bucket_map = build_product_group_bucket_features(
        work,
        customer_col=resolved_config.customer_col,
        product_group_col=resolved_config.product_group_col,
        revenue_col=resolved_config.revenue_col,
        bucket_size=resolved_config.bucket_size,
    )
    if not bucket_features.empty:
        blocks.append(bucket_features)

    hierarchy = build_family_hierarchy_features(
        work,
        customer_col=resolved_config.customer_col,
        product_family_col=resolved_config.product_family_col,
        product_group_col=resolved_config.product_group_col,
        revenue_col=resolved_config.revenue_col,
    )
    if not hierarchy.empty:
        blocks.append(hierarchy)

    raw_features = pd.concat(blocks, axis=1).sort_index()
    finalized_features, imputer = finalize_customer_feature_matrix(
        raw_features,
        max_missing_ratio=resolved_config.feature_max_missing_ratio,
        imputation_strategy=resolved_config.feature_imputation_strategy,
    )

    metadata = {
        "bucket_map": bucket_map,
        "feature_groups": describe_feature_groups(finalized_features),
        "raw_feature_columns": raw_features.columns.tolist(),
        "final_feature_columns": finalized_features.columns.tolist(),
        "dropped_feature_columns": [column for column in raw_features.columns if column not in finalized_features.columns],
        "imputation_strategy": resolved_config.feature_imputation_strategy,
        "feature_max_missing_ratio": resolved_config.feature_max_missing_ratio,
        "imputer_statistics": {
            column: float(value)
            for column, value in zip(finalized_features.columns.tolist(), imputer.statistics_.tolist())
        },
    }
    if return_metadata:
        return finalized_features, metadata
    return finalized_features


class CustomerFeatureMatrixBuilder:
    """Pipeline wrapper that exports a customer-level feature matrix."""

    input_type = ARTIFACT_DATAFRAME
    output_type = ARTIFACT_MATRIX
    input_artifacts = {
        "df": ArtifactSpec(name="df", kind=ARTIFACT_DATAFRAME, dense=True, description="Processed transaction dataframe.")
    }
    output_artifacts = {
        "customer_features": ArtifactSpec(
            name="customer_features",
            kind=ARTIFACT_MATRIX,
            dense=True,
            description="Customer-level engineered feature matrix.",
        )
    }
    contract = StageContract(
        input_type=input_type,
        output_type=output_type,
        input_artifacts=input_artifacts,
        output_artifacts=output_artifacts,
        dense=True,
        description="Rich customer feature matrix builder.",
    )

    def __init__(self, **kwargs: Any) -> None:
        # Inputs: feature-builder keyword arguments. Outputs: initialized builder with stored config.
        self.config = CustomerFeatureConfig(**kwargs)
        self.customer_features_: np.ndarray | None = None
        self.customer_index_: np.ndarray | None = None
        self.feature_columns_: list[str] = []
        self.metadata_: dict[str, Any] = {}

    def fit(self, df: pd.DataFrame) -> "CustomerFeatureMatrixBuilder":
        # Inputs: processed transaction dataframe. Outputs: fitted builder with stored dense feature matrix.
        feature_df, metadata = build_customer_feature_matrix(df, self.config, return_metadata=True)
        self.customer_index_ = feature_df.index.to_numpy()
        self.feature_columns_ = feature_df.columns.tolist()
        self.customer_features_ = feature_df.to_numpy(dtype=float)
        self.metadata_ = metadata
        return self

    def export_artifacts(self) -> dict[str, object]:
        # Inputs: none. Outputs: named pipeline artifact dictionary containing the customer feature matrix.
        if self.customer_features_ is None:
            raise RuntimeError("The feature builder must be fitted before exporting artifacts.")
        return {"customer_features": self.customer_features_}


__all__ = [
    "CustomerFeatureConfig",
    "CustomerFeatureMatrixBuilder",
    "build_customer_feature_matrix",
    "build_family_hierarchy_features",
    "build_product_group_bucket_features",
    "build_product_group_bucket_map",
    "build_top_value_share_block",
    "build_full_share_block",
    "describe_feature_groups",
    "read_lumen_csv",
    "schema_report",
]
