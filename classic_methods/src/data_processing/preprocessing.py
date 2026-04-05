"""Configurable dataframe preprocessor for recommendation experiments.

The preprocessor is designed around a small number of modular stages:
- identifier cleanup and item-code indexing
- customer removal based on per-column null ratios
- row-level price consistency filtering
- optional customer and item frequency filters

It returns the cleaned dataframe together with an ``item_idx -> item code``
mapping so the downstream embedding / clustering code can work with stable
integer ids.
"""

from __future__ import annotations

import logging
from pathlib import Path
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Sequence

import pandas as pd
from .imputers import imputer
logger = logging.getLogger(__name__)

try:
    from ..pipeline_contracts import (
        ARTIFACT_DATAFRAME,
        ARTIFACT_MAPPING,
        ArtifactSpec,
        StageContract,
    )
except ImportError:  # pragma: no cover - supports direct imports from src/
    from pipeline_contracts import (  # type: ignore
        ARTIFACT_DATAFRAME,
        ARTIFACT_MAPPING,
        ArtifactSpec,
        StageContract,
    )


@dataclass(frozen=True)
class NullRemovalRule:
    """Remove a customer if any listed column exceeds the null threshold."""

    columns: tuple[str, ...]
    max_null_fraction: float
    name: str | None = None


@dataclass(frozen=True)
class PreprocessorConfig:
    """Configuration for modular preprocessing."""

    csv_path: str | None = None
    customer_col: str = "CustomerID"
    item_col: str = "Item Code"
    date_columns: tuple[str, ...] = ("Order Date",)
    price_col: str = "Invoiced price"
    tx_price_col: str = "Invoiced price (TX)"
    item_idx_col: str = "item_idx"
    null_rules: tuple[NullRemovalRule, ...] = field(default_factory=tuple)
    min_num_purchases_per_customer: int | None = None
    min_num_items_per_customer: int | None = None
    min_num_purchases_per_item_rows: int | None = None
    min_num_unique_customers_per_item: int | None = None
    map_item_codes: bool = True
    drop_price_gt_tx: bool = True
    drop_missing_customer_id: bool = True
    drop_missing_item_code: bool = True
    reindex_items_at_end: bool = True
    impute: bool = True


def _ensure_tuple(value: Sequence[str] | str | None) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    return tuple(value)


def _normalize_null_rule(rule: NullRemovalRule | Mapping[str, Any], index: int) -> NullRemovalRule:
    if isinstance(rule, NullRemovalRule):
        columns = _ensure_tuple(rule.columns)
        name = rule.name or f"null_rule_{index + 1}"
        return NullRemovalRule(columns=columns, max_null_fraction=float(rule.max_null_fraction), name=name)

    columns = rule.get("columns", rule.get("column"))
    if columns is None:
        raise ValueError("NullRemovalRule requires 'columns' or 'column'.")
    max_null_fraction = rule.get("max_null_fraction", rule.get("threshold"))
    if max_null_fraction is None:
        raise ValueError("NullRemovalRule requires 'max_null_fraction' or 'threshold'.")
    name = rule.get("name") or f"null_rule_{index + 1}"
    return NullRemovalRule(
        columns=_ensure_tuple(columns),
        max_null_fraction=float(max_null_fraction),
        name=str(name),
    )


def _normalize_config(config: PreprocessorConfig) -> PreprocessorConfig:
    null_rules = tuple(
        _normalize_null_rule(rule, index)
        for index, rule in enumerate(tuple(config.null_rules or ()))
    )
    return replace(config, null_rules=null_rules)


@dataclass
class StageResult:
    """Counts for a single preprocessing stage."""

    customers_removed: int = 0
    rows_removed: int = 0
    customers_before: int | None = None
    customers_after: int | None = None
    customers_lost: int = 0
    rows_before: int | None = None
    rows_after: int | None = None
    details: dict[str, Any] = field(default_factory=dict)


class Preprocessor:
    """Modular dataframe preprocessor for the classic methods pipeline."""

    output_type = ARTIFACT_DATAFRAME
    output_artifacts = {
        "dataframe": ArtifactSpec(
            name="dataframe",
            kind=ARTIFACT_DATAFRAME,
            dense=True,
            description="Preprocessed dataframe.",
        ),
        "idx2item": ArtifactSpec(
            name="idx2item",
            kind=ARTIFACT_MAPPING,
            dense=True,
            description="Mapping from item indices back to original item codes.",
        ),
    }

    def __init__(self, config: PreprocessorConfig | None = None, **kwargs: Any) -> None:
        if config is None:
            config = PreprocessorConfig(**kwargs)
        elif kwargs:
            config = replace(config, **kwargs)
        self.config = _normalize_config(config)
        self.input_type = ARTIFACT_DATAFRAME if self.config.csv_path is None else None
        self.input_artifacts = (
            {
                "df": ArtifactSpec(
                    name="df",
                    kind=ARTIFACT_DATAFRAME,
                    dense=True,
                    description="Raw dataframe loaded from the source CSV file.",
                )
            }
            if self.config.csv_path is None
            else {}
        )
        self.contract = StageContract(
            input_type=self.input_type,
            output_type=self.output_type,
            input_artifacts=self.input_artifacts,
            output_artifacts=self.output_artifacts,
            dense=True,
            description="Configurable dataframe preprocessing stage.",
        )
        self._report: dict[str, Any] = {}
        self.item2idx_: dict[Any, int] = {}
        self.idx2item_: dict[int, Any] = {}
        self.output_dataframe_: pd.DataFrame | None = None
        self.output_idx2item_: dict[int, Any] = {}

    @classmethod
    def from_rules(
        cls,
        *,
        null_rules: Sequence[NullRemovalRule | Mapping[str, Any]] | None = None,
        **kwargs: Any,
    ) -> "Preprocessor":
        """Convenience constructor for defining only the active stages."""

        config = PreprocessorConfig(
            null_rules=tuple(
                _normalize_null_rule(rule, index) for index, rule in enumerate(null_rules or ())
            ),
            **kwargs,
        )
        return cls(config=config)

    def fit_transform(
        self,
        df: pd.DataFrame | None = None,
        impute: bool | None = None,
    ) -> tuple[pd.DataFrame, dict[int, Any]]:
        """Run all enabled preprocessing stages and return the cleaned frame."""

        work = self._resolve_input_dataframe(df)
        should_impute = self.config.impute if impute is None else bool(impute)
        self._report = {
            "initial_rows": int(work.shape[0]),
            "initial_customers": int(work[self.config.customer_col].nunique(dropna=True))
            if self.config.customer_col in work.columns
            else None,
            "initial_items": int(work[self.config.item_col].nunique(dropna=True))
            if self.config.item_col in work.columns
            else None,
            "stages": {},
        }

        work, missing_id_result = self._drop_missing_identifier_rows(work)
        self._store_stage("missing_identifier_rows", missing_id_result)

        work, date_result = self._parse_date_columns(work)
        self._store_stage("date_parsing", date_result)

        work, item_map_result = self._prepare_item_indices(work)
        self._store_stage("item_mapping", item_map_result)

        work, null_rule_result = self._remove_customers_by_null_rules(work)
        self._store_stage("customer_null_rules", null_rule_result)

        work, price_result = self._remove_rows_where_price_exceeds_tx(work)
        self._store_stage("price_filter", price_result)

        work, item_result = self._apply_item_frequency_filters(work)
        self._store_stage("item_frequency_filters", item_result)

        work, customer_result = self._apply_customer_frequency_filters(work)
        self._store_stage("customer_frequency_filters", customer_result)

        if self.config.reindex_items_at_end:
            work, remap_result = self._reindex_item_codes(work)
            self._store_stage("final_item_reindex", remap_result)

        if should_impute:
            work = imputer(
                dataframe=work,
                per_customer=True,
                categorical_unknown=True,
                customer_col=self.config.customer_col,
            )

        self.output_dataframe_ = work.copy()
        self.output_idx2item_ = dict(self.idx2item_)

        self._report["final_rows"] = int(work.shape[0])
        self._report["final_customers"] = int(work[self.config.customer_col].nunique(dropna=True)) if not work.empty else 0
        self._report["final_items"] = int(work[self.config.item_col].nunique(dropna=True)) if not work.empty else 0

        self._log_summary()
        return work, dict(self.idx2item_)

    def preprocess(self, df: pd.DataFrame | None = None) -> tuple[pd.DataFrame, dict[int, Any]]:
        """Alias for :meth:`fit_transform`."""

        return self.fit_transform(df)

    def export_artifacts(self) -> dict[str, object]:
        """Export the fitted dataframe and the reverse item-code mapping."""

        if self.output_dataframe_ is None:
            raise RuntimeError("The preprocessor must be fitted before exporting artifacts.")
        return {
            "dataframe": self.output_dataframe_,
            "idx2item": dict(self.output_idx2item_),
        }

    def _store_stage(self, stage_name: str, result: StageResult) -> None:
        self._report["stages"][stage_name] = {
            "customers_removed": result.customers_removed,
            "customers_before": result.customers_before,
            "customers_after": result.customers_after,
            "customers_lost": result.customers_lost,
            "rows_before": result.rows_before,
            "rows_after": result.rows_after,
            "rows_removed": result.rows_removed,
            "details": result.details,
        }
        if result.rows_before is not None and result.rows_after is not None:
            logger.info(
                "Stage '%s': rows %d -> %d (-%d), customers %d -> %d (-%d, direct=%d).",
                stage_name,
                result.rows_before,
                result.rows_after,
                result.rows_removed,
                result.customers_before if result.customers_before is not None else -1,
                result.customers_after if result.customers_after is not None else -1,
                result.customers_lost,
                result.customers_removed,
            )

    def report(self) -> dict[str, Any]:
        """Print the preprocessing report and return the raw report dict."""

        print("\n--- Preprocessing report ---")
        for stage_name, stage_info in self._report.get("stages", {}).items():
            print(f"\n[{stage_name}]")
            print(
                f"rows: {stage_info['rows_before']} -> {stage_info['rows_after']} "
                f"(-{stage_info['rows_removed']})\n"
                f"customers: {stage_info['customers_before']} -> {stage_info['customers_after']} "
                f"(-{stage_info['customers_lost']}, direct={stage_info['customers_removed']})"
            )
            print(stage_info["details"])

        print("\n--- Summary ---")
        print(
            f"rows: {self._report.get('initial_rows')} -> {self._report.get('final_rows')}\n"
            f"customers: {self._report.get('initial_customers')} -> {self._report.get('final_customers')}\n"
            f"items: {self._report.get('initial_items')} -> {self._report.get('final_items')}"
        )
        return self._report

    @property
    def report_data(self) -> dict[str, Any]:
        """Access the raw report dictionary without printing it."""

        return self._report

    def _resolve_input_dataframe(self, df: pd.DataFrame | None) -> pd.DataFrame:
        if df is not None:
            return df.copy()
        if self.config.csv_path is None:
            raise ValueError("Preprocessor requires either a dataframe input or config.csv_path.")
        return self._read_dataframe(Path(self.config.csv_path))

    def _read_dataframe(self, csv_path: Path) -> pd.DataFrame:
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
                return pd.read_csv(csv_path, **kwargs)
            except (UnicodeDecodeError, pd.errors.ParserError) as exc:
                last_error = exc
        if last_error is not None:
            raise last_error
        return pd.read_csv(csv_path)

    def _drop_missing_identifier_rows(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StageResult]:
        result = StageResult()
        result.rows_before = int(df.shape[0])
        result.customers_before = int(df[self.config.customer_col].nunique(dropna=True))

        if self.config.customer_col not in df.columns:
            raise KeyError(f"Missing required customer column: {self.config.customer_col}")
        if self.config.item_col not in df.columns:
            raise KeyError(f"Missing required item column: {self.config.item_col}")

        mask = pd.Series(False, index=df.index)
        if self.config.drop_missing_customer_id:
            missing_customer = df[self.config.customer_col].isna()
            result.details["missing_customer_rows"] = int(missing_customer.sum())
            mask = mask | missing_customer
        if self.config.drop_missing_item_code:
            missing_item = df[self.config.item_col].isna()
            result.details["missing_item_rows"] = int(missing_item.sum())
            mask = mask | missing_item

        result.rows_removed = int(mask.sum())
        result.rows_after = int((~mask).sum())
        result.customers_after = int(df.loc[~mask, self.config.customer_col].nunique(dropna=True))
        result.customers_lost = result.customers_before - result.customers_after
        if result.rows_removed:
            logger.info(
                "Dropped %d rows with missing identifiers (%s).",
                result.rows_removed,
                result.details,
            )
        return df.loc[~mask].copy(), result

    def _map_item_codes(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StageResult]:
        result = StageResult()
        result.rows_before = int(df.shape[0])
        result.rows_after = int(df.shape[0])
        result.customers_before = int(df[self.config.customer_col].nunique(dropna=True))
        result.customers_after = result.customers_before

        if df.empty:
            df[self.config.item_idx_col] = pd.Series(dtype="int64")
            self.item2idx_ = {}
            self.idx2item_ = {}
            return df, result

        unique_items = pd.unique(df[self.config.item_col].dropna())
        self.item2idx_ = {item_code: idx for idx, item_code in enumerate(unique_items)}
        self.idx2item_ = {idx: item_code for item_code, idx in self.item2idx_.items()}
        df = df.copy()
        df[self.config.item_idx_col] = df[self.config.item_col].map(self.item2idx_)

        result.details["unique_items"] = len(self.idx2item_)
        logger.info("Mapped %d unique item codes to item indices.", len(self.idx2item_))
        return df, result

    def _prepare_item_indices(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StageResult]:
        if self.config.map_item_codes:
            return self._map_item_codes(df)

        result = StageResult()
        result.rows_before = int(df.shape[0])
        result.rows_after = int(df.shape[0])
        result.customers_before = int(df[self.config.customer_col].nunique(dropna=True))
        result.customers_after = result.customers_before

        if self.config.item_idx_col not in df.columns:
            raise KeyError(
                f"Missing required item index column '{self.config.item_idx_col}' while map_item_codes is disabled."
            )
        if self.config.item_col not in df.columns:
            raise KeyError(
                f"Missing required item column '{self.config.item_col}' while map_item_codes is disabled."
            )

        work = df.copy()
        item_idx_series = pd.to_numeric(work[self.config.item_idx_col], errors="raise").astype(int)
        work[self.config.item_idx_col] = item_idx_series

        unique_pairs = (
            work[[self.config.item_idx_col, self.config.item_col]]
            .drop_duplicates()
            .sort_values(self.config.item_idx_col, kind="stable")
        )
        self.idx2item_ = {
            int(row[self.config.item_idx_col]): row[self.config.item_col]
            for _, row in unique_pairs.iterrows()
        }
        self.item2idx_ = {item_code: item_idx for item_idx, item_code in self.idx2item_.items()}
        result.details["preserved_existing_item_idx"] = True
        result.details["unique_items"] = len(self.idx2item_)
        logger.info("Preserved %d existing item indices.", len(self.idx2item_))
        return work, result

    def _parse_date_columns(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StageResult]:
        result = StageResult()
        if df.empty or not self.config.date_columns:
            return df, result

        df = df.copy()
        result.rows_before = int(df.shape[0])
        result.rows_after = int(df.shape[0])
        result.customers_before = int(df[self.config.customer_col].nunique(dropna=True))
        result.customers_after = result.customers_before
        parsed_details: dict[str, dict[str, int]] = {}

        for column in self.config.date_columns:
            if column not in df.columns:
                raise KeyError(f"Missing date column: {column}")

            raw_values = df[column].astype("string")
            sentinel_mask = raw_values.str.startswith("9999-", na=False)
            parsed = pd.to_datetime(raw_values, errors="coerce")
            parsed = parsed.mask(sentinel_mask, pd.NaT)

            df[column] = parsed
            parsed_details[column] = {
                "rows_converted_to_datetime": int(parsed.notna().sum()),
                "rows_set_to_null_due_to_9999_prefix": int(sentinel_mask.sum()),
                "rows_total_null_after_parsing": int(parsed.isna().sum()),
            }

        result.details = parsed_details
        logger.info("Parsed date columns: %s", parsed_details)
        return df, result

    def _remove_customers_by_null_rules(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StageResult]:
        result = StageResult()
        result.rows_before = int(df.shape[0])
        result.customers_before = int(df[self.config.customer_col].nunique(dropna=True))
        if not self.config.null_rules or df.empty:
            result.rows_after = result.rows_before
            result.customers_after = result.customers_before
            return df, result

        grouped_sizes = df.groupby(self.config.customer_col, sort=False).size()
        null_fraction_cache: dict[str, pd.Series] = {}
        customers_to_remove: set[Any] = set()

        for rule in self.config.null_rules:
            if not rule.columns:
                continue

            rule_mask = pd.Series(False, index=grouped_sizes.index)
            column_details: dict[str, float] = {}
            for column in rule.columns:
                if column not in df.columns:
                    raise KeyError(f"Missing column '{column}' required by null rule '{rule.name}'.")
                if column not in null_fraction_cache:
                    null_fraction_cache[column] = (
                        df.groupby(self.config.customer_col, sort=False)[column]
                        .apply(lambda series: series.isna().mean())
                        .reindex(grouped_sizes.index, fill_value=0.0)
                    )
                fractions = null_fraction_cache[column]
                column_details[column] = float(fractions.max()) if not fractions.empty else 0.0
                rule_mask = rule_mask | (fractions > rule.max_null_fraction)

            flagged_customers = rule_mask[rule_mask].index.tolist()
            flagged_rows = int(grouped_sizes.reindex(flagged_customers, fill_value=0).sum())
            result.details[rule.name or ",".join(rule.columns)] = {
                "columns": list(rule.columns),
                "max_null_fraction": rule.max_null_fraction,
                "customers_removed": len(flagged_customers),
                "rows_removed": flagged_rows,
                "max_fraction_seen_by_column": column_details,
            }
            customers_to_remove.update(flagged_customers)

        if customers_to_remove:
            mask = df[self.config.customer_col].isin(customers_to_remove)
            result.customers_removed = len(customers_to_remove)
            result.rows_removed = int(mask.sum())
            result.rows_after = int((~mask).sum())
            result.customers_after = int(df.loc[~mask, self.config.customer_col].nunique(dropna=True))
            result.customers_lost = result.customers_before - result.customers_after
            logger.info(
                "Removed %d customers and %d rows using null-ratio rules.",
                result.customers_removed,
                result.rows_removed,
            )
            for name, details in result.details.items():
                logger.info(
                    "Null rule '%s' removed %d customers and %d rows.",
                    name,
                    details["customers_removed"],
                    details["rows_removed"],
                )
            return df.loc[~mask].copy(), result

        logger.info("Null-ratio rules removed no customers.")
        result.rows_after = result.rows_before
        result.customers_after = result.customers_before
        return df, result

    def _remove_rows_where_price_exceeds_tx(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StageResult]:
        result = StageResult()
        result.rows_before = int(df.shape[0])
        result.customers_before = int(df[self.config.customer_col].nunique(dropna=True))
        if not self.config.drop_price_gt_tx or df.empty:
            result.rows_after = result.rows_before
            result.customers_after = result.customers_before
            return df, result

        for column in (self.config.price_col, self.config.tx_price_col):
            if column not in df.columns:
                raise KeyError(f"Missing required price column: {column}")

        price = pd.to_numeric(df[self.config.price_col], errors="coerce")
        tx_price = pd.to_numeric(df[self.config.tx_price_col], errors="coerce")
        mask = price.notna() & tx_price.notna() & (price > tx_price)

        result.rows_removed = int(mask.sum())
        result.rows_after = int((~mask).sum())
        result.customers_after = int(df.loc[~mask, self.config.customer_col].nunique(dropna=True))
        result.customers_lost = result.customers_before - result.customers_after
        result.details = {
            "price_col": self.config.price_col,
            "tx_price_col": self.config.tx_price_col,
        }

        if result.rows_removed:
            logger.info(
                "Removed %d rows where %s > %s.",
                result.rows_removed,
                self.config.price_col,
                self.config.tx_price_col,
            )
        return df.loc[~mask].copy(), result

    def _apply_item_frequency_filters(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StageResult]:
        result = StageResult()
        result.rows_before = int(df.shape[0])
        result.customers_before = int(df[self.config.customer_col].nunique(dropna=True))
        if df.empty:
            result.rows_after = 0
            result.customers_after = 0
            return df, result

        remove_items: set[Any] = set()

        if self.config.min_num_purchases_per_item_rows is not None:
            item_rows = df.groupby(self.config.item_col, sort=False).size()
            bad_items = item_rows[item_rows < self.config.min_num_purchases_per_item_rows].index.tolist()
            remove_items.update(bad_items)
            result.details["min_num_purchases_per_item_rows"] = {
                "threshold": self.config.min_num_purchases_per_item_rows,
                "items_removed": len(bad_items),
                "rows_removed": int(item_rows.reindex(bad_items, fill_value=0).sum()),
            }

        if self.config.min_num_unique_customers_per_item is not None:
            item_customers = df.groupby(self.config.item_col, sort=False)[self.config.customer_col].nunique(dropna=True)
            bad_items = item_customers[item_customers < self.config.min_num_unique_customers_per_item].index.tolist()
            remove_items.update(bad_items)
            result.details["min_num_unique_customers_per_item"] = {
                "threshold": self.config.min_num_unique_customers_per_item,
                "items_removed": len(bad_items),
                "rows_removed": int(
                    df[df[self.config.item_col].isin(bad_items)].shape[0]
                ),
            }

        if remove_items:
            mask = df[self.config.item_col].isin(remove_items)
            result.rows_removed = int(mask.sum())
            result.rows_after = int((~mask).sum())
            result.customers_after = int(df.loc[~mask, self.config.customer_col].nunique(dropna=True))
            result.customers_lost = result.customers_before - result.customers_after
            result.details["union_items_removed"] = len(remove_items)
            logger.info(
                "Removed %d items and %d rows using item-frequency filters.",
                len(remove_items),
                result.rows_removed,
            )
            return df.loc[~mask].copy(), result

        logger.info("Item-frequency filters removed no rows.")
        result.rows_after = result.rows_before
        result.customers_after = result.customers_before
        return df, result

    def _apply_customer_frequency_filters(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StageResult]:
        result = StageResult()
        result.rows_before = int(df.shape[0])
        result.customers_before = int(df[self.config.customer_col].nunique(dropna=True))
        if df.empty:
            result.rows_after = 0
            result.customers_after = 0
            return df, result

        remove_customers: set[Any] = set()

        if self.config.min_num_purchases_per_customer is not None:
            customer_rows = df.groupby(self.config.customer_col, sort=False).size()
            bad_customers = customer_rows[
                customer_rows < self.config.min_num_purchases_per_customer
            ].index.tolist()
            remove_customers.update(bad_customers)
            result.details["min_num_purchases_per_customer"] = {
                "threshold": self.config.min_num_purchases_per_customer,
                "customers_removed": len(bad_customers),
                "rows_removed": int(customer_rows.reindex(bad_customers, fill_value=0).sum()),
            }

        if self.config.min_num_items_per_customer is not None:
            customer_items = df.groupby(self.config.customer_col, sort=False)[self.config.item_col].nunique(dropna=True)
            bad_customers = customer_items[
                customer_items < self.config.min_num_items_per_customer
            ].index.tolist()
            remove_customers.update(bad_customers)
            result.details["min_num_items_per_customer"] = {
                "threshold": self.config.min_num_items_per_customer,
                "customers_removed": len(bad_customers),
                "rows_removed": int(df[df[self.config.customer_col].isin(bad_customers)].shape[0]),
            }

        if remove_customers:
            mask = df[self.config.customer_col].isin(remove_customers)
            result.customers_removed = len(remove_customers)
            result.rows_removed = int(mask.sum())
            result.rows_after = int((~mask).sum())
            result.customers_after = int(df.loc[~mask, self.config.customer_col].nunique(dropna=True))
            result.customers_lost = result.customers_before - result.customers_after
            result.details["union_customers_removed"] = len(remove_customers)
            logger.info(
                "Removed %d customers and %d rows using customer-frequency filters.",
                len(remove_customers),
                result.rows_removed,
            )
            return df.loc[~mask].copy(), result

        logger.info("Customer-frequency filters removed no rows.")
        result.rows_after = result.rows_before
        result.customers_after = result.customers_before
        return df, result

    def _reindex_item_codes(self, df: pd.DataFrame) -> tuple[pd.DataFrame, StageResult]:
        result = StageResult()
        result.rows_before = int(df.shape[0])
        result.rows_after = int(df.shape[0])
        result.customers_before = int(df[self.config.customer_col].nunique(dropna=True))
        result.customers_after = result.customers_before
        if df.empty:
            self.item2idx_ = {}
            self.idx2item_ = {}
            df = df.copy()
            df[self.config.item_idx_col] = pd.Series(dtype="int64")
            return df, result

        unique_items = pd.unique(df[self.config.item_col].dropna())
        self.item2idx_ = {item_code: idx for idx, item_code in enumerate(unique_items)}
        self.idx2item_ = {idx: item_code for item_code, idx in self.item2idx_.items()}
        df = df.copy()
        df[self.config.item_idx_col] = df[self.config.item_col].map(self.item2idx_)

        result.details["final_unique_items"] = len(self.idx2item_)
        logger.info("Reindexed %d remaining item codes.", len(self.idx2item_))
        return df, result

    def _log_summary(self) -> None:
        logger.info(
            "Preprocessing complete: %d -> %d rows, %d -> %d customers, %d -> %d items.",
            self._report.get("initial_rows", 0),
            self._report.get("final_rows", 0),
            self._report.get("initial_customers", 0),
            self._report.get("final_customers", 0),
            self._report.get("initial_items", 0),
            self._report.get("final_items", 0),
        )


__all__ = [
    "NullRemovalRule",
    "Preprocessor",
    "PreprocessorConfig",
]
