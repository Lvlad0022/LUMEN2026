"""Imputation helpers for dataframe preprocessing."""

from __future__ import annotations

from typing import Any

import pandas as pd
from pandas.api.types import (
    is_bool_dtype,
    is_datetime64_any_dtype,
    is_numeric_dtype,
)


def _first_mode(series: pd.Series) -> Any:
    values = series.dropna()
    if values.empty:
        return None
    modes = values.mode(dropna=True)
    if modes.empty:
        return None
    return modes.iloc[0]


def _datetime_mean(series: pd.Series) -> Any:
    values = pd.to_datetime(series.dropna(), errors="coerce")
    values = values.dropna()
    if values.empty:
        return pd.NaT
    as_datetime64 = values.to_numpy(dtype="datetime64[ns]")
    mean_ns = int(as_datetime64.view("int64").mean())
    return pd.to_datetime(mean_ns)


def _infer_categorical_fill(series: pd.Series, categorical_unknown: bool) -> Any:
    if categorical_unknown:
        return "UNKNOWN"
    mode_value = _first_mode(series)
    return "UNKNOWN" if mode_value is None else mode_value


def _looks_like_datetime_column(column: str) -> bool:
    lowered = column.lower()
    return "date" in lowered or "time" in lowered


def imputer(
    dataframe: pd.DataFrame,
    per_customer: bool,
    categorical_unknown: bool,
    customer_col: str = "CustomerID",
) -> pd.DataFrame:
    """Impute missing values in a dataframe.

    Parameters
    ----------
    dataframe:
        Input dataframe.
    per_customer:
        If True, use per-customer statistics with global fallback.
    categorical_unknown:
        If True, categorical columns are filled with ``"UNKNOWN"``.
        If False, categorical columns are filled with the mode.
    customer_col:
        Customer identifier column used for grouping when ``per_customer=True``.
    """

    if customer_col not in dataframe.columns:
        raise KeyError(f"Missing customer column: {customer_col}")

    df = dataframe.copy()
    global_customer_mode = _first_mode(df[customer_col])
    global_customer_fill = "UNKNOWN" if global_customer_mode is None else global_customer_mode

    for column in df.columns:
        if column == customer_col:
            df[column] = df[column].fillna(global_customer_fill)
            continue

        series = df[column]

        if is_datetime64_any_dtype(series) or (_looks_like_datetime_column(column) and not is_numeric_dtype(series)):
            parsed = pd.to_datetime(series, errors="coerce")
            global_fill = _datetime_mean(parsed)
            if per_customer:
                customer_fill = (
                    df.groupby(customer_col, sort=False)[column]
                    .transform(lambda s: _datetime_mean(pd.to_datetime(s, errors="coerce")))
                )
                df[column] = parsed.fillna(customer_fill).fillna(global_fill)
            else:
                df[column] = parsed.fillna(global_fill)
            continue

        if is_numeric_dtype(series) and not is_bool_dtype(series):
            global_fill = series.mean(skipna=True)
            if pd.isna(global_fill):
                global_fill = 0.0
            if per_customer:
                customer_fill = df.groupby(customer_col, sort=False)[column].transform("mean")
                df[column] = series.fillna(customer_fill).fillna(global_fill)
            else:
                df[column] = series.fillna(global_fill)
            continue

        if per_customer and not categorical_unknown:
            per_customer_mode = df.groupby(customer_col, sort=False)[column].transform(_first_mode)
            global_fill = _infer_categorical_fill(series, categorical_unknown=False)
            df[column] = series.fillna(per_customer_mode).fillna(global_fill)
        else:
            df[column] = series.fillna(_infer_categorical_fill(series, categorical_unknown))

    return df


__all__ = ["imputer"]
