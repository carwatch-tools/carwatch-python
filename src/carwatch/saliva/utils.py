"""Utilities for CARWatch saliva dataframes."""

from __future__ import annotations

from datetime import datetime, time

import numpy as np
import pandas as pd

from carwatch.exceptions import SchemaError

__all__ = ["saliva_feature_wide_to_long", "sample_times_datetime_to_minute"]


def saliva_feature_wide_to_long(data: pd.DataFrame, saliva_type: str) -> pd.DataFrame:
    """Convert wide saliva features into a feature-indexed long dataframe."""
    prefix = f"{saliva_type}_"
    columns = [column for column in data.columns if str(column).startswith(prefix)]
    if not columns:
        raise SchemaError(f"No feature columns found for saliva type {saliva_type!r}.")
    selected = data[columns].rename(
        columns=lambda column: str(column).removeprefix(prefix)
    )
    selected.columns.name = "saliva_feature"
    return selected.stack().to_frame(saliva_type)


def sample_times_datetime_to_minute(
    sample_times: pd.Series | pd.DataFrame,
) -> pd.Series | pd.DataFrame:
    """Convert datetime-like sampling times to minutes relative to the first sample."""
    if not isinstance(sample_times, pd.Series | pd.DataFrame):
        raise TypeError("Sample times must be a pandas Series or DataFrame.")
    is_series = isinstance(sample_times, pd.Series)
    if is_series:
        sample_indices = [
            name
            for name in ("scheduled_sample", "sample")
            if name in sample_times.index.names
        ]
        if len(sample_indices) != 1:
            raise SchemaError(
                "Long-format sample times require exactly one 'scheduled_sample' "
                "or 'sample' index level."
            )
        wide = sample_times.unstack(sample_indices[0])
    else:
        wide = sample_times.copy()
    if wide.empty:
        return sample_times.copy()

    first = wide.to_numpy().ravel()[0]
    if isinstance(first, str | time):
        converted = wide.apply(lambda column: pd.to_timedelta(column.astype(str)))
    elif isinstance(first, datetime | pd.Timestamp | np.datetime64):
        converted = wide.apply(pd.to_datetime)
    elif isinstance(first, pd.Timedelta | np.timedelta64):
        converted = wide.apply(pd.to_timedelta)
    else:
        raise TypeError(
            "Sample times must contain datetime-, time-, or timedelta-like values."
        )

    relative = converted.sub(converted.iloc[:, 0], axis=0)
    relative = relative.apply(lambda column: column.dt.total_seconds() / 60)
    if is_series:
        return relative.stack(dropna=False)
    return relative
