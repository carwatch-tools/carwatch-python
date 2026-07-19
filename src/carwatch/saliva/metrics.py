"""Compute established features from longitudinal saliva measurements."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy import integrate, stats

from carwatch.exceptions import SchemaError


@dataclass(frozen=True)
class _PreparedData:
    values: pd.DataFrame
    times: pd.DataFrame | None
    sample_labels: list
    all_sample_labels: list


def max_value(
    data: pd.DataFrame,
    saliva_type: str | Sequence[str] = "cortisol",
    remove_s0: bool = False,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Compute the maximum measured value for each sampling series."""
    if not isinstance(saliva_type, str):
        return {
            current: max_value(data, saliva_type=current, remove_s0=remove_s0)
            for current in saliva_type
        }
    prepared = _prepare_data(data, saliva_type=saliva_type, remove_s0=remove_s0)
    output = prepared.values.max(axis=1).to_frame(f"{saliva_type}_max_val")
    return _finalize(output)


def initial_value(
    data: pd.DataFrame,
    saliva_type: str | Sequence[str] = "cortisol",
    remove_s0: bool = False,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Return the first measured value for each sampling series."""
    if not isinstance(saliva_type, str):
        return {
            current: initial_value(data, saliva_type=current, remove_s0=remove_s0)
            for current in saliva_type
        }
    prepared = _prepare_data(data, saliva_type=saliva_type, remove_s0=remove_s0)
    output = prepared.values.iloc[:, 0].to_frame(f"{saliva_type}_ini_val")
    return _finalize(output)


def max_increase(
    data: pd.DataFrame,
    saliva_type: str | Sequence[str] = "cortisol",
    remove_s0: bool = False,
    percent: bool = False,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Compute the maximum increase from the first measured sample."""
    if not isinstance(saliva_type, str):
        return {
            current: max_increase(
                data,
                saliva_type=current,
                remove_s0=remove_s0,
                percent=percent,
            )
            for current in saliva_type
        }
    prepared = _prepare_data(data, saliva_type=saliva_type, remove_s0=remove_s0)
    increase = prepared.values.iloc[:, 1:].max(axis=1) - prepared.values.iloc[:, 0]
    suffix = "max_inc"
    if percent:
        with np.errstate(divide="ignore", invalid="ignore"):
            increase = 100 * increase / prepared.values.iloc[:, 0].abs()
        suffix = "max_inc_percent"
    return _finalize(increase.to_frame(f"{saliva_type}_{suffix}"))


def auc(
    data: pd.DataFrame,
    saliva_type: str | Sequence[str] = "cortisol",
    remove_s0: bool = False,
    compute_auc_post: bool = False,
    sample_times: np.ndarray | Sequence[float] | str | None = None,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    r"""Compute area under the curve with respect to ground and increase.

    The trapezoidal formulas follow Pruessner et al. (2003). Sampling times
    can be supplied explicitly, named by a dataframe column, or read from the
    default ``time`` column. Individual time vectors are supported for each
    participant and day.
    """
    if not isinstance(saliva_type, str):
        return {
            current: auc(
                data,
                saliva_type=current,
                remove_s0=remove_s0,
                compute_auc_post=compute_auc_post,
                sample_times=sample_times,
            )
            for current in saliva_type
        }
    prepared = _prepare_data(
        data,
        saliva_type=saliva_type,
        remove_s0=remove_s0,
        time_column=sample_times if isinstance(sample_times, str) else "time",
    )
    times = _resolve_sample_times(
        prepared, None if isinstance(sample_times, str) else sample_times
    )
    _validate_sample_times(times)

    rows: list[dict[str, float]] = []
    for values, current_times in zip(prepared.values.to_numpy(), times, strict=True):
        current = {
            "auc_g": _trapezoid_complete(values, current_times),
            "auc_i": _trapezoid_complete(values - values[0], current_times),
        }
        if compute_auc_post:
            mask = current_times >= 0
            if mask.sum() < 2:
                current["auc_i_post"] = float("nan")
            else:
                post_values = values[mask]
                current["auc_i_post"] = _trapezoid_complete(
                    post_values - post_values[0],
                    current_times[mask],
                )
        rows.append(current)
    output = pd.DataFrame(rows, index=prepared.values.index).add_prefix(
        f"{saliva_type}_"
    )
    return _finalize(output)


def slope(
    data: pd.DataFrame,
    sample_labels: tuple | Sequence | None = None,
    sample_idx: tuple[int, int] | Sequence[int] | None = None,
    saliva_type: str | Sequence[str] = "cortisol",
    sample_times: np.ndarray | Sequence[float] | str | None = None,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Compute the slope between two saliva samples."""
    if sample_idx is None and sample_labels is None:
        raise IndexError("Either 'sample_labels' or 'sample_idx' must be supplied.")
    if sample_idx is not None and sample_labels is not None:
        raise IndexError("Specify either 'sample_labels' or 'sample_idx', not both.")
    if not isinstance(saliva_type, str):
        return {
            current: slope(
                data,
                sample_labels=sample_labels,
                sample_idx=sample_idx,
                saliva_type=current,
                sample_times=sample_times,
            )
            for current in saliva_type
        }

    prepared = _prepare_data(
        data,
        saliva_type=saliva_type,
        time_column=sample_times if isinstance(sample_times, str) else "time",
    )
    indices, labels = _resolve_sample_pair(
        prepared.sample_labels, sample_labels, sample_idx
    )
    times = _resolve_sample_times(
        prepared, None if isinstance(sample_times, str) else sample_times
    )
    _validate_sample_times(times)
    values = prepared.values.to_numpy()[:, indices]
    selected_times = times[:, indices]
    denominator = selected_times[:, 1] - selected_times[:, 0]
    with np.errstate(divide="ignore", invalid="ignore"):
        result = (values[:, 1] - values[:, 0]) / denominator
    result[np.isnan(values).any(axis=1) | np.isnan(selected_times).any(axis=1)] = np.nan
    name = f"{saliva_type}_slope{labels[0]}{labels[1]}"
    return _finalize(pd.DataFrame({name: result}, index=prepared.values.index))


def standard_features(
    data: pd.DataFrame,
    saliva_type: str | Sequence[str] = "cortisol",
    group_cols: str | Sequence[str] | None = None,
    keep_index: bool = True,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Compute argument of maximum, mean, standard deviation, skew, and kurtosis."""
    if not isinstance(saliva_type, str):
        return {
            current: standard_features(
                data,
                saliva_type=current,
                group_cols=group_cols,
                keep_index=keep_index,
            )
            for current in saliva_type
        }
    _validate_data(data, saliva_type)
    if group_cols is not None:
        return _standard_features_grouped(
            data,
            saliva_type=saliva_type,
            group_cols=group_cols,
        )

    prepared = _prepare_data(data, saliva_type=saliva_type)
    values = prepared.values.to_numpy(dtype=float)
    output = pd.DataFrame(
        {
            f"{saliva_type}_argmax": [_nanargmax(row) for row in values],
            f"{saliva_type}_mean": np.nanmean(values, axis=1),
            f"{saliva_type}_std": np.nanstd(values, axis=1, ddof=1),
            f"{saliva_type}_skew": stats.skew(values, axis=1, nan_policy="omit"),
            f"{saliva_type}_kurt": stats.kurtosis(values, axis=1, nan_policy="omit"),
        },
        index=prepared.values.index if keep_index else None,
    )
    return _finalize(output)


def mean_se(
    data: pd.DataFrame,
    saliva_type: str | Sequence[str] = "cortisol",
    group_cols: str | Sequence[str] | None = None,
    remove_s0: bool = False,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Compute mean and standard error per saliva sample."""
    if not isinstance(saliva_type, str):
        return {
            current: mean_se(
                data,
                saliva_type=current,
                group_cols=group_cols,
                remove_s0=remove_s0,
            )
            for current in saliva_type
        }
    _validate_data(data, saliva_type)
    frame = data.reset_index()
    if remove_s0:
        frame = frame.loc[~frame["sample"].isin([0, "0", "S0"])]
    groups = _as_list(group_cols)
    groups.append("sample")
    if "time" in frame and "time" not in groups:
        unique_times = frame.groupby("sample", dropna=False)["time"].nunique(
            dropna=False
        )
        if unique_times.le(1).all():
            groups.append("time")
    grouped = frame.groupby(groups, sort=False, dropna=False)[saliva_type]
    output = pd.concat(
        [grouped.mean().rename("mean"), grouped.sem().rename("se")], axis=1
    )
    output.columns.name = "saliva_feature"
    return output


def compute_features(
    data: pd.DataFrame,
    saliva_type: str | Sequence[str] = "cortisol",
    *,
    sample_times: np.ndarray | Sequence[float] | str | None = None,
    slope_pairs: Sequence[tuple] | None = None,
    remove_s0: bool = False,
) -> pd.DataFrame | dict[str, pd.DataFrame]:
    """Compute the common saliva response feature set in one call.

    The default output contains AUCg, AUCi, initial value, maximum value,
    maximum increase, and the slope from the first to the last sample.
    """
    if not isinstance(saliva_type, str):
        return {
            current: compute_features(
                data,
                saliva_type=current,
                sample_times=sample_times,
                slope_pairs=slope_pairs,
                remove_s0=remove_s0,
            )
            for current in saliva_type
        }
    prepared = _prepare_data(data, saliva_type=saliva_type, remove_s0=remove_s0)
    pairs = (
        list(slope_pairs)
        if slope_pairs is not None
        else [(prepared.sample_labels[0], prepared.sample_labels[-1])]
    )
    features = [
        auc(
            data,
            saliva_type=saliva_type,
            remove_s0=remove_s0,
            sample_times=sample_times,
        ),
        initial_value(data, saliva_type=saliva_type, remove_s0=remove_s0),
        max_value(data, saliva_type=saliva_type, remove_s0=remove_s0),
        max_increase(data, saliva_type=saliva_type, remove_s0=remove_s0),
    ]
    features.extend(
        slope(
            _remove_s0(data) if remove_s0 else data,
            sample_labels=pair,
            saliva_type=saliva_type,
            sample_times=_remove_s0_times(sample_times, prepared)
            if remove_s0
            else sample_times,
        )
        for pair in pairs
    )
    return pd.concat(features, axis=1)


def _prepare_data(
    data: pd.DataFrame,
    *,
    saliva_type: str,
    remove_s0: bool = False,
    time_column: str = "time",
) -> _PreparedData:
    _validate_data(data, saliva_type)
    frame = data.reset_index()
    group_cols = [name for name in data.index.names if name != "sample"]
    all_labels = list(pd.unique(frame["sample"]))
    labels = [
        label for label in all_labels if not (remove_s0 and label in {0, "0", "S0"})
    ]
    if len(labels) < 1:
        raise SchemaError("No saliva samples remain after filtering.")
    frame = frame.loc[frame["sample"].isin(labels)]
    values = frame.pivot(
        index=group_cols, columns="sample", values=saliva_type
    ).reindex(columns=labels)
    values = values.apply(pd.to_numeric, errors="raise").astype(float)
    values.columns.name = "sample"
    times = None
    if time_column in frame:
        times = frame.pivot(
            index=group_cols, columns="sample", values=time_column
        ).reindex(columns=labels)
        times = times.apply(pd.to_numeric, errors="coerce")
    return _PreparedData(
        values=values, times=times, sample_labels=labels, all_sample_labels=all_labels
    )


def _resolve_sample_times(
    prepared: _PreparedData,
    sample_times: np.ndarray | Sequence[float] | None,
) -> np.ndarray:
    if sample_times is None:
        if prepared.times is None:
            raise ValueError(
                "No sample times specified and no 'time' column is available."
            )
        return prepared.times.to_numpy(dtype=float)

    times = np.asarray(sample_times, dtype=float).squeeze()
    selected_positions = [
        prepared.all_sample_labels.index(label) for label in prepared.sample_labels
    ]
    if times.ndim == 1:
        if times.shape[0] == len(prepared.all_sample_labels):
            times = times[selected_positions]
        elif times.shape[0] != len(prepared.sample_labels):
            raise ValueError(
                "One-dimensional sample times must match the number of samples."
            )
        return np.tile(times, (len(prepared.values), 1))
    if times.ndim == 2:
        if times.shape[0] != len(prepared.values):
            raise ValueError(
                "Individual sample times must have one row per sampling series."
            )
        if times.shape[1] == len(prepared.all_sample_labels):
            times = times[:, selected_positions]
        elif times.shape[1] != len(prepared.sample_labels):
            raise ValueError(
                "Individual sample times must match the number of samples."
            )
        return times
    raise ValueError("Sample times must be one- or two-dimensional.")


def _validate_sample_times(times: np.ndarray) -> None:
    for row in times:
        if np.isnan(row).any():
            continue
        if np.any(np.diff(row) <= 0):
            raise ValueError(
                "'sample_times' must be strictly increasing within each sampling series."
            )


def _trapezoid_complete(values: np.ndarray, times: np.ndarray) -> float:
    if np.isnan(values).any() or np.isnan(times).any():
        return float("nan")
    return float(integrate.trapezoid(values, times))


def _resolve_sample_pair(
    columns: Sequence,
    sample_labels: tuple | Sequence | None,
    sample_idx: tuple[int, int] | Sequence[int] | None,
) -> tuple[list[int], list]:
    if sample_labels is not None:
        labels = list(sample_labels)
        if len(labels) != 2:
            raise IndexError("Exactly two sample labels must be supplied.")
        missing = [label for label in labels if label not in columns]
        if missing:
            raise IndexError(f"Unknown sample labels: {missing}")
        return [columns.index(label) for label in labels], labels
    indices = list(sample_idx or [])
    if len(indices) != 2:
        raise IndexError("Exactly two sample indices must be supplied.")
    try:
        labels = [columns[index] for index in indices]
    except IndexError as exc:
        raise IndexError("Sample index is outside the available sample range.") from exc
    return indices, labels


def _standard_features_grouped(
    data: pd.DataFrame,
    *,
    saliva_type: str,
    group_cols: str | Sequence[str],
) -> pd.DataFrame:
    frame = data.reset_index()
    groups = _as_list(group_cols)
    missing = [column for column in groups if column not in frame]
    if missing:
        raise SchemaError(f"Unknown grouping columns: {missing}")
    grouped = frame.groupby(groups, sort=False, dropna=False)[saliva_type]
    output = grouped.agg([_nanargmax, "mean", "std", _skew, _kurtosis])
    output.columns = [
        f"{saliva_type}_argmax",
        f"{saliva_type}_mean",
        f"{saliva_type}_std",
        f"{saliva_type}_skew",
        f"{saliva_type}_kurt",
    ]
    return _finalize(output)


def _nanargmax(values) -> float:
    values = np.asarray(values, dtype=float)
    if np.isnan(values).all():
        return float("nan")
    return float(np.nanargmax(values))


def _skew(values) -> float:
    return float(stats.skew(values, nan_policy="omit"))


def _kurtosis(values) -> float:
    return float(stats.kurtosis(values, nan_policy="omit"))


def _remove_s0(data: pd.DataFrame) -> pd.DataFrame:
    return data.drop(index=[0, "0", "S0"], level="sample", errors="ignore")


def _remove_s0_times(sample_times, prepared: _PreparedData):
    if sample_times is None or isinstance(sample_times, str):
        return sample_times
    times = np.asarray(sample_times)
    if times.shape[-1] == len(prepared.all_sample_labels):
        positions = [
            prepared.all_sample_labels.index(label) for label in prepared.sample_labels
        ]
        return times[..., positions]
    return sample_times


def _as_list(value: str | Sequence[str] | None) -> list[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


def _validate_data(data: pd.DataFrame, saliva_type: str) -> None:
    if not isinstance(data, pd.DataFrame):
        raise TypeError("Saliva data must be a pandas DataFrame.")
    if "sample" not in data.index.names:
        raise SchemaError("Saliva data require a 'sample' index level.")
    if saliva_type not in data:
        raise SchemaError(
            f"Saliva data are missing measurement column: {saliva_type!r}"
        )
    if data.index.has_duplicates:
        raise SchemaError("Saliva data contain duplicate sample rows.")
    if len(data.index.names) < 2:
        raise SchemaError(
            "Saliva data require at least one grouping index in addition to 'sample'."
        )


def _finalize(data: pd.DataFrame) -> pd.DataFrame:
    data.columns.name = "saliva_feature"
    return data
