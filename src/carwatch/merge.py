"""Merge saliva measurements with CARWatch study results."""

from __future__ import annotations

import math
import re
from collections.abc import Mapping
from datetime import datetime, time
from numbers import Real
from typing import Literal

import pandas as pd
from pandas.api.types import is_list_like, is_numeric_dtype

import carwatch.exceptions as exceptions
import carwatch.logs as logs

_SUMMARY_INDEX = ["participant", "day", "scheduled_sample"]
_SALIVA_INDEX = ["participant", "sample"]
_MISSING_CARWATCH_DATA_OPTIONS = {"ignore", "raise", "impute"}
_ABSOLUTE_TIME_PATTERN = re.compile(
    r"^(?P<hour>[01]\d|2[0-3]):(?P<minute>[0-5]\d)(?::(?P<second>[0-5]\d))?$"
)
_PROVENANCE_COLUMNS = {
    "lab_value_available",
    "mismatch_corrected",
    "sampling_event_recorded",
    "sampling_time_imputed",
}


def merge_saliva(
    study_results: pd.DataFrame,
    saliva: pd.DataFrame,
    *,
    correct_swaps: bool = True,
    missing_carwatch_data: Literal["ignore", "raise", "impute"] = "ignore",
    sampling_schedule=None,
) -> pd.DataFrame:
    """Merge laboratory measurements with a CARWatch Study Manager summary.

    By default, laboratory samples are matched to ``recorded_sample`` from the
    app. This assigns accidentally swapped tubes to the sampling position at
    which they were actually collected. If no app record exists,
    ``scheduled_sample`` is used as a fallback.

    Parameters
    ----------
    study_results
        Study Manager summary returned by
        :func:`carwatch.io.load_study_manager_export`.
    saliva
        Measurements returned by :func:`carwatch.io.load_saliva` with the
        ``participant`` and ``sample`` index levels.
    correct_swaps
        Match laboratory tubes using ``recorded_sample``. Set to ``False`` to
        merge measurements by ``scheduled_sample`` instead.
    missing_carwatch_data
        How to handle laboratory values without a recorded CARWatch sampling
        event: ``"ignore"``, ``"raise"``, or ``"impute"``.
    sampling_schedule
        Required when ``missing_carwatch_data="impute"``. A one-dimensional
        array-like applies to every day. A dictionary maps day IDs to
        one-dimensional schedules. Numeric values denote minutes relative to
        awakening; strings denote absolute times in ``HH:MM`` or
        ``HH:MM:SS`` format.

    Returns
    -------
    pandas.DataFrame
        Study sampling positions and laboratory samples indexed by
        ``participant``, ``day``, and ``scheduled_sample``. Availability,
        mismatch correction, and sampling-time imputation remain explicit in
        boolean provenance columns.

    """
    _validate_options(correct_swaps, missing_carwatch_data, sampling_schedule)

    sample_events = logs.extract_sample_events_from_summary(study_results)
    day_summary = logs.extract_day_summary_from_summary(study_results)
    events = _normalize_sample_events(sample_events)
    laboratory, measurement_columns = _normalize_saliva(saliva)
    _validate_output_columns(events, measurement_columns)

    events["_event_row"] = range(len(events))
    recorded = _clean_identifier(events["recorded_sample"])
    if correct_swaps:
        events["_match_sample"] = recorded.fillna(events["scheduled_sample"])
    else:
        events["_match_sample"] = events["scheduled_sample"]

    _validate_event_matches(events)
    laboratory = laboratory.rename(columns={"sample": "_matched_sample"})
    merged = events.merge(
        laboratory,
        how="outer",
        left_on=["participant", "_match_sample"],
        right_on=["participant", "_matched_sample"],
        sort=False,
        validate="one_to_one",
        indicator="_merge_source",
    ).sort_values("_event_row", kind="stable", na_position="last")

    merged["scheduled_sample"] = _clean_identifier(merged["scheduled_sample"]).fillna(
        merged["_matched_sample"]
    )
    merged["sampling_event_recorded"] = (
        merged["sampling_event_recorded"].astype("boolean").fillna(False).astype(bool)
    )
    merged["lab_value_available"] = merged[measurement_columns].notna().all(axis=1)
    merged["sampling_time_imputed"] = False

    matched = merged["_merge_source"].eq("both")
    if correct_swaps:
        corrected = matched & merged["scheduled_sample"].ne(merged["_matched_sample"])
        merged["mismatch_corrected"] = corrected.fillna(False).astype(bool)
    else:
        merged["mismatch_corrected"] = False

    merged = _handle_missing_carwatch_data(
        merged,
        sample_events=sample_events,
        day_summary=day_summary,
        mode=missing_carwatch_data,
        sampling_schedule=sampling_schedule,
    )

    helper_columns = [
        "_event_row",
        "_match_sample",
        "_merge_source",
        "_matched_sample",
    ]
    return merged.drop(columns=helper_columns).set_index(_SUMMARY_INDEX)


def _validate_options(
    correct_swaps: bool,
    missing_carwatch_data: str,
    sampling_schedule,
) -> None:
    if not isinstance(correct_swaps, bool):
        raise TypeError("'correct_swaps' must be a boolean.")
    if (
        not isinstance(missing_carwatch_data, str)
        or missing_carwatch_data not in _MISSING_CARWATCH_DATA_OPTIONS
    ):
        raise ValueError(
            "'missing_carwatch_data' must be one of "
            f"{sorted(_MISSING_CARWATCH_DATA_OPTIONS)}, got {missing_carwatch_data!r}."
        )
    if missing_carwatch_data == "impute" and sampling_schedule is None:
        raise ValueError(
            "'sampling_schedule' is required when 'missing_carwatch_data' is 'impute'."
        )
    if missing_carwatch_data != "impute" and sampling_schedule is not None:
        raise ValueError(
            "'sampling_schedule' is only allowed when "
            "'missing_carwatch_data' is 'impute'."
        )


def _handle_missing_carwatch_data(
    merged: pd.DataFrame,
    *,
    sample_events: pd.DataFrame,
    day_summary: pd.DataFrame,
    mode: str,
    sampling_schedule,
) -> pd.DataFrame:
    missing_event_with_value = (
        ~merged["sampling_event_recorded"] & merged["lab_value_available"]
    )
    if mode == "ignore":
        return merged
    if mode == "raise" and missing_event_with_value.any():
        positions = _sampling_positions(merged.loc[missing_event_with_value])
        raise exceptions.MergeError(
            f"CARWatch sampling events are missing for laboratory samples: {positions}"
        )
    if mode == "raise":
        return merged

    missing_sampling_time = missing_event_with_value & merged["sampling_time"].isna()
    schedule = _normalize_sampling_schedule(
        sampling_schedule,
        sample_events=sample_events,
        affected_days=_affected_days(merged.loc[missing_sampling_time]),
    )
    if not missing_sampling_time.any():
        return merged

    references = day_summary.reset_index().set_index(["participant", "day"])
    updates = []
    for row_index, row in merged.loc[missing_sampling_time].iterrows():
        participant = row["participant"]
        day = row["day"]
        sample = row["scheduled_sample"]
        if pd.isna(day) or day not in schedule or sample not in schedule[day]:
            raise exceptions.MergeError(
                "Cannot impute a sampling time because the laboratory sample "
                "cannot be assigned to a scheduled Study Manager position: "
                f"participant={participant!r}, day={day!r}, sample={sample!r}."
            )
        try:
            reference = references.loc[(participant, day)]
        except KeyError as error:
            raise exceptions.MergeError(
                "Cannot impute a sampling time because day-level Study Manager "
                "information is missing for "
                f"participant={participant!r}, day={day!r}."
            ) from error

        scheduled_value = schedule[day][sample]
        sampling_time, time_min = _resolve_scheduled_time(
            scheduled_value,
            reference=reference,
            participant=participant,
            day=day,
            sample=sample,
        )
        updates.append((row_index, sampling_time, time_min))

    for row_index, sampling_time, time_min in updates:
        merged.at[row_index, "sampling_time"] = sampling_time
        merged.at[row_index, "time_min"] = time_min
        merged.at[row_index, "sampling_time_imputed"] = True
    return merged


def _normalize_sampling_schedule(
    sampling_schedule,
    *,
    sample_events: pd.DataFrame,
    affected_days: set[str],
) -> dict[str, dict[str, float | time]]:
    samples_by_day = _samples_by_day(sample_events)
    valid_days = set(samples_by_day)

    if isinstance(sampling_schedule, Mapping):
        invalid_keys = [key for key in sampling_schedule if not isinstance(key, str)]
        if invalid_keys:
            raise exceptions.SchemaError(
                "Sampling schedule dictionary keys must be Study Manager day IDs."
            )
        unknown_days = sorted(set(sampling_schedule).difference(valid_days))
        if unknown_days:
            raise exceptions.SchemaError(
                f"Sampling schedule contains unknown day IDs: {unknown_days}."
            )
        missing_days = sorted(affected_days.difference(sampling_schedule))
        if missing_days:
            raise exceptions.SchemaError(
                f"Sampling schedule is missing affected day IDs: {missing_days}."
            )
        schedules_by_day = {
            day: _schedule_values(values, day=day)
            for day, values in sampling_schedule.items()
        }
    else:
        values = _schedule_values(sampling_schedule)
        schedules_by_day = {day: values for day in samples_by_day}

    normalized: dict[str, dict[str, float | time]] = {}
    for day, values in schedules_by_day.items():
        samples = samples_by_day[day]
        if len(values) != len(samples):
            raise exceptions.SchemaError(
                f"Sampling schedule for {day!r} has {len(values)} values, "
                f"but the Study Manager summary defines {len(samples)} samples."
            )
        normalized[day] = dict(zip(samples, values, strict=True))
    return normalized


def _schedule_values(values, *, day: str | None = None) -> list[float | time]:
    label = "sampling schedule" if day is None else f"sampling schedule for {day!r}"
    if (
        isinstance(values, (str, bytes, Mapping))
        or not is_list_like(values)
        or getattr(values, "ndim", 1) != 1
    ):
        raise exceptions.SchemaError(f"The {label} must be a one-dimensional array.")

    values = list(values)
    if any(
        is_list_like(value) and not isinstance(value, (str, bytes)) for value in values
    ):
        raise exceptions.SchemaError(f"The {label} must be a one-dimensional array.")

    normalized: list[float | time] = []
    relative_values: list[float] = []
    for value in values:
        if isinstance(value, str):
            normalized.append(_parse_absolute_time(value, label=label))
        elif isinstance(value, Real) and not isinstance(value, bool):
            numeric = float(value)
            if not math.isfinite(numeric):
                raise exceptions.SchemaError(
                    f"The {label} contains a non-finite relative time: {value!r}."
                )
            normalized.append(numeric)
            relative_values.append(numeric)
        else:
            raise exceptions.SchemaError(
                f"The {label} contains an invalid value: {value!r}. "
                "Use numeric relative minutes or absolute time strings."
            )

    if any(
        current <= previous
        for previous, current in zip(relative_values, relative_values[1:])
    ):
        raise exceptions.SchemaError(
            f"Relative values in the {label} must be strictly increasing."
        )
    return normalized


def _parse_absolute_time(value: str, *, label: str) -> time:
    match = _ABSOLUTE_TIME_PATTERN.fullmatch(value.strip())
    if match is None:
        raise exceptions.SchemaError(
            f"The {label} contains an invalid absolute time: {value!r}. "
            "Expected 'HH:MM' or 'HH:MM:SS'."
        )
    return time(
        hour=int(match.group("hour")),
        minute=int(match.group("minute")),
        second=int(match.group("second") or 0),
    )


def _resolve_scheduled_time(
    value: float | time,
    *,
    reference: pd.Series,
    participant,
    day,
    sample,
) -> tuple[pd.Timestamp, float]:
    awakening_time = reference["awakening_time"]
    if isinstance(value, float):
        if pd.isna(awakening_time):
            raise exceptions.MergeError(
                "Cannot convert a relative sampling schedule without an "
                "awakening time for "
                f"participant={participant!r}, day={day!r}, sample={sample!r}."
            )
        return awakening_time + pd.to_timedelta(value, unit="m"), value

    date = reference["date"]
    if pd.isna(date):
        raise exceptions.MergeError(
            "Cannot convert an absolute sampling schedule without a date for "
            f"participant={participant!r}, day={day!r}, sample={sample!r}."
        )
    sampling_time = _combine_date_and_time(date, value)
    if pd.isna(awakening_time):
        time_min = float("nan")
    else:
        time_min = (sampling_time - awakening_time).total_seconds() / 60
    return sampling_time, time_min


def _combine_date_and_time(date: pd.Timestamp, clock_time: time) -> pd.Timestamp:
    timestamp = pd.Timestamp(datetime.combine(date.date(), clock_time))
    if date.tz is not None:
        timestamp = timestamp.tz_localize(
            date.tz,
            ambiguous="raise",
            nonexistent="raise",
        )
    return timestamp


def _samples_by_day(sample_events: pd.DataFrame) -> dict[str, list[str]]:
    positions = sample_events.index.to_frame(index=False)[
        ["day", "scheduled_sample"]
    ].drop_duplicates()
    return {
        day: group["scheduled_sample"].tolist()
        for day, group in positions.groupby("day", sort=False)
    }


def _affected_days(data: pd.DataFrame) -> set[str]:
    return set(data["day"].dropna().astype(str))


def _sampling_positions(data: pd.DataFrame) -> list[dict]:
    return data[_SUMMARY_INDEX].to_dict(orient="records")


def _normalize_sample_events(sample_events: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(sample_events, pd.DataFrame):
        raise TypeError("'sample_events' must be a pandas DataFrame.")
    if sample_events.empty:
        raise exceptions.SchemaError("Sample events must not be empty.")
    if list(sample_events.index.names) != _SUMMARY_INDEX:
        raise exceptions.SchemaError(
            "Sample events must come from a Study Manager summary extractor."
        )

    events = sample_events.reset_index().copy()
    missing = {"recorded_sample", "sampling_event_recorded"}.difference(events.columns)
    if missing:
        raise exceptions.SchemaError(
            f"Sample events are missing required columns: {sorted(missing)}"
        )
    try:
        events["sampling_event_recorded"] = events["sampling_event_recorded"].astype(
            "boolean"
        )
    except (TypeError, ValueError) as error:
        raise exceptions.SchemaError(
            "Column 'sampling_event_recorded' must contain boolean values."
        ) from error
    if events["sampling_event_recorded"].isna().any():
        raise exceptions.SchemaError(
            "Column 'sampling_event_recorded' contains missing values."
        )
    for column in ["participant", "day", "scheduled_sample"]:
        events[column] = _clean_identifier(events[column])
        if events[column].isna().any():
            raise exceptions.SchemaError(
                f"Sample event index column {column!r} contains missing values."
            )
    if events.duplicated(_SUMMARY_INDEX).any():
        duplicates = events.loc[
            events.duplicated(_SUMMARY_INDEX, keep=False), _SUMMARY_INDEX
        ]
        raise exceptions.SchemaError(
            "Sample events contain duplicate sampling positions: "
            f"{duplicates.drop_duplicates().to_dict(orient='records')}"
        )
    return events


def _normalize_saliva(saliva: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if not isinstance(saliva, pd.DataFrame):
        raise TypeError("'saliva' must be a pandas DataFrame.")
    if list(saliva.index.names) != _SALIVA_INDEX:
        raise exceptions.SchemaError(
            f"Saliva data require index levels {_SALIVA_INDEX}, got {saliva.index.names}."
        )
    if saliva.empty:
        raise exceptions.SchemaError("Saliva data must not be empty.")
    measurement_columns = saliva.columns.tolist()
    if not measurement_columns:
        raise exceptions.SchemaError("Saliva data do not contain a measurement column.")
    non_numeric = [
        column for column in measurement_columns if not is_numeric_dtype(saliva[column])
    ]
    if non_numeric:
        raise exceptions.SchemaError(
            f"Saliva measurement columns must be numeric: {non_numeric}"
        )

    laboratory = saliva.reset_index().copy()
    for column in _SALIVA_INDEX:
        laboratory[column] = _clean_identifier(laboratory[column])
        if laboratory[column].isna().any():
            raise exceptions.SchemaError(
                f"Saliva index level {column!r} contains missing values."
            )
    if laboratory.duplicated(_SALIVA_INDEX).any():
        raise exceptions.SchemaError(
            "Saliva data contain duplicate participant/sample pairs."
        )
    return laboratory, measurement_columns


def _validate_output_columns(events: pd.DataFrame, measurements: list[str]) -> None:
    output_columns = set(events.columns).union(_PROVENANCE_COLUMNS)
    conflicts = set(measurements).intersection(output_columns)
    if conflicts:
        raise exceptions.SchemaError(
            f"Saliva measurement columns conflict with merge output: {sorted(conflicts)}"
        )


def _validate_event_matches(events: pd.DataFrame) -> None:
    match_columns = ["participant", "_match_sample"]
    duplicates = events.duplicated(match_columns, keep=False)
    if duplicates.any():
        values = events.loc[duplicates, match_columns].drop_duplicates()
        raise exceptions.MergeError(
            "Multiple sample events refer to the same physical saliva tube: "
            f"{values.to_dict(orient='records')}"
        )


def _clean_identifier(data: pd.Series) -> pd.Series:
    return data.astype("string").str.strip().replace("", pd.NA)
