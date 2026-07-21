"""Extract events and summaries from CARWatch log data."""

from __future__ import annotations

import math

import pandas as pd

from carwatch.exceptions import SchemaError

__all__ = [
    "extract_awakening_events_from_raw_logs",
    "extract_awakening_events_from_summary",
    "convert_raw_logs_to_study_manager_summary",
    "extract_day_summary_from_summary",
    "extract_sample_events_from_raw_logs",
    "extract_sample_events_from_summary",
]

_RAW_LOG_INDEX = ["participant", "date", "timestamp"]
_REQUIRED_RAW_LOG_COLUMNS = {"action", "payload"}
_SUMMARY_COLUMN_LEVELS = ["day", "sample", "variable"]
_DAY_SAMPLE = "day"
_DAY_VARIABLES = (
    "date",
    "awakening_time",
    "awakening_type",
    "mismatch_summary",
)
_SAMPLE_VARIABLES = ("sampling_time", "barcode", "recorded_sample")


def convert_raw_logs_to_study_manager_summary(raw_logs: pd.DataFrame) -> pd.DataFrame:
    """Convert raw CARWatch logs to a canonical Study Manager summary.

    Awakening and sampling events follow the Study Manager processing logic:
    the first awakening event per log file is used, sampling events are sorted
    chronologically, legacy sample IDs are normalized, repeated expected sample
    IDs are advanced, and sample mismatches are retained once per day.

    Parameters
    ----------
    raw_logs
        Event dataframe returned by :func:`carwatch.io.load_raw_logs`.

    Returns
    -------
    pandas.DataFrame
        One row per participant with the same ``day``/``sample``/``variable``
        column levels as :func:`carwatch.io.load_study_manager_export`.

    """
    raw_logs = _normalize_raw_logs(raw_logs)
    if "source_file" not in raw_logs:
        raise SchemaError("Raw logs require a 'source_file' column for conversion.")

    day_records = []
    for group in _raw_log_file_groups(raw_logs):
        awakening = _extract_awakening_info(group)
        samples = _extract_sampling_rows_from_group(group)
        if awakening is None and not samples:
            continue
        day_records.append(
            {
                "participant": group["participant"].iloc[0],
                "date": group["date"].iloc[0],
                "awakening": awakening,
                "samples": samples,
                "source_file": group["source_file"].iloc[0],
            }
        )

    if not day_records:
        raise SchemaError(
            "Raw logs do not contain awakening or sampling events for conversion."
        )

    day_records.sort(
        key=lambda record: (
            str(record["participant"]),
            record["date"],
            str(record["source_file"]),
        )
    )
    reference_samples = _reference_sample_ids(day_records)
    participants = sorted(
        {str(record["participant"]) for record in day_records},
        key=str.casefold,
    )
    records_by_participant = {
        participant: [
            record
            for record in day_records
            if str(record["participant"]) == participant
        ]
        for participant in participants
    }
    day_count = max(len(records) for records in records_by_participant.values())
    columns = _summary_column_order(day_count, reference_samples)
    rows = [
        _summary_participant_row(
            records_by_participant[participant],
            day_count=day_count,
            reference_samples=reference_samples,
        )
        for participant in participants
    ]
    result = pd.DataFrame(
        rows,
        index=pd.Index(participants, name="participant", dtype="string"),
    ).reindex(columns=columns)
    result.columns = pd.MultiIndex.from_tuples(columns, names=_SUMMARY_COLUMN_LEVELS)
    return _set_summary_string_dtypes(result)


def extract_sample_events_from_raw_logs(raw_logs: pd.DataFrame) -> pd.DataFrame:
    """Extract barcode scans from raw CARWatch log events.

    ``scheduled_sample`` identifies the sample defined by the study schedule.
    ``recorded_sample`` identifies the sample recorded by the app.

    Parameters
    ----------
    raw_logs
        Event dataframe returned by :func:`carwatch.io.load_raw_logs`.

    Returns
    -------
    pandas.DataFrame
        One row per ``barcode_scanned`` event.

    """
    raw_logs = _normalize_raw_logs(raw_logs)
    rows = [
        row
        for group in _raw_log_file_groups(raw_logs)
        for row in _extract_sampling_rows_from_group(group)
    ]
    columns = [
        "participant",
        "date",
        "sampling_time",
        "day_expected",
        "day_scanned",
        "scheduled_sample",
        "recorded_sample",
        "sampling_event_recorded",
        "sample_mismatch",
        "source_file",
    ]
    return pd.DataFrame(rows, columns=columns)


def extract_awakening_events_from_raw_logs(raw_logs: pd.DataFrame) -> pd.DataFrame:
    """Extract one app-reported awakening event per participant and date.

    The first ``spontaneous_awakening`` or ``alarm_stop`` event on each day is
    returned, matching the Study Manager processing logic.

    Parameters
    ----------
    raw_logs
        Event dataframe returned by :func:`carwatch.io.load_raw_logs`.

    Returns
    -------
    pandas.DataFrame
        Awakening timestamps and reporting types.

    """
    raw_logs = _normalize_raw_logs(raw_logs)
    candidate = raw_logs.loc[
        raw_logs["action"].isin(["spontaneous_awakening", "alarm_stop"])
    ].copy()
    if candidate.empty:
        return pd.DataFrame(
            columns=["participant", "date", "awakening_time", "awakening_type"]
        )

    candidate = candidate.sort_values(
        ["participant", "date", "timestamp"],
        kind="stable",
        na_position="last",
    )
    candidate = candidate.drop_duplicates(["participant", "date"], keep="first")
    candidate["awakening_type"] = candidate["action"].map(
        {"spontaneous_awakening": "self-report", "alarm_stop": "alarm"}
    )
    result = candidate.rename(columns={"timestamp": "awakening_time"})[
        ["participant", "date", "awakening_time", "awakening_type"]
    ].reset_index(drop=True)
    result["awakening_time"] = result["awakening_time"].dt.floor("s")
    return result


def extract_sample_events_from_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Extract one row per scheduled sample from a Study Manager summary."""
    _validate_summary(summary)
    rows: list[dict] = []
    index: list[tuple] = []
    for participant in summary.index:
        for day in _days(summary):
            awakening_time = _summary_value(
                summary,
                participant,
                (day, _DAY_SAMPLE, "awakening_time"),
            )
            for sample in _samples(summary, day):
                sampling_time = _summary_value(
                    summary,
                    participant,
                    (day, sample, "sampling_time"),
                )
                recorded_sample = _summary_value(
                    summary,
                    participant,
                    (day, sample, "recorded_sample"),
                )
                barcode = _summary_value_or_na(
                    summary,
                    participant,
                    (day, sample, "barcode"),
                )
                rows.append(
                    {
                        "sampling_time": sampling_time,
                        "time_min": _minutes_between(sampling_time, awakening_time),
                        "recorded_sample": recorded_sample,
                        "sampling_event_recorded": any(
                            pd.notna(value)
                            for value in (sampling_time, recorded_sample, barcode)
                        ),
                        "sample_mismatch": _sample_mismatch(sample, recorded_sample),
                    }
                )
                index.append((participant, day, sample))

    result = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(
            index, names=["participant", "day", "scheduled_sample"]
        ),
        columns=[
            "sampling_time",
            "time_min",
            "recorded_sample",
            "sampling_event_recorded",
            "sample_mismatch",
        ],
    )
    result["recorded_sample"] = pd.array(result["recorded_sample"], dtype="string")
    result["sampling_event_recorded"] = result["sampling_event_recorded"].astype(bool)
    result["sample_mismatch"] = pd.array(result["sample_mismatch"], dtype="boolean")
    return result


def extract_awakening_events_from_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Extract awakening information per participant and day from a summary."""
    day_summary = extract_day_summary_from_summary(summary)
    return day_summary[["date", "awakening_time", "awakening_type"]].copy()


def extract_day_summary_from_summary(summary: pd.DataFrame) -> pd.DataFrame:
    """Extract day-level information from a Study Manager summary."""
    _validate_summary(summary)
    rows: list[dict] = []
    index: list[tuple] = []
    for participant in summary.index:
        for day in _days(summary):
            rows.append(
                {
                    variable: _summary_value(
                        summary,
                        participant,
                        (day, _DAY_SAMPLE, variable),
                    )
                    for variable in _DAY_VARIABLES
                }
            )
            index.append((participant, day))

    result = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=["participant", "day"]),
        columns=_DAY_VARIABLES,
    )
    for column in ["awakening_type", "mismatch_summary"]:
        result[column] = pd.array(result[column], dtype="string")
    return result


def _scheduled_sample(payload: dict) -> str | None:
    if "sample_expected" in payload:
        value = payload["sample_expected"]
        return "" if value is None else str(value)
    saliva_id = payload.get("saliva_id")
    if saliva_id is None:
        return None
    # Special-case known legacy ID that maps to the "SE" sampling position.
    if payload.get("id") == 815:
        return "SE"
    return f"S{saliva_id}"


def _raw_log_file_groups(raw_logs: pd.DataFrame) -> list[pd.DataFrame]:
    group_columns = ["participant", "source_file"]
    if "source_file" not in raw_logs:
        group_columns = ["participant", "date"]
    ordered = raw_logs.sort_values(
        [*group_columns, "timestamp"],
        kind="stable",
        na_position="last",
    )
    return [
        group.copy()
        for _, group in ordered.groupby(group_columns, sort=False, dropna=False)
    ]


def _extract_awakening_info(group: pd.DataFrame) -> dict | None:
    awakening = group.loc[
        group["action"].isin(["spontaneous_awakening", "alarm_stop"])
    ].sort_values("timestamp", kind="stable")
    if awakening.empty:
        return None
    first = awakening.iloc[0]
    return {
        "awakening_time": first["timestamp"].floor("s"),
        "awakening_type": {
            "spontaneous_awakening": "self-report",
            "alarm_stop": "alarm",
        }[first["action"]],
    }


def _extract_sampling_rows_from_group(group: pd.DataFrame) -> list[dict]:
    scans = group.loc[group["action"].eq("barcode_scanned")].sort_values(
        "timestamp", kind="stable"
    )
    sample_ids: list[str] = []
    rows = []
    for row in scans.itertuples(index=False):
        payload = row.payload or {}
        scheduled_sample = _scheduled_sample(payload) or ""
        if scheduled_sample in sample_ids or scheduled_sample == "SM":
            scheduled_sample = _advance_duplicate_sample(scheduled_sample, sample_ids)
        sample_ids.append(scheduled_sample)
        recorded_sample = payload.get("sample_scanned") or scheduled_sample
        rows.append(
            {
                "participant": row.participant,
                "date": row.date,
                "sampling_time": row.timestamp.floor("s"),
                "day_expected": payload.get("day_expected"),
                "day_scanned": payload.get("day_scanned"),
                "scheduled_sample": scheduled_sample,
                "recorded_sample": recorded_sample,
                "sampling_event_recorded": True,
                "sample_mismatch": bool(
                    scheduled_sample
                    and recorded_sample
                    and scheduled_sample != recorded_sample
                ),
                "source_file": getattr(row, "source_file", None),
                "barcode": payload.get("barcode_value"),
            }
        )
    return rows


def _advance_duplicate_sample(sample: str, previous_samples: list[str]) -> str:
    if not previous_samples:
        return sample
    suffixes = []
    for previous in previous_samples:
        try:
            suffixes.append(float(previous[-1]))
        except (IndexError, ValueError):
            suffixes.append(float("nan"))
    if any(math.isnan(value) for value in suffixes):
        return sample
    max_index = max(suffixes, default=0)
    if not max_index:
        return sample
    return f"{sample[:-1]}{int(max_index + 1)}"


def _reference_sample_ids(day_records: list[dict]) -> list[str]:
    reference = []
    max_count = 0
    for record in day_records:
        samples = record["samples"]
        if len(samples) > max_count:
            max_count = len(samples)
            reference = [sample["scheduled_sample"] for sample in samples]
    return reference


def _summary_column_order(
    day_count: int, reference_samples: list[str]
) -> list[tuple[str, str, str]]:
    columns = []
    for day_number in range(1, day_count + 1):
        day = f"D{day_number}"
        columns.extend((day, _DAY_SAMPLE, variable) for variable in _DAY_VARIABLES)
        for sample in reference_samples:
            columns.extend((day, sample, variable) for variable in _SAMPLE_VARIABLES)
    return columns


def _summary_participant_row(
    day_records: list[dict],
    *,
    day_count: int,
    reference_samples: list[str],
) -> dict[tuple[str, str, str], object]:
    row = {}
    for day_number in range(1, day_count + 1):
        day = f"D{day_number}"
        record = day_records[day_number - 1] if day_number <= len(day_records) else None
        if record is None:
            row.update(
                {
                    (day, _DAY_SAMPLE, "date"): pd.NaT,
                    (day, _DAY_SAMPLE, "awakening_time"): pd.NaT,
                    (day, _DAY_SAMPLE, "awakening_type"): pd.NA,
                    (day, _DAY_SAMPLE, "mismatch_summary"): pd.NA,
                }
            )
            continue

        awakening = record["awakening"] or {}
        row[(day, _DAY_SAMPLE, "date")] = record["date"]
        row[(day, _DAY_SAMPLE, "awakening_time")] = awakening.get(
            "awakening_time", pd.NaT
        )
        row[(day, _DAY_SAMPLE, "awakening_type")] = awakening.get(
            "awakening_type", pd.NA
        )
        mismatches = [
            f"{sample['scheduled_sample']}->{sample['recorded_sample']}"
            for sample in record["samples"]
            if sample["scheduled_sample"]
            and sample["recorded_sample"]
            and sample["scheduled_sample"] != sample["recorded_sample"]
        ]
        row[(day, _DAY_SAMPLE, "mismatch_summary")] = (
            ";".join(mismatches) if mismatches else pd.NA
        )
        samples = {sample["scheduled_sample"]: sample for sample in record["samples"]}
        for sample_id in reference_samples:
            sample = samples.get(sample_id)
            row[(day, sample_id, "sampling_time")] = (
                sample["sampling_time"] if sample else pd.NaT
            )
            row[(day, sample_id, "barcode")] = sample["barcode"] if sample else pd.NA
            row[(day, sample_id, "recorded_sample")] = (
                sample["recorded_sample"] if sample else pd.NA
            )
    return row


def _set_summary_string_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    for column in data.columns:
        if column[-1] in {
            "awakening_type",
            "mismatch_summary",
            "barcode",
            "recorded_sample",
        }:
            data[column] = pd.array(data[column], dtype="string")
    return data


def _normalize_raw_logs(raw_logs: pd.DataFrame) -> pd.DataFrame:
    if not isinstance(raw_logs, pd.DataFrame):
        raise TypeError("'raw_logs' must be a pandas DataFrame.")
    if not isinstance(raw_logs.index, pd.MultiIndex):
        raise SchemaError(
            f"Raw logs require a MultiIndex with levels {_RAW_LOG_INDEX}."
        )
    if list(raw_logs.index.names) != _RAW_LOG_INDEX:
        raise SchemaError(f"Raw log index levels must be named {_RAW_LOG_INDEX}.")
    missing = _REQUIRED_RAW_LOG_COLUMNS.difference(raw_logs.columns)
    if missing:
        raise SchemaError(
            f"Log dataframe is missing required columns: {sorted(missing)}"
        )
    return raw_logs.reset_index()


def _validate_summary(summary: pd.DataFrame) -> None:
    if not isinstance(summary, pd.DataFrame):
        raise TypeError("'summary' must be a pandas DataFrame.")
    if summary.index.name != "participant":
        raise SchemaError("Study summary requires a 'participant' index.")
    if summary.index.has_duplicates:
        raise SchemaError("Study summary contains duplicate participants.")
    if not isinstance(summary.columns, pd.MultiIndex):
        raise SchemaError("Study summary requires MultiIndex columns.")
    if list(summary.columns.names) != _SUMMARY_COLUMN_LEVELS:
        raise SchemaError(
            f"Study summary column levels must be named {_SUMMARY_COLUMN_LEVELS}."
        )
    if summary.columns.has_duplicates:
        raise SchemaError("Study summary contains duplicate columns.")


def _days(summary: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(summary.columns.get_level_values("day")))


def _samples(summary: pd.DataFrame, day: str) -> list[str]:
    day_columns = summary.xs(day, axis=1, level="day").columns
    return [
        sample
        for sample in dict.fromkeys(day_columns.get_level_values("sample"))
        if sample != _DAY_SAMPLE
    ]


def _summary_value(summary: pd.DataFrame, participant, column: tuple):
    if column not in summary.columns:
        raise SchemaError(f"Study summary is missing required column: {column}")
    return summary.at[participant, column]


def _summary_value_or_na(summary: pd.DataFrame, participant, column: tuple):
    if column not in summary.columns:
        return pd.NA
    return summary.at[participant, column]


def _minutes_between(later: pd.Timestamp, earlier: pd.Timestamp) -> float:
    if pd.isna(later) or pd.isna(earlier):
        return float("nan")
    return (later - earlier).total_seconds() / 60


def _sample_mismatch(scheduled_sample: str, recorded_sample) -> object:
    if _is_missing(recorded_sample):
        return pd.NA
    return str(scheduled_sample) != str(recorded_sample)


def _is_missing(value) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""
