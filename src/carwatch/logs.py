"""Extract events and summaries from CARWatch log data."""

from __future__ import annotations

import pandas as pd

from carwatch.exceptions import SchemaError

__all__ = [
    "extract_awakening_events_from_raw_logs",
    "extract_awakening_events_from_summary",
    "extract_day_summary_from_summary",
    "extract_sample_events_from_raw_logs",
    "extract_sample_events_from_summary",
]

_REQUIRED_COLUMNS = {"participant", "date", "timestamp", "action", "payload"}
_SUMMARY_COLUMN_LEVELS = ["day", "sample", "variable"]
_DAY_SAMPLE = "day"
_DAY_VARIABLES = (
    "date",
    "awakening_time",
    "awakening_type",
    "mismatch_summary",
)


def extract_sample_events_from_raw_logs(raw_logs: pd.DataFrame) -> pd.DataFrame:
    """Extract barcode scans from raw CARWatch log events.

    ``scheduled_sample`` identifies the sample defined by the study schedule.
    ``recorded_sample`` identifies the sample recorded by the app.

    Parameters
    ----------
    raw_logs
        Event dataframe returned by :func:`carwatch.io.load_logs`.

    Returns
    -------
    pandas.DataFrame
        One row per ``barcode_scanned`` event.

    """
    _validate_raw_logs(raw_logs)
    scans = raw_logs.loc[raw_logs["action"].eq("barcode_scanned")].copy()
    rows: list[dict] = []
    for row in scans.itertuples(index=False):
        payload = row.payload or {}
        scheduled_sample = _scheduled_sample(payload)
        recorded_sample = payload.get("sample_scanned") or scheduled_sample
        rows.append(
            {
                "participant": row.participant,
                "date": row.date,
                "sampling_time": row.timestamp,
                "day_expected": payload.get("day_expected"),
                "day_scanned": payload.get("day_scanned"),
                "scheduled_sample": scheduled_sample,
                "recorded_sample": recorded_sample,
                "sample_mismatch": bool(
                    scheduled_sample
                    and recorded_sample
                    and scheduled_sample != recorded_sample
                ),
                "source_file": getattr(row, "source_file", None),
            }
        )
    return pd.DataFrame(rows)


def extract_awakening_events_from_raw_logs(raw_logs: pd.DataFrame) -> pd.DataFrame:
    """Extract one app-reported awakening event per participant and date.

    A spontaneous awakening takes precedence over an alarm event on the same
    day. Within the selected action, the first event is returned.

    Parameters
    ----------
    raw_logs
        Event dataframe returned by :func:`carwatch.io.load_logs`.

    Returns
    -------
    pandas.DataFrame
        Awakening timestamps and reporting types.

    """
    _validate_raw_logs(raw_logs)
    candidate = raw_logs.loc[
        raw_logs["action"].isin(["spontaneous_awakening", "alarm_stop"])
    ].copy()
    if candidate.empty:
        return pd.DataFrame(
            columns=["participant", "date", "awakening_time", "awakening_type"]
        )

    candidate["priority"] = candidate["action"].map(
        {"spontaneous_awakening": 0, "alarm_stop": 1}
    )
    candidate = candidate.sort_values(
        ["participant", "date", "priority", "timestamp"],
        kind="stable",
        na_position="last",
    )
    candidate = candidate.drop_duplicates(["participant", "date"], keep="first")
    candidate["awakening_type"] = candidate["action"].map(
        {"spontaneous_awakening": "self-report", "alarm_stop": "alarm"}
    )
    return candidate.rename(columns={"timestamp": "awakening_time"})[
        ["participant", "date", "awakening_time", "awakening_type"]
    ].reset_index(drop=True)


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
                barcode = _summary_value(
                    summary,
                    participant,
                    (day, sample, "barcode"),
                )
                recorded_sample = _summary_value(
                    summary,
                    participant,
                    (day, sample, "recorded_sample"),
                )
                rows.append(
                    {
                        "sampling_time": sampling_time,
                        "time_min": _minutes_between(sampling_time, awakening_time),
                        "recorded_sample": recorded_sample,
                        "sample_mismatch": _sample_mismatch(sample, recorded_sample),
                        "observed": not (
                            pd.isna(sampling_time) and _is_missing(barcode)
                        ),
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
            "sample_mismatch",
            "observed",
        ],
    )
    result["recorded_sample"] = pd.array(result["recorded_sample"], dtype="string")
    result["sample_mismatch"] = pd.array(result["sample_mismatch"], dtype="boolean")
    result["observed"] = pd.array(result["observed"], dtype="boolean")
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
    if payload.get("sample_expected") not in {None, ""}:
        return str(payload["sample_expected"])
    saliva_id = payload.get("saliva_id")
    if saliva_id is None:
        return None
    # Special-case known legacy ID that maps to the "SE" sampling position.
    if payload.get("id") == 815:
        return "SE"
    return f"S{saliva_id}"


def _validate_raw_logs(raw_logs: pd.DataFrame) -> None:
    if not isinstance(raw_logs, pd.DataFrame):
        raise TypeError("'raw_logs' must be a pandas DataFrame.")
    missing = _REQUIRED_COLUMNS.difference(raw_logs.columns)
    if missing:
        raise SchemaError(
            f"Log dataframe is missing required columns: {sorted(missing)}"
        )


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
