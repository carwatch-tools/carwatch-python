"""Extract structured information from raw CARWatch log events."""

from __future__ import annotations

import pandas as pd

from carwatch.exceptions import SchemaError

__all__ = ["extract_awakening", "extract_samples"]

_REQUIRED_COLUMNS = {"study", "participant", "date", "timestamp", "action", "payload"}


def extract_samples(logs: pd.DataFrame) -> pd.DataFrame:
    """Extract barcode scans from raw CARWatch log events.

    The returned ``sample`` is the expected sampling position, while
    ``sample_scanned`` identifies the physical tube that was scanned.

    Parameters
    ----------
    logs
        Event dataframe returned by :func:`carwatch.io.load_logs`.

    Returns
    -------
    pandas.DataFrame
        One row per ``barcode_scanned`` event.

    """
    _validate_logs(logs)
    scans = logs.loc[logs["action"].eq("barcode_scanned")].copy()
    rows: list[dict] = []
    for row in scans.itertuples(index=False):
        payload = row.payload or {}
        expected = _expected_sample(payload)
        scanned = payload.get("sample_scanned") or expected
        rows.append(
            {
                "study": row.study,
                "participant": row.participant,
                "date": row.date,
                "sampling_time": row.timestamp,
                "day_expected": payload.get("day_expected"),
                "day_scanned": payload.get("day_scanned"),
                "sample": expected,
                "sample_scanned": scanned,
                "barcode": _string_or_missing(payload.get("barcode_value")),
                "sample_mismatch": bool(expected and scanned and expected != scanned),
                "source_file": getattr(row, "source_file", None),
                "event_index": getattr(row, "event_index", None),
            }
        )
    return pd.DataFrame(rows)


def extract_awakening(logs: pd.DataFrame) -> pd.DataFrame:
    """Extract one app-reported awakening event per participant and date.

    A spontaneous awakening takes precedence over an alarm event on the same
    day. Within the selected action, the first event is returned.

    Parameters
    ----------
    logs
        Event dataframe returned by :func:`carwatch.io.load_logs`.

    Returns
    -------
    pandas.DataFrame
        Awakening timestamps and reporting types.

    """
    _validate_logs(logs)
    candidate = logs.loc[
        logs["action"].isin(["spontaneous_awakening", "alarm_stop"])
    ].copy()
    if candidate.empty:
        return pd.DataFrame(
            columns=["study", "participant", "date", "awakening_time", "awakening_type"]
        )

    candidate["priority"] = candidate["action"].map(
        {"spontaneous_awakening": 0, "alarm_stop": 1}
    )
    candidate = candidate.sort_values(
        ["study", "participant", "date", "priority", "timestamp"],
        kind="stable",
        na_position="last",
    )
    candidate = candidate.drop_duplicates(
        ["study", "participant", "date"], keep="first"
    )
    candidate["awakening_type"] = candidate["action"].map(
        {"spontaneous_awakening": "self-report", "alarm_stop": "alarm"}
    )
    return candidate.rename(columns={"timestamp": "awakening_time"})[
        ["study", "participant", "date", "awakening_time", "awakening_type"]
    ].reset_index(drop=True)


def _expected_sample(payload: dict) -> str | None:
    if payload.get("sample_expected") not in {None, ""}:
        return str(payload["sample_expected"])
    saliva_id = payload.get("saliva_id")
    if saliva_id is None:
        return None
    # Special-case known legacy ID that maps to the "SE" sampling position.
    if payload.get("id") == 815:
        return "SE"
    return f"S{saliva_id}"


def _string_or_missing(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(value)


def _validate_logs(logs: pd.DataFrame) -> None:
    if not isinstance(logs, pd.DataFrame):
        raise TypeError("'logs' must be a pandas DataFrame.")
    missing = _REQUIRED_COLUMNS.difference(logs.columns)
    if missing:
        raise SchemaError(
            f"Log dataframe is missing required columns: {sorted(missing)}"
        )
