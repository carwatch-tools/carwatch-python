"""Merge saliva measurements with CARWatch sample events."""

from __future__ import annotations

import pandas as pd
from pandas.api.types import is_numeric_dtype

import carwatch.exceptions as exceptions

_SUMMARY_INDEX = ["participant", "day", "sample"]
_RAW_LOG_INDEX = ["participant", "date", "sample"]
_SALIVA_INDEX = ["subject", "sample"]
_PROVENANCE_COLUMNS = {
    "match_method",
    "merge_status",
    "mismatch_corrected",
}


def merge_saliva(
    sample_events: pd.DataFrame,
    saliva: pd.DataFrame,
    *,
    correct_swaps: bool = True,
    allow_unmatched: bool = True,
) -> pd.DataFrame:
    """Merge laboratory measurements onto CARWatch sample events.

    By default, each event is matched to the physical tube recorded in
    ``sample_scanned``. This assigns accidentally swapped tubes to the sampling
    position at which they were actually collected. If no scanned tube is
    recorded, the expected ``sample`` is used as a fallback.

    Parameters
    ----------
    sample_events
        Sample events returned by
        :func:`carwatch.logs.extract_sample_events_from_raw_logs` or
        :func:`carwatch.logs.extract_sample_events_from_summary`.
    saliva
        Measurements returned by :func:`carwatch.io.load_saliva` with the
        ``subject`` and ``sample`` index levels.
    correct_swaps
        Match laboratory tubes using ``sample_scanned``. Set to ``False`` to
        merge measurements by the expected ``sample`` labels instead.
    allow_unmatched
        Retain unmatched events and unused laboratory measurements if ``True``.
        If ``False``, an incomplete merge raises
        :class:`carwatch.exceptions.MergeError`.

    Returns
    -------
    pandas.DataFrame
        Sample events enriched with saliva measurements and merge provenance.
        Summary events retain a ``participant``/``day``/``sample`` index; raw
        log events use ``participant``/``date``/``sample``.

    """
    if not isinstance(correct_swaps, bool):
        raise TypeError("'correct_swaps' must be a boolean.")
    if not isinstance(allow_unmatched, bool):
        raise TypeError("'allow_unmatched' must be a boolean.")

    events, index_columns = _normalize_sample_events(sample_events)
    laboratory, measurement_columns = _normalize_saliva(saliva)
    _validate_output_columns(events, measurement_columns)

    events["_event_row"] = range(len(events))
    laboratory["_saliva_row"] = range(len(laboratory))
    scanned = _clean_identifier(events["sample_scanned"])
    if correct_swaps:
        events["_match_sample"] = scanned.fillna(events["sample"])
        events["_match_method"] = pd.Series(
            "expected_sample", index=events.index, dtype="string"
        ).mask(scanned.notna(), "sample_scanned")
    else:
        events["_match_sample"] = events["sample"]
        events["_match_method"] = pd.array(
            ["expected_sample"] * len(events), dtype="string"
        )

    _validate_event_matches(events)
    laboratory = laboratory.rename(
        columns={"subject": "participant", "sample": "_matched_sample"}
    )
    merged = events.merge(
        laboratory,
        how="left",
        left_on=["participant", "_match_sample"],
        right_on=["participant", "_matched_sample"],
        sort=False,
        validate="one_to_one",
        indicator="_merge_source",
    ).sort_values("_event_row", kind="stable")

    matched = merged["_merge_source"].eq("both")
    merged["match_method"] = merged["_match_method"].where(matched, pd.NA)
    merged["merge_status"] = pd.array(
        matched.map({True: "matched", False: "unmatched"}), dtype="string"
    )
    if correct_swaps:
        corrected = matched & merged["sample"].ne(merged["_matched_sample"])
        merged["mismatch_corrected"] = corrected.fillna(False).astype(bool)
    else:
        merged["mismatch_corrected"] = False

    unused_saliva = set(laboratory["_saliva_row"]) - set(
        merged.loc[matched, "_saliva_row"].astype(int)
    )
    if not allow_unmatched and ((~matched).any() or unused_saliva):
        raise exceptions.MergeError(
            f"Merge left {int((~matched).sum())} sample events unmatched and "
            f"{len(unused_saliva)} saliva measurements unused."
        )

    helper_columns = [
        "_event_row",
        "_saliva_row",
        "_match_sample",
        "_match_method",
        "_merge_source",
        "_matched_sample",
    ]
    return merged.drop(columns=helper_columns).set_index(index_columns)


def _normalize_sample_events(
    sample_events: pd.DataFrame,
) -> tuple[pd.DataFrame, list[str]]:
    if not isinstance(sample_events, pd.DataFrame):
        raise TypeError("'sample_events' must be a pandas DataFrame.")
    if sample_events.empty:
        raise exceptions.SchemaError("Sample events must not be empty.")

    index_names = list(sample_events.index.names)
    if index_names in [_SUMMARY_INDEX, _RAW_LOG_INDEX]:
        events = sample_events.reset_index().copy()
        index_columns = index_names
    elif set(_SUMMARY_INDEX).issubset(sample_events.columns):
        events = sample_events.copy()
        index_columns = _SUMMARY_INDEX
    elif set(_RAW_LOG_INDEX).issubset(sample_events.columns):
        events = sample_events.copy()
        index_columns = _RAW_LOG_INDEX
    else:
        raise exceptions.SchemaError(
            "Sample events must come from a raw-log or summary sample extractor."
        )

    missing = {"sample_scanned"}.difference(events.columns)
    if missing:
        raise exceptions.SchemaError(
            f"Sample events are missing required columns: {sorted(missing)}"
        )
    for column in ["participant", "sample"]:
        events[column] = _clean_identifier(events[column])
    for column in index_columns:
        if events[column].isna().any():
            raise exceptions.SchemaError(
                f"Sample event index column {column!r} contains missing values."
            )
    if events.duplicated(index_columns).any():
        duplicates = events.loc[
            events.duplicated(index_columns, keep=False), index_columns
        ]
        raise exceptions.SchemaError(
            "Sample events contain duplicate sampling positions: "
            f"{duplicates.drop_duplicates().to_dict(orient='records')}"
        )
    return events, index_columns


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
            "Saliva data contain duplicate subject/sample pairs."
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
