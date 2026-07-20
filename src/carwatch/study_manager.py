"""Extract focused views from wide CARWatch Study Manager results."""

from __future__ import annotations

import pandas as pd

from carwatch.exceptions import SchemaError

__all__ = ["extract_awakening", "extract_samples"]

_COLUMN_LEVELS = ["day", "sample", "variable"]
_DAY_SAMPLE = "day"
_DAY_VARIABLES = (
    "date",
    "awakening_time",
    "awakening_type",
    "mismatch_summary",
)


def extract_awakening(study_results: pd.DataFrame) -> pd.DataFrame:
    """Extract one row of day-level information per participant and day.

    The recorded mismatch summary is retained once per day alongside the
    awakening information instead of being repeated for every sample.
    """
    _validate_results(study_results)
    rows: list[dict] = []
    index: list[tuple] = []
    for participant in study_results.index:
        for day in _days(study_results):
            rows.append(
                {
                    variable: _value(
                        study_results,
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


def extract_samples(study_results: pd.DataFrame) -> pd.DataFrame:
    """Extract one row per expected sample and derive sample-level fields."""
    _validate_results(study_results)
    rows: list[dict] = []
    index: list[tuple] = []
    for participant in study_results.index:
        for day in _days(study_results):
            awakening_time = _value(
                study_results,
                participant,
                (day, _DAY_SAMPLE, "awakening_time"),
            )
            for sample in _samples(study_results, day):
                sampling_time = _value(
                    study_results,
                    participant,
                    (day, sample, "sampling_time"),
                )
                barcode = _value(
                    study_results,
                    participant,
                    (day, sample, "barcode"),
                )
                sample_scanned = _value(
                    study_results,
                    participant,
                    (day, sample, "sample_scanned"),
                )
                rows.append(
                    {
                        "sampling_time": sampling_time,
                        "time": _minutes_between(sampling_time, awakening_time),
                        "barcode": barcode,
                        "sample_scanned": sample_scanned,
                        "sample_mismatch": _sample_mismatch(sample, sample_scanned),
                        "observed": not (
                            pd.isna(sampling_time) and _is_missing(barcode)
                        ),
                    }
                )
                index.append((participant, day, sample))

    result = pd.DataFrame(
        rows,
        index=pd.MultiIndex.from_tuples(index, names=["participant", "day", "sample"]),
        columns=[
            "sampling_time",
            "time",
            "barcode",
            "sample_scanned",
            "sample_mismatch",
            "observed",
        ],
    )
    result["barcode"] = pd.array(result["barcode"], dtype="string")
    result["sample_scanned"] = pd.array(result["sample_scanned"], dtype="string")
    result["sample_mismatch"] = pd.array(result["sample_mismatch"], dtype="boolean")
    result["observed"] = pd.array(result["observed"], dtype="boolean")
    return result


def _validate_results(study_results: pd.DataFrame) -> None:
    if not isinstance(study_results, pd.DataFrame):
        raise TypeError("'study_results' must be a pandas DataFrame.")
    if study_results.index.name != "participant":
        raise SchemaError("Study results require a 'participant' index.")
    if study_results.index.has_duplicates:
        raise SchemaError("Study results contain duplicate participants.")
    if not isinstance(study_results.columns, pd.MultiIndex):
        raise SchemaError("Study results require MultiIndex columns.")
    if list(study_results.columns.names) != _COLUMN_LEVELS:
        raise SchemaError(f"Study result column levels must be named {_COLUMN_LEVELS}.")
    if study_results.columns.has_duplicates:
        raise SchemaError("Study results contain duplicate columns.")


def _days(study_results: pd.DataFrame) -> list[str]:
    return list(dict.fromkeys(study_results.columns.get_level_values("day")))


def _samples(study_results: pd.DataFrame, day: str) -> list[str]:
    day_columns = study_results.xs(day, axis=1, level="day").columns
    return [
        sample
        for sample in dict.fromkeys(day_columns.get_level_values("sample"))
        if sample != _DAY_SAMPLE
    ]


def _value(study_results: pd.DataFrame, participant, column: tuple):
    if column not in study_results.columns:
        raise SchemaError(f"Study results are missing required column: {column}")
    return study_results.at[participant, column]


def _minutes_between(later: pd.Timestamp, earlier: pd.Timestamp) -> float:
    if pd.isna(later) or pd.isna(earlier):
        return float("nan")
    return (later - earlier).total_seconds() / 60


def _sample_mismatch(sample: str, sample_scanned) -> object:
    if _is_missing(sample_scanned):
        return pd.NA
    return str(sample) != str(sample_scanned)


def _is_missing(value) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""
