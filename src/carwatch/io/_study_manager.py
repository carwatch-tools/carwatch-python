"""Load study results exported by the CARWatch Study Manager."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

from carwatch.exceptions import SchemaError

_DATE_COLUMN = re.compile(r"^date_d(?P<day>\d+)$", re.IGNORECASE)
_SAMPLE_COLUMN = re.compile(
    r"^(?P<field>sampling_time|sample_barcode|sample_scanned)_d(?P<day>\d+)_(?P<sample>.+)$",
    re.IGNORECASE,
)
_DAY_SAMPLE = "day"
_DAY_VARIABLES = (
    "date",
    "awakening_time",
    "awakening_type",
    "mismatch_summary",
)
_SAMPLE_VARIABLES = ("sampling_time", "barcode", "recorded_sample")
_COLUMN_LEVELS = ["day", "sample", "variable"]


def load_study_manager_export(
    path: str | Path, *, tz: str = "Europe/Berlin"
) -> pd.DataFrame:
    """Load a CARWatch Study Manager CSV export in canonical wide format.

    The returned dataframe contains one row per participant. Its columns use
    the levels ``day``, ``sample``, and ``variable``. Day-level information
    uses the reserved sample label ``"day"``. Sample-level information uses
    the expected sample label from the protocol.

    Parameters
    ----------
    path
        Path to the Study Manager CSV export.
    tz
        Timezone used to combine the exported dates and local times.

    Returns
    -------
    pandas.DataFrame
        Wide results indexed by participant with a three-level column index.

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Study results file does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("Study results must be supplied as a CSV file.")

    data = pd.read_csv(path, dtype="string", keep_default_na=False)
    if data.empty:
        raise SchemaError("Study results file does not contain any participants.")

    columns = _case_insensitive_columns(data)
    participant_col = _required_column(columns, "participant id")
    days, samples = _discover_layout(data.columns)
    if not days:
        raise SchemaError("Study results do not contain any 'date_D*' columns.")
    if not any(samples.values()):
        raise SchemaError("Study results do not contain any sample columns.")
    missing_dates = sorted(set(samples).difference(days))
    if missing_dates:
        raise SchemaError(
            f"Study results contain samples without matching date columns for days: {missing_dates}"
        )

    participants = [_clean_string(value) for value in data[participant_col]]
    if any(participant is None for participant in participants):
        raise SchemaError("Participant ID must not be empty.")
    participant_index = pd.Index(participants, name="participant", dtype="string")
    if participant_index.has_duplicates:
        duplicates = participant_index[participant_index.duplicated()].unique().tolist()
        raise SchemaError(
            f"Study results contain duplicate participant IDs: {duplicates}"
        )

    rows = [
        _normalize_participant(
            source_row,
            columns=columns,
            days=days,
            samples=samples,
            tz=tz,
        )
        for source_row in data.to_dict(orient="records")
    ]
    column_order = _column_order(days, samples)
    result = pd.DataFrame(rows, index=participant_index).reindex(columns=column_order)
    result.columns = pd.MultiIndex.from_tuples(column_order, names=_COLUMN_LEVELS)
    return _set_string_dtypes(result)


def _normalize_participant(
    source_row: dict,
    *,
    columns: dict[str, str],
    days: list[int],
    samples: dict[int, list[str]],
    tz: str,
) -> dict[tuple[str, str, str], object]:
    normalized: dict[tuple[str, str, str], object] = {}
    for day_number in days:
        day = f"D{day_number}"
        date = _parse_date(
            _get_value(source_row, columns, f"date_d{day_number}"),
            tz=tz,
        )
        normalized[(day, _DAY_SAMPLE, "date")] = date
        normalized[(day, _DAY_SAMPLE, "awakening_time")] = _combine_date_time(
            date,
            _get_value(source_row, columns, f"awakening_time_d{day_number}_app"),
            tz=tz,
        )
        normalized[(day, _DAY_SAMPLE, "awakening_type")] = _get_value(
            source_row, columns, f"awakening_type_d{day_number}"
        )
        normalized[(day, _DAY_SAMPLE, "mismatch_summary")] = _get_value(
            source_row, columns, f"sample_mismatches_d{day_number}"
        )

        for sample in samples.get(day_number, []):
            normalized[(day, sample, "sampling_time")] = _combine_date_time(
                date,
                _get_value(
                    source_row,
                    columns,
                    f"sampling_time_d{day_number}_{sample}",
                ),
                tz=tz,
            )
            normalized[(day, sample, "barcode")] = _get_value(
                source_row, columns, f"sample_barcode_d{day_number}_{sample}"
            )
            normalized[(day, sample, "recorded_sample")] = _get_value(
                source_row, columns, f"sample_scanned_d{day_number}_{sample}"
            )
    return normalized


def _column_order(
    days: list[int], samples: dict[int, list[str]]
) -> list[tuple[str, str, str]]:
    columns: list[tuple[str, str, str]] = []
    for day_number in days:
        day = f"D{day_number}"
        columns.extend((day, _DAY_SAMPLE, variable) for variable in _DAY_VARIABLES)
        for sample in samples.get(day_number, []):
            columns.extend((day, sample, variable) for variable in _SAMPLE_VARIABLES)
    return columns


def _discover_layout(columns) -> tuple[list[int], dict[int, list[str]]]:
    days: list[int] = []
    samples: dict[int, list[str]] = {}
    for column in columns:
        if match := _DATE_COLUMN.fullmatch(str(column)):
            day = int(match.group("day"))
            if day not in days:
                days.append(day)
        if match := _SAMPLE_COLUMN.fullmatch(str(column)):
            day = int(match.group("day"))
            sample = match.group("sample")
            if sample.lower() == _DAY_SAMPLE:
                raise SchemaError(f"Sample label {_DAY_SAMPLE!r} is reserved.")
            samples.setdefault(day, [])
            if sample not in samples[day]:
                samples[day].append(sample)
    return sorted(days), {
        day: sorted(day_samples, key=_natural_key)
        for day, day_samples in samples.items()
    }


def _natural_key(value: str) -> list[tuple[int, object]]:
    parts = re.split(r"(\d+)", value)
    return [
        (0, int(part)) if part.isdigit() else (1, part.casefold()) for part in parts
    ]


def _set_string_dtypes(data: pd.DataFrame) -> pd.DataFrame:
    for column in data.columns:
        if column[-1] in {
            "awakening_type",
            "mismatch_summary",
            "barcode",
            "recorded_sample",
        }:
            data[column] = pd.array(data[column], dtype="string")
    return data


def _case_insensitive_columns(data: pd.DataFrame) -> dict[str, str]:
    lowered = [str(column).lower() for column in data.columns]
    if len(lowered) != len(set(lowered)):
        raise SchemaError("Study results contain columns that differ only by case.")
    return dict(zip(lowered, data.columns, strict=True))


def _required_column(columns: dict[str, str], name: str) -> str:
    try:
        return columns[name.lower()]
    except KeyError as exc:
        raise SchemaError(
            f"Study results are missing required column: {name!r}"
        ) from exc


def _get_value(source_row: dict, columns: dict[str, str], name: str) -> str | None:
    column = columns.get(name.lower())
    if column is None:
        return None
    return _clean_string(source_row[column])


def _clean_string(value) -> str | None:
    if value is None or pd.isna(value):
        return None
    value = str(value).strip()
    return value or None


def _parse_date(value: str | None, *, tz: str) -> pd.Timestamp:
    if value is None:
        return pd.NaT
    try:
        return pd.Timestamp(value).tz_localize(
            tz, ambiguous="raise", nonexistent="raise"
        )
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"Invalid study date: {value!r}") from exc


def _combine_date_time(
    date: pd.Timestamp, value: str | None, *, tz: str
) -> pd.Timestamp:
    if pd.isna(date) or value is None:
        return pd.NaT
    try:
        midnight = pd.Timestamp(date.date())
        local = midnight + pd.to_timedelta(value)
        return local.tz_localize(tz, ambiguous="raise", nonexistent="raise")
    except (TypeError, ValueError) as exc:
        raise SchemaError(f"Invalid local time for timezone {tz!r}: {value!r}") from exc
