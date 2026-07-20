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
_INDEX = ["participant", "day", "sample"]
_OUTPUT_COLUMNS = [
    "date",
    "awakening_time",
    "awakening_time_google_fit",
    "awakening_type",
    "sampling_time",
    "time",
    "barcode",
    "sample_scanned",
    "sample_mismatch",
    "mismatch_summary",
    "observed",
]


def load_study_results(path: str | Path, *, tz: str = "Europe/Berlin") -> pd.DataFrame:
    """Load and normalize a CARWatch Study Manager CSV export.

    The Study Manager stores one participant per row and encodes days and
    samples in column names. This function reshapes that representation into
    one row per expected sample. The expected sample is stored in the
    ``sample`` index level; the physical tube is stored in
    ``sample_scanned``.

    Parameters
    ----------
    path
        Path to the Study Manager CSV export.
    tz
        Timezone used to combine the exported dates and local times.

    Returns
    -------
    pandas.DataFrame
        Normalized results indexed by ``participant``, ``day``, and expected
        ``sample``.

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

    rows: list[dict] = []
    for source_row in data.to_dict(orient="records"):
        participant = _clean_string(source_row[participant_col])
        if participant is None:
            raise SchemaError("Participant ID must not be empty.")
        for day_number in days:
            rows.extend(
                _normalize_day(
                    source_row,
                    columns=columns,
                    samples=samples.get(day_number, []),
                    participant=participant,
                    day_number=day_number,
                    tz=tz,
                )
            )

    result = pd.DataFrame(rows).set_index(_INDEX)
    if result.index.has_duplicates:
        duplicates = result.index[result.index.duplicated()].unique().tolist()
        raise SchemaError(f"Study results contain duplicate sample rows: {duplicates}")
    return result[_OUTPUT_COLUMNS]


def _normalize_day(
    source_row: dict,
    *,
    columns: dict[str, str],
    samples: list[str],
    participant: str,
    day_number: int,
    tz: str,
) -> list[dict]:
    day = f"D{day_number}"
    date_value = _clean_string(
        source_row[_required_column(columns, f"date_d{day_number}")]
    )
    date = _parse_date(date_value, tz=tz)
    awakening = _combine_date_time(
        date,
        _get_value(source_row, columns, f"awakening_time_d{day_number}_app"),
        tz=tz,
    )
    awakening_google_fit = _combine_date_time(
        date,
        _get_value(source_row, columns, f"awakening_time_d{day_number}_google_fit"),
        tz=tz,
    )
    awakening_type = _get_value(source_row, columns, f"awakening_type_d{day_number}")
    mismatch_summary = _get_value(
        source_row, columns, f"sample_mismatches_d{day_number}"
    )

    normalized: list[dict] = []
    for sample in samples:
        sampling_time = _combine_date_time(
            date,
            _get_value(source_row, columns, f"sampling_time_d{day_number}_{sample}"),
            tz=tz,
        )
        barcode = _get_value(
            source_row, columns, f"sample_barcode_d{day_number}_{sample}"
        )
        scanned = (
            _get_value(source_row, columns, f"sample_scanned_d{day_number}_{sample}")
            or sample
        )
        time = _minutes_between(sampling_time, awakening)
        observed = not (pd.isna(sampling_time) and barcode is None)
        normalized.append(
            {
                "participant": participant,
                "day": day,
                "sample": sample,
                "date": date,
                "awakening_time": awakening,
                "awakening_time_google_fit": awakening_google_fit,
                "awakening_type": awakening_type,
                "sampling_time": sampling_time,
                "time": time,
                "barcode": barcode,
                "sample_scanned": scanned,
                "sample_mismatch": sample != scanned,
                "mismatch_summary": mismatch_summary,
                "observed": observed,
            }
        )
    return normalized


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
            samples.setdefault(day, [])
            if sample not in samples[day]:
                samples[day].append(sample)
    return sorted(days), samples


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
        return pd.Timestamp(value).tz_localize(tz)
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


def _minutes_between(later: pd.Timestamp, earlier: pd.Timestamp) -> float:
    if pd.isna(later) or pd.isna(earlier):
        return float("nan")
    return (later - earlier).total_seconds() / 60
