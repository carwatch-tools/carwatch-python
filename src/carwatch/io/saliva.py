"""Load laboratory saliva measurements from CSV files."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Literal, TypeAlias

import pandas as pd

from carwatch.exceptions import SchemaError

SalivaFormat: TypeAlias = Literal["auto", "long", "wide"]
ColumnMapping: TypeAlias = Mapping[str, str]

_SAMPLE_COLUMN = re.compile(r"^[A-Za-z]+\d+$")


def load_saliva(
    path: str | Path,
    *,
    format: SalivaFormat = "auto",
    participant_col: str = "participant",
    study_col: str | None = None,
    day_col: str | None = "day",
    sample_col: str = "sample",
    barcode_col: str | None = None,
    value_cols: str | Sequence[str] | None = None,
    sample_columns: Mapping[str, str] | Sequence[str] | None = None,
    value_name: str | None = None,
    participant_map: ColumnMapping | None = None,
    day_map: ColumnMapping | None = None,
    sample_map: ColumnMapping | None = None,
) -> pd.DataFrame:
    """Load saliva measurements in long or wide CSV format.

    Parameters
    ----------
    path
        Path to a CSV file containing laboratory measurements.
    format
        Input format. ``"auto"`` selects long format if ``sample_col`` is
        present and otherwise attempts conservative wide-format detection.
    participant_col, study_col, day_col, sample_col, barcode_col
        Source column names for identifiers. Study and day are optional.
    value_cols
        Measurement columns in long format. If omitted, columns that can be
        losslessly converted to numbers are selected.
    sample_columns
        Wide-format sample columns. A mapping maps source columns to canonical
        physical tube labels; a sequence uses the source labels unchanged.
    value_name
        Name of the measurement column created from wide-format input.
    participant_map, day_map, sample_map
        Optional mappings from source identifiers to canonical identifiers.

    Returns
    -------
    pandas.DataFrame
        Measurements indexed by participant, optional study/day, and physical
        tube sample.

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Saliva file does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("Saliva measurements must be supplied as a CSV file.")
    if format not in {"auto", "long", "wide"}:
        raise ValueError("'format' must be one of {'auto', 'long', 'wide'}.")

    data = pd.read_csv(path, dtype="string", keep_default_na=False)
    if data.empty:
        raise SchemaError("Saliva file does not contain any measurements.")
    _require_columns(data, [participant_col])

    selected_format = _select_format(
        data,
        format=format,
        sample_col=sample_col,
        participant_col=participant_col,
        study_col=study_col,
        day_col=day_col,
        barcode_col=barcode_col,
        sample_columns=sample_columns,
    )
    if selected_format == "long":
        result = _normalize_long(
            data,
            participant_col=participant_col,
            study_col=study_col,
            day_col=day_col,
            sample_col=sample_col,
            barcode_col=barcode_col,
            value_cols=value_cols,
        )
    else:
        result = _normalize_wide(
            data,
            participant_col=participant_col,
            study_col=study_col,
            day_col=day_col,
            barcode_col=barcode_col,
            sample_columns=sample_columns,
            value_name=value_name,
        )

    result = _apply_identifier_map(result, "participant", participant_map)
    result = _apply_identifier_map(result, "day", day_map)
    result = _apply_identifier_map(result, "sample", sample_map)
    index_cols = [
        column
        for column in ["study", "participant", "day", "sample"]
        if column in result
    ]
    result = result.set_index(index_cols)
    if result.index.has_duplicates:
        duplicates = result.index[result.index.duplicated()].unique().tolist()
        raise SchemaError(
            f"Saliva data contain duplicate physical samples: {duplicates}"
        )
    return result


def _normalize_long(
    data: pd.DataFrame,
    *,
    participant_col: str,
    study_col: str | None,
    day_col: str | None,
    sample_col: str,
    barcode_col: str | None,
    value_cols: str | Sequence[str] | None,
) -> pd.DataFrame:
    identifier_cols = _present_identifier_columns(
        data,
        participant_col=participant_col,
        study_col=study_col,
        day_col=day_col,
        sample_col=sample_col,
        barcode_col=barcode_col,
    )
    _require_columns(data, [participant_col, sample_col])
    values = _normalize_value_cols(value_cols)
    if values is None:
        values = _infer_numeric_columns(data, excluded=set(identifier_cols.values()))
    _require_columns(data, values)
    if not values:
        raise SchemaError("No numeric saliva measurement columns could be inferred.")

    result = data.rename(
        columns={source: target for target, source in identifier_cols.items()}
    ).copy()
    for column in values:
        result[column] = _to_numeric(result[column], column)
    result = _replace_empty_strings(result)
    return result


def _normalize_wide(
    data: pd.DataFrame,
    *,
    participant_col: str,
    study_col: str | None,
    day_col: str | None,
    barcode_col: str | None,
    sample_columns: Mapping[str, str] | Sequence[str] | None,
    value_name: str | None,
) -> pd.DataFrame:
    if barcode_col is not None:
        raise SchemaError(
            "A single 'barcode_col' is not supported for wide saliva data."
        )
    if value_name is None or not value_name.strip():
        raise SchemaError("'value_name' is required for wide saliva data.")

    sample_mapping = _normalize_sample_columns(data, sample_columns)
    _require_columns(data, sample_mapping)
    identifiers = _present_identifier_columns(
        data,
        participant_col=participant_col,
        study_col=study_col,
        day_col=day_col,
    )
    id_source_cols = list(identifiers.values())
    result = data[id_source_cols + list(sample_mapping)].melt(
        id_vars=id_source_cols,
        value_vars=list(sample_mapping),
        var_name="_source_sample",
        value_name=value_name,
    )
    result["sample"] = result.pop("_source_sample").map(sample_mapping)
    result = result.rename(
        columns={source: target for target, source in identifiers.items()}
    )
    result[value_name] = _to_numeric(result[value_name], value_name)
    return _replace_empty_strings(result)


def _select_format(
    data: pd.DataFrame,
    *,
    format: SalivaFormat,
    sample_col: str,
    participant_col: str,
    study_col: str | None,
    day_col: str | None,
    barcode_col: str | None,
    sample_columns: Mapping[str, str] | Sequence[str] | None,
) -> Literal["long", "wide"]:
    if format != "auto":
        return format
    if sample_col in data:
        return "long"
    if sample_columns is not None:
        return "wide"
    excluded = {participant_col, study_col, day_col, barcode_col, None}
    candidates = [
        column
        for column in data.columns
        if column not in excluded and _SAMPLE_COLUMN.fullmatch(column)
    ]
    if candidates:
        return "wide"
    raise SchemaError(
        "Could not infer saliva format. Specify 'format' and the corresponding columns explicitly."
    )


def _present_identifier_columns(
    data: pd.DataFrame,
    *,
    participant_col: str,
    study_col: str | None = None,
    day_col: str | None = None,
    sample_col: str | None = None,
    barcode_col: str | None = None,
) -> dict[str, str]:
    result = {"participant": participant_col}
    optional = {
        "study": study_col,
        "day": day_col,
        "sample": sample_col,
        "barcode": barcode_col,
    }
    for canonical, source in optional.items():
        if source is not None and source in data:
            result[canonical] = source
    return result


def _normalize_sample_columns(
    data: pd.DataFrame, sample_columns: Mapping[str, str] | Sequence[str] | None
) -> dict[str, str]:
    if sample_columns is None:
        inferred = [
            column for column in data.columns if _SAMPLE_COLUMN.fullmatch(column)
        ]
        if not inferred:
            raise SchemaError("No wide-format sample columns could be inferred.")
        return {column: column for column in inferred}
    if isinstance(sample_columns, Mapping):
        return {str(source): str(target) for source, target in sample_columns.items()}
    if isinstance(sample_columns, str):
        sample_columns = [sample_columns]
    return {str(column): str(column) for column in sample_columns}


def _normalize_value_cols(value_cols: str | Sequence[str] | None) -> list[str] | None:
    if value_cols is None:
        return None
    if isinstance(value_cols, str):
        return [value_cols]
    return list(value_cols)


def _infer_numeric_columns(data: pd.DataFrame, *, excluded: set[str]) -> list[str]:
    numeric: list[str] = []
    for column in data.columns:
        if column in excluded:
            continue
        non_empty = data[column].str.strip().ne("")
        if (
            non_empty.any()
            and pd.to_numeric(data.loc[non_empty, column], errors="coerce")
            .notna()
            .all()
        ):
            numeric.append(column)
    return numeric


def _to_numeric(data: pd.Series, column: str) -> pd.Series:
    cleaned = data.mask(data.str.strip().eq(""), pd.NA)
    converted = pd.to_numeric(cleaned, errors="coerce")
    invalid = cleaned.notna() & converted.isna()
    if invalid.any():
        values = cleaned.loc[invalid].unique().tolist()
        raise SchemaError(
            f"Column {column!r} contains non-numeric measurements: {values}"
        )
    return converted


def _replace_empty_strings(data: pd.DataFrame) -> pd.DataFrame:
    string_columns = data.select_dtypes(include=["string", "object"]).columns
    data[string_columns] = data[string_columns].replace(r"^\s*$", pd.NA, regex=True)
    return data


def _apply_identifier_map(
    data: pd.DataFrame, column: str, mapping: ColumnMapping | None
) -> pd.DataFrame:
    if column not in data:
        return data
    data[column] = data[column].astype("string").str.strip()
    if mapping is not None:
        normalized_mapping = {
            str(source): str(target) for source, target in mapping.items()
        }
        data[column] = data[column].map(
            lambda value: normalized_mapping.get(value, value), na_action="ignore"
        )
    if data[column].isna().any():
        raise SchemaError(f"Identifier column {column!r} contains missing values.")
    return data


def _require_columns(data: pd.DataFrame, columns: Sequence[str]) -> None:
    missing = [column for column in columns if column not in data]
    if missing:
        raise SchemaError(f"Saliva data are missing required columns: {missing}")
