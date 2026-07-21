"""Load long-format saliva measurements from CSV files."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from carwatch.exceptions import SchemaError

_IDENTIFIER_COLUMNS = ["participant", "sample"]


def load_saliva(path: str | Path, *, saliva_type: str = "cortisol") -> pd.DataFrame:
    """Load saliva measurements exported in BioPsyKit long format.

    The CSV file must contain exactly three columns: ``participant``, ``sample``,
    and the saliva biomarker specified by ``saliva_type``. Measurement values
    must be numeric, but may be missing.

    Parameters
    ----------
    path
        Path to the saliva CSV file.
    saliva_type
        Name of the saliva biomarker column. Default: ``"cortisol"``.

    Returns
    -------
    pandas.DataFrame
        Measurements indexed by ``participant`` and ``sample``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ValueError
        If ``path`` is not a CSV file or ``saliva_type`` is empty.
    carwatch.exceptions.SchemaError
        If the file does not follow the expected long-format schema.

    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Saliva file does not exist: {path}")
    if path.suffix.lower() != ".csv":
        raise ValueError("Saliva measurements must be supplied as a CSV file.")
    if not isinstance(saliva_type, str) or not saliva_type.strip():
        raise ValueError("'saliva_type' must be a non-empty string.")

    saliva_type = saliva_type.strip()
    expected_columns = [*_IDENTIFIER_COLUMNS, saliva_type]
    data = pd.read_csv(path, dtype="string")
    if data.empty:
        raise SchemaError("Saliva file does not contain any measurements.")

    _validate_columns(data, expected_columns)
    data = data[expected_columns].copy()
    _normalize_identifiers(data)
    data[saliva_type] = _to_numeric(data[saliva_type], saliva_type)

    result = data.set_index(_IDENTIFIER_COLUMNS)
    if result.index.has_duplicates:
        duplicates = result.index[result.index.duplicated()].unique().tolist()
        raise SchemaError(
            f"Saliva data contain duplicate participant/sample pairs: {duplicates}"
        )
    return result


def _validate_columns(data: pd.DataFrame, expected: list[str]) -> None:
    missing = [column for column in expected if column not in data]
    unexpected = [column for column in data if column not in expected]
    if missing or unexpected:
        details = []
        if missing:
            details.append(f"missing columns: {missing}")
        if unexpected:
            details.append(f"unexpected columns: {unexpected}")
        raise SchemaError(
            f"Saliva data must contain exactly {expected}; " + "; ".join(details) + "."
        )


def _normalize_identifiers(data: pd.DataFrame) -> None:
    for column in _IDENTIFIER_COLUMNS:
        data[column] = data[column].str.strip().replace("", pd.NA)
        if data[column].isna().any():
            raise SchemaError(f"Identifier column {column!r} contains missing values.")


def _to_numeric(data: pd.Series, column: str) -> pd.Series:
    converted = pd.to_numeric(data, errors="coerce")
    invalid = data.notna() & converted.isna()
    if invalid.any():
        values = data.loc[invalid].unique().tolist()
        raise SchemaError(
            f"Column {column!r} contains non-numeric measurements: {values}"
        )
    return converted
