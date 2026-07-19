"""Merge laboratory saliva measurements with CARWatch study results."""

from __future__ import annotations

from typing import Literal, TypeAlias

import pandas as pd

from carwatch.exceptions import MergeError, SchemaError

MatchMethod: TypeAlias = Literal["auto", "barcode", "scanned_sample", "expected_sample"]

_RESULT_INDEX = ["study", "participant", "day", "sample"]
_SALIVA_REQUIRED_INDEX = ["participant", "sample"]


def merge_saliva(
    study_results: pd.DataFrame,
    saliva: pd.DataFrame,
    *,
    match_by: MatchMethod = "auto",
    allow_unmatched: bool = True,
) -> pd.DataFrame:
    """Merge laboratory values onto their expected CARWatch sample positions.

    Laboratory samples are identified by their physical tube label or barcode.
    CARWatch results are indexed by the expected sampling position. In automatic
    mode, matching therefore uses the barcode first and ``sample_scanned`` as a
    fallback. After matching, the laboratory value is indexed by the expected
    sample and inherits its sampling timestamp.

    Parameters
    ----------
    study_results
        Normalized Study Manager results returned by
        :func:`carwatch.io.load_study_results`.
    saliva
        Normalized laboratory data returned by
        :func:`carwatch.io.load_saliva`.
    match_by
        Matching strategy. ``"expected_sample"`` is an explicit opt-out from
        swap correction for laboratory data that have already been corrected.
    allow_unmatched
        If ``True``, retain unmatched expected samples with missing laboratory
        values. If ``False``, unmatched expected samples or unused laboratory
        measurements raise :class:`~carwatch.exceptions.MergeError`.

    Returns
    -------
    pandas.DataFrame
        Study results enriched with laboratory values and merge provenance.

    """
    _validate_inputs(study_results, saliva, match_by)
    result_rows = study_results.reset_index().copy()
    saliva_rows = saliva.reset_index().copy()
    result_rows["_result_row"] = range(len(result_rows))
    saliva_rows["_saliva_row"] = range(len(saliva_rows))

    saliva_rows = _complete_context(result_rows, saliva_rows)
    saliva_rows = saliva_rows.rename(
        columns={"sample": "saliva_sample", "barcode": "saliva_barcode"}
    )
    context = ["study", "participant", "day"]
    lookups = _build_lookups(saliva_rows, context=context, match_by=match_by)

    matches: dict[int, tuple[int, str]] = {}
    used_saliva: dict[int, int] = {}
    for row in result_rows.to_dict(orient="records"):
        matched = _find_match(row, context=context, lookups=lookups, match_by=match_by)
        if matched is None:
            continue
        saliva_row, method = matched
        saliva_id = int(saliva_row["_saliva_row"])
        result_id = int(row["_result_row"])
        if saliva_id in used_saliva:
            previous = used_saliva[saliva_id]
            raise MergeError(
                "One physical saliva tube maps to multiple expected samples: "
                f"result rows {previous} and {result_id}."
            )
        used_saliva[saliva_id] = result_id
        matches[result_id] = (saliva_id, method)

    output = _attach_saliva(result_rows, saliva_rows, matches, context=context)
    unmatched_results = output["merge_status"].eq("unmatched")
    unused_saliva = set(saliva_rows["_saliva_row"]) - set(used_saliva)
    if not allow_unmatched and (unmatched_results.any() or unused_saliva):
        raise MergeError(
            f"Merge left {int(unmatched_results.sum())} expected samples unmatched and "
            f"{len(unused_saliva)} laboratory samples unused."
        )
    return output.set_index(_RESULT_INDEX).drop(columns="_result_row")


def _validate_inputs(
    study_results: pd.DataFrame, saliva: pd.DataFrame, match_by: str
) -> None:
    if not isinstance(study_results, pd.DataFrame) or not isinstance(
        saliva, pd.DataFrame
    ):
        raise TypeError("'study_results' and 'saliva' must be pandas DataFrames.")
    if match_by not in {"auto", "barcode", "scanned_sample", "expected_sample"}:
        raise ValueError(
            "'match_by' must be one of {'auto', 'barcode', 'scanned_sample', 'expected_sample'}."
        )
    missing_result = set(_RESULT_INDEX).difference(study_results.index.names)
    if missing_result:
        raise SchemaError(
            f"Study results index is missing levels: {sorted(missing_result)}"
        )
    missing_saliva = set(_SALIVA_REQUIRED_INDEX).difference(saliva.index.names)
    if missing_saliva:
        raise SchemaError(f"Saliva index is missing levels: {sorted(missing_saliva)}")
    if study_results.index.has_duplicates or saliva.index.has_duplicates:
        raise SchemaError(
            "Input indices must uniquely identify study and physical saliva samples."
        )
    if "sample_scanned" not in study_results:
        raise SchemaError("Study results are missing the 'sample_scanned' column.")


def _complete_context(result: pd.DataFrame, saliva: pd.DataFrame) -> pd.DataFrame:
    saliva = saliva.copy()
    if "study" not in saliva:
        saliva = _infer_context_column(
            result,
            saliva,
            column="study",
            group_cols=["participant"],
        )
    if "day" not in saliva:
        saliva = _infer_context_column(
            result,
            saliva,
            column="day",
            group_cols=["study", "participant"],
        )
    return saliva


def _infer_context_column(
    result: pd.DataFrame,
    saliva: pd.DataFrame,
    *,
    column: str,
    group_cols: list[str],
) -> pd.DataFrame:
    candidates = result[group_cols + [column]].drop_duplicates()
    ambiguous = candidates.duplicated(group_cols, keep=False)
    relevant = saliva[group_cols].drop_duplicates()
    ambiguous_candidates = candidates.loc[ambiguous, group_cols].drop_duplicates()
    if not relevant.merge(ambiguous_candidates, on=group_cols, how="inner").empty:
        raise MergeError(
            f"Saliva data do not contain {column!r}, and it cannot be inferred because multiple values exist."
        )
    return saliva.merge(candidates, on=group_cols, how="left", validate="many_to_one")


def _build_lookups(
    saliva: pd.DataFrame,
    *,
    context: list[str],
    match_by: MatchMethod,
) -> dict[str, dict[tuple, dict]]:
    lookups: dict[str, dict[tuple, dict]] = {}
    if match_by in {"auto", "barcode"}:
        lookups["barcode"] = _build_lookup(
            saliva,
            context=context,
            identifier="saliva_barcode",
        )
    if match_by in {"auto", "scanned_sample", "expected_sample"}:
        lookups["sample"] = _build_lookup(
            saliva,
            context=context,
            identifier="saliva_sample",
        )
    return lookups


def _build_lookup(
    saliva: pd.DataFrame, *, context: list[str], identifier: str
) -> dict[tuple, dict]:
    if identifier not in saliva:
        return {}
    lookup: dict[tuple, dict] = {}
    for row in saliva.to_dict(orient="records"):
        value = row.get(identifier)
        if _is_missing(value):
            continue
        key = (*[row[column] for column in context], str(value))
        if key in lookup:
            raise MergeError(
                f"Laboratory identifier is not unique within a sampling day: {key}"
            )
        lookup[key] = row
    return lookup


def _find_match(
    row: dict,
    *,
    context: list[str],
    lookups: dict[str, dict[tuple, dict]],
    match_by: MatchMethod,
) -> tuple[dict, str] | None:
    context_values = tuple(row[column] for column in context)
    if match_by in {"auto", "barcode"} and not _is_missing(row.get("barcode")):
        key = (*context_values, str(row["barcode"]))
        if matched := lookups.get("barcode", {}).get(key):
            return matched, "barcode"
        if match_by == "barcode":
            return None

    if match_by in {"auto", "scanned_sample"}:
        identifier = row.get("sample_scanned") or row["sample"]
        method = "scanned_sample"
    elif match_by == "expected_sample":
        identifier = row["sample"]
        method = "expected_sample"
    else:
        return None
    key = (*context_values, str(identifier))
    matched = lookups.get("sample", {}).get(key)
    return (matched, method) if matched is not None else None


def _attach_saliva(
    result: pd.DataFrame,
    saliva: pd.DataFrame,
    matches: dict[int, tuple[int, str]],
    *,
    context: list[str],
) -> pd.DataFrame:
    saliva_lookup = saliva.set_index("_saliva_row", drop=False)
    excluded = {*context, "_saliva_row"}
    saliva_columns = [column for column in saliva.columns if column not in excluded]
    output_columns = _output_column_names(result, saliva_columns)
    output = result.copy()
    for source, target in output_columns.items():
        output[target] = pd.NA

    methods: list[str | None] = []
    statuses: list[str] = []
    corrections: list[bool] = []
    for result_id in output["_result_row"]:
        matched = matches.get(int(result_id))
        if matched is None:
            methods.append(None)
            statuses.append("unmatched")
            corrections.append(False)
            continue
        saliva_id, method = matched
        saliva_row = saliva_lookup.loc[saliva_id]
        mask = output["_result_row"].eq(result_id)
        for source, target in output_columns.items():
            output.loc[mask, target] = saliva_row[source]
        methods.append(method)
        statuses.append("matched")
        mismatch = bool(output.loc[mask, "sample_mismatch"].iloc[0])
        corrections.append(mismatch and method in {"barcode", "scanned_sample"})

    output["match_method"] = pd.array(methods, dtype="string")
    output["merge_status"] = pd.array(statuses, dtype="string")
    output["mismatch_corrected"] = corrections
    return output


def _output_column_names(
    result: pd.DataFrame, saliva_columns: list[str]
) -> dict[str, str]:
    output: dict[str, str] = {}
    for column in saliva_columns:
        if column in {"saliva_sample", "saliva_barcode"}:
            output[column] = column
        elif column in result:
            output[column] = f"saliva_{column}"
        else:
            output[column] = column
    return output


def _is_missing(value) -> bool:
    return value is None or pd.isna(value) or str(value).strip() == ""
