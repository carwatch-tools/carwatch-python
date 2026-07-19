import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import carwatch as cw
from carwatch.exceptions import MergeError, SchemaError


def study_results(days=("D1",)):
    index = pd.MultiIndex.from_product(
        [["study"], ["02"], days, ["B1", "B2", "B3", "B4"]],
        names=["study", "participant", "day", "sample"],
    )
    result = pd.DataFrame(index=index)
    result["sample_scanned"] = ["B1", "B3", "B2", "B4"] * len(days)
    result["barcode"] = ["0010101", "0010103", "0010102", "0010104"] * len(days)
    result["sample_mismatch"] = [False, True, True, False] * len(days)
    result["sampling_time"] = pd.date_range("2025-05-15 06:00", periods=len(result), freq="15min", tz="Europe/Berlin")
    result["time"] = [0, 30, 45, 60] * len(days)
    return result


def saliva(*, with_barcode=True, with_day=True):
    levels = [["02"]]
    names = ["participant"]
    if with_day:
        levels.append(["D1"])
        names.append("day")
    levels.append(["B1", "B2", "B3", "B4"])
    names.append("sample")
    result = pd.DataFrame(
        {"cortisol": [1.0, 2.0, 3.0, 4.0]},
        index=pd.MultiIndex.from_product(levels, names=names),
    )
    if with_barcode:
        result["barcode"] = ["0010101", "0010102", "0010103", "0010104"]
    return result


def test_merge_saliva_corrects_swapped_samples_by_barcode():
    result = cw.merge_saliva(study_results(), saliva())

    assert result["cortisol"].tolist() == [1.0, 3.0, 2.0, 4.0]
    assert result["saliva_sample"].tolist() == ["B1", "B3", "B2", "B4"]
    assert result["match_method"].tolist() == ["barcode"] * 4
    assert result["mismatch_corrected"].tolist() == [False, True, True, False]


def test_merge_saliva_falls_back_to_scanned_sample():
    result = cw.merge_saliva(study_results(), saliva(with_barcode=False))

    assert result["cortisol"].tolist() == [1.0, 3.0, 2.0, 4.0]
    assert result["match_method"].tolist() == ["scanned_sample"] * 4


def test_merge_saliva_can_use_already_corrected_expected_samples():
    result = cw.merge_saliva(study_results(), saliva(with_barcode=False), match_by="expected_sample")

    assert result["cortisol"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert not result["mismatch_corrected"].any()


def test_merge_saliva_infers_single_missing_day():
    result = cw.merge_saliva(study_results(), saliva(with_day=False))

    assert result.index.get_level_values("day").unique().tolist() == ["D1"]
    assert result["cortisol"].tolist() == [1.0, 3.0, 2.0, 4.0]


def test_merge_saliva_rejects_ambiguous_missing_day():
    with pytest.raises(MergeError, match="cannot be inferred"):
        cw.merge_saliva(study_results(days=("D1", "D2")), saliva(with_day=False))


def test_merge_saliva_retains_unmatched_expected_samples():
    laboratory = saliva().drop(("02", "D1", "B4"))
    result = cw.merge_saliva(study_results(), laboratory)

    assert result["merge_status"].tolist() == ["matched", "matched", "matched", "unmatched"]
    assert pd.isna(result.iloc[-1]["cortisol"])


def test_merge_saliva_can_require_complete_matching():
    laboratory = saliva().drop(("02", "D1", "B4"))

    with pytest.raises(MergeError, match="unmatched"):
        cw.merge_saliva(study_results(), laboratory, allow_unmatched=False)


def test_merge_saliva_rejects_duplicate_barcodes():
    laboratory = saliva()
    laboratory["barcode"] = ["0010101", "0010101", "0010103", "0010104"]

    with pytest.raises(MergeError, match="not unique"):
        cw.merge_saliva(study_results(), laboratory)


def test_merge_saliva_rejects_non_bijective_swap():
    results = study_results()
    results["sample_scanned"] = ["B1", "B3", "B3", "B4"]
    results["barcode"] = pd.NA

    with pytest.raises(MergeError, match="multiple expected samples"):
        cw.merge_saliva(results, saliva(with_barcode=False))


def test_merge_saliva_does_not_mutate_inputs():
    results = study_results()
    laboratory = saliva()
    expected_results = results.copy(deep=True)
    expected_saliva = laboratory.copy(deep=True)

    cw.merge_saliva(results, laboratory)

    assert_frame_equal(results, expected_results)
    assert_frame_equal(laboratory, expected_saliva)


def test_merge_saliva_validates_schemas():
    with pytest.raises(SchemaError):
        cw.merge_saliva(pd.DataFrame(), saliva())
    with pytest.raises(SchemaError):
        cw.merge_saliva(study_results(), pd.DataFrame())
    with pytest.raises(ValueError, match="match_by"):
        cw.merge_saliva(study_results(), saliva(), match_by="invalid")
