from io import StringIO

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import carwatch as cw
from carwatch.exceptions import MergeError, SchemaError


SUMMARY = """Participant ID,date_D1,awakening_time_D1_app,awakening_type_D1,sampling_time_D1_B1,sample_barcode_D1_B1,sample_scanned_D1_B1,sampling_time_D1_B2,sample_barcode_D1_B2,sample_scanned_D1_B2,sampling_time_D1_B3,sample_barcode_D1_B3,sample_scanned_D1_B3,sampling_time_D1_B4,sample_barcode_D1_B4,sample_scanned_D1_B4
02,2025-05-15,06:00:00,self-report,06:00:00,0010101,B1,06:30:00,0010103,B3,06:45:00,0010102,B2,07:00:00,0010104,B4
"""

SALIVA = """subject,sample,cortisol
02,B1,1.0
02,B2,2.0
02,B3,3.0
02,B4,4.0
"""


def _summary_events(tmp_path):
    path = tmp_path / "summary.csv"
    path.write_text(SUMMARY)
    summary = cw.io.load_study_results(path)
    return cw.logs.extract_sample_events_from_summary(summary)


def _saliva(tmp_path, content=SALIVA):
    path = tmp_path / "saliva.csv"
    path.write_text(content)
    return cw.io.load_saliva(path)


def test_merge_saliva_corrects_swaps_from_summary(tmp_path):
    result = cw.merge_saliva(_summary_events(tmp_path), _saliva(tmp_path))

    assert result.index.names == ["participant", "day", "sample"]
    assert result["cortisol"].tolist() == [1.0, 3.0, 2.0, 4.0]
    assert result["match_method"].tolist() == ["sample_scanned"] * 4
    assert result["mismatch_corrected"].tolist() == [False, True, True, False]
    assert "saliva_sample" not in result
    assert "barcode" not in result
    assert "time_min" in result


def test_merge_saliva_supports_raw_log_sample_events(tmp_path):
    path = tmp_path / "carwatch_logs_02.csv"
    path.write_text(
        '1747282435999;local;barcode_scanned;{"sample_expected":"B1","sample_scanned":"B1"}\n'
        '1747284231091;local;barcode_scanned;{"sample_expected":"B2","sample_scanned":"B3"}\n'
    )
    raw_logs = cw.io.load_logs(path)
    events = cw.logs.extract_sample_events_from_raw_logs(raw_logs)

    result = cw.merge_saliva(events, _saliva(tmp_path))

    assert result.index.names == ["participant", "date", "sample"]
    assert result["cortisol"].tolist() == [1.0, 3.0]
    assert result["mismatch_corrected"].tolist() == [False, True]


def test_merge_saliva_can_skip_swap_correction(tmp_path):
    result = cw.merge_saliva(
        _summary_events(tmp_path),
        _saliva(tmp_path),
        correct_swaps=False,
    )

    assert result["cortisol"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert result["match_method"].tolist() == ["expected_sample"] * 4
    assert not result["mismatch_corrected"].any()


def test_merge_saliva_uses_expected_sample_if_scan_is_missing(tmp_path):
    events = _summary_events(tmp_path)
    events.loc[("02", "D1", "B1"), "sample_scanned"] = pd.NA

    result = cw.merge_saliva(events, _saliva(tmp_path))

    assert result.loc[("02", "D1", "B1"), "cortisol"] == 1.0
    assert result.loc[("02", "D1", "B1"), "match_method"] == "expected_sample"


def test_merge_saliva_distinguishes_missing_value_from_unmatched_sample(tmp_path):
    content = SALIVA.replace("02,B1,1.0", "02,B1,").replace("02,B4,4.0\n", "")

    result = cw.merge_saliva(_summary_events(tmp_path), _saliva(tmp_path, content))

    assert result.loc[("02", "D1", "B1"), "merge_status"] == "matched"
    assert pd.isna(result.loc[("02", "D1", "B1"), "cortisol"])
    assert result.loc[("02", "D1", "B4"), "merge_status"] == "unmatched"


def test_merge_saliva_can_require_complete_matching(tmp_path):
    content = SALIVA.replace("02,B4,4.0\n", "")

    with pytest.raises(MergeError, match="unmatched"):
        cw.merge_saliva(
            _summary_events(tmp_path),
            _saliva(tmp_path, content),
            allow_unmatched=False,
        )


def test_merge_saliva_rejects_non_bijective_swaps(tmp_path):
    events = _summary_events(tmp_path)
    events.loc[("02", "D1", "B2"), "sample_scanned"] = "B1"

    with pytest.raises(MergeError, match="same physical saliva tube"):
        cw.merge_saliva(events, _saliva(tmp_path))


def test_merge_saliva_rejects_duplicate_sampling_positions(tmp_path):
    events = _summary_events(tmp_path).reset_index()
    events = pd.concat([events, events.iloc[[0]]], ignore_index=True)

    with pytest.raises(SchemaError, match="duplicate sampling positions"):
        cw.merge_saliva(events, _saliva(tmp_path))


def test_merge_saliva_rejects_incompatible_schemas(tmp_path):
    with pytest.raises(SchemaError, match="sample extractor"):
        cw.merge_saliva(pd.DataFrame({"sample": ["B1"]}), _saliva(tmp_path))
    with pytest.raises(SchemaError, match="index levels"):
        cw.merge_saliva(_summary_events(tmp_path), pd.DataFrame({"cortisol": [1.0]}))


def test_merge_saliva_does_not_mutate_inputs(tmp_path):
    events = _summary_events(tmp_path)
    saliva = _saliva(tmp_path)
    expected_events = events.copy(deep=True)
    expected_saliva = saliva.copy(deep=True)

    cw.merge_saliva(events, saliva)

    assert_frame_equal(events, expected_events)
    assert_frame_equal(saliva, expected_saliva)


def test_merge_saliva_rejects_non_numeric_measurements(tmp_path):
    events = _summary_events(tmp_path)
    saliva = pd.read_csv(StringIO(SALIVA), dtype="string").set_index(
        ["subject", "sample"]
    )

    with pytest.raises(SchemaError, match="numeric"):
        cw.merge_saliva(events, saliva)
