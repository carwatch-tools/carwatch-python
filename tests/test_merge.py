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

    assert result.index.names == ["participant", "day", "scheduled_sample"]
    assert result["cortisol"].tolist() == [1.0, 3.0, 2.0, 4.0]
    assert result["mismatch_corrected"].tolist() == [False, True, True, False]
    assert result["sampling_event_recorded"].all()
    assert result["lab_value_available"].all()
    assert "saliva_sample" not in result
    assert "barcode" not in result
    assert "sample_id_source" not in result
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

    assert result.index.names == ["participant", "date", "scheduled_sample"]
    assert result["cortisol"].tolist() == [1.0, 3.0, 2.0, 4.0]
    assert result["sampling_event_recorded"].tolist() == [True, True, False, False]
    assert result["lab_value_available"].tolist() == [True, True, True, True]
    assert result["mismatch_corrected"].tolist() == [False, True, False, False]


def test_merge_saliva_can_skip_swap_correction(tmp_path):
    result = cw.merge_saliva(
        _summary_events(tmp_path),
        _saliva(tmp_path),
        correct_swaps=False,
    )

    assert result["cortisol"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert not result["mismatch_corrected"].any()


def test_merge_saliva_uses_scheduled_sample_if_recorded_sample_is_missing(tmp_path):
    events = _summary_events(tmp_path)
    events.loc[("02", "D1", "B1"), "recorded_sample"] = pd.NA

    result = cw.merge_saliva(events, _saliva(tmp_path))

    assert result.loc[("02", "D1", "B1"), "cortisol"] == 1.0


@pytest.fixture
def availability_cases(tmp_path):
    log_path = tmp_path / "carwatch_logs_02.csv"
    log_path.write_text(
        '1747282435999;local;barcode_scanned;{"sample_expected":"B1","sample_scanned":"B1"}\n'
        '1747284231091;local;barcode_scanned;{"sample_expected":"B2","sample_scanned":"B2"}\n'
    )
    saliva = _saliva(
        tmp_path,
        """subject,sample,cortisol
02,B1,1.0
02,B2,
02,B3,3.0
02,B4,
""",
    )
    events = cw.logs.extract_sample_events_from_raw_logs(cw.io.load_logs(log_path))
    return cw.merge_saliva(events, saliva).reset_index().set_index("scheduled_sample")


def test_merge_marks_recorded_event_with_lab_value(availability_cases):
    result = availability_cases.loc["B1"]

    assert bool(result["sampling_event_recorded"])
    assert bool(result["lab_value_available"])


def test_merge_marks_recorded_event_without_lab_value(availability_cases):
    result = availability_cases.loc["B2"]

    assert bool(result["sampling_event_recorded"])
    assert not bool(result["lab_value_available"])
    assert pd.isna(result["cortisol"])


def test_merge_marks_missing_event_with_lab_value(availability_cases):
    result = availability_cases.loc["B3"]

    assert not bool(result["sampling_event_recorded"])
    assert bool(result["lab_value_available"])
    assert result["cortisol"] == 3.0


def test_merge_marks_missing_event_without_lab_value(availability_cases):
    result = availability_cases.loc["B4"]

    assert not bool(result["sampling_event_recorded"])
    assert not bool(result["lab_value_available"])
    assert pd.isna(result["cortisol"])


def test_merge_preserves_missing_summary_event(tmp_path):
    events = _summary_events(tmp_path)
    events.loc[("02", "D1", "B4"), "sampling_event_recorded"] = False

    result = cw.merge_saliva(events, _saliva(tmp_path))

    assert not result.loc[("02", "D1", "B4"), "sampling_event_recorded"]
    assert result.loc[("02", "D1", "B4"), "lab_value_available"]


def test_merge_saliva_rejects_non_bijective_swaps(tmp_path):
    events = _summary_events(tmp_path)
    events.loc[("02", "D1", "B2"), "recorded_sample"] = "B1"

    with pytest.raises(MergeError, match="same physical saliva tube"):
        cw.merge_saliva(events, _saliva(tmp_path))


def test_merge_saliva_rejects_duplicate_sampling_positions(tmp_path):
    events = _summary_events(tmp_path).reset_index()
    events = pd.concat([events, events.iloc[[0]]], ignore_index=True)

    with pytest.raises(SchemaError, match="duplicate sampling positions"):
        cw.merge_saliva(events, _saliva(tmp_path))


def test_merge_saliva_rejects_incompatible_schemas(tmp_path):
    with pytest.raises(SchemaError, match="sample extractor"):
        cw.merge_saliva(pd.DataFrame({"scheduled_sample": ["B1"]}), _saliva(tmp_path))
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
