import warnings
import zipfile
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import carwatch as cw
from carwatch.exceptions import LogParseError, SchemaError


CURRENT_LOG = """1747282410799;Thu May 15 2025 06:13:30 GMT+02:00;spontaneous_awakening;{"id":0}
1747282435999;Thu May 15 2025 06:13:55 GMT+02:00;barcode_scanned;{"id":0,"saliva_id":100,"barcode_value":"0010101","day_scanned":1,"day_expected":1,"sample_scanned":"B1","sample_expected":"B1"}
1747284231091;Thu May 15 2025 06:43:51 GMT+02:00;barcode_scanned;{"id":1,"saliva_id":101,"barcode_value":"0010103","day_scanned":1,"day_expected":1,"sample_scanned":"B3","sample_expected":"B2"}
"""

MULTILINE_LOG = """1776429093567;Fri Apr 17 2026 2:31:33 pm GMT+02:00;spontaneous_awakening;{
  "id" : -1
}
1776429099447;Fri Apr 17 2026 2:31:39 pm GMT+02:00;barcode_scanned;{
  "saliva_id" : 101,
  "barcode_value" : "0010101",
  "sample_scanned" : "S1",
  "sample_expected" : "S1"
}
"""


def test_load_current_log_and_preserve_identifiers(tmp_path):
    path = tmp_path / "carwatch_logs_02.csv"
    path.write_text(CURRENT_LOG)

    data = cw.io.load_raw_logs(path)

    assert len(data) == 3
    assert data.columns.tolist() == [
        "action",
        "payload",
        "timestamp_ms",
        "source_file",
    ]
    assert data.index.names == ["participant", "date", "timestamp"]
    assert data.index.get_level_values("participant").unique().tolist() == ["02"]
    assert str(data.index.get_level_values("timestamp").dtype.tz) == "Europe/Berlin"
    assert data.iloc[1]["payload"]["barcode_value"] == "0010101"
    assert data.index.get_level_values("date")[0] == pd.Timestamp(
        "2025-05-15", tz="Europe/Berlin"
    )


def test_load_multiline_json(tmp_path):
    path = tmp_path / "carwatch_test_VP_01_20260417.csv"
    path.write_text(MULTILINE_LOG)

    data = cw.io.load_raw_logs(path)

    assert len(data) == 2
    assert data.iloc[1]["payload"]["sample_expected"] == "S1"
    assert data.index.get_level_values("participant").unique().tolist() == ["VP_01"]
    assert data.index.get_level_values("date")[0] == pd.Timestamp(
        "2026-04-17", tz="Europe/Berlin"
    )


def test_load_legacy_three_column_log(tmp_path):
    path = tmp_path / "carwatch_study_VP01_20250515.csv"
    path.write_text('1747282410799;spontaneous_awakening;{"id":0}\n')

    data = cw.io.load_raw_logs(path)

    assert data.iloc[0]["action"] == "spontaneous_awakening"


def test_load_zip_ignores_hidden_files(tmp_path):
    path = tmp_path / "logs.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("carwatch_logs_02.csv", CURRENT_LOG)
        archive.writestr("__MACOSX/._carwatch_logs_02.csv", CURRENT_LOG)

    data = cw.io.load_raw_logs(path)

    assert len(data) == 3
    assert data["source_file"].str.contains("!carwatch_logs_02.csv", regex=False).all()


@pytest.mark.parametrize("errors", ["warn", "ignore"])
def test_invalid_json_can_be_retained(tmp_path, errors):
    path = tmp_path / "carwatch_logs_02.csv"
    path.write_text("1747282410799;local;spontaneous_awakening;{invalid}\n")

    with warnings.catch_warnings(record=True) as caught:
        data = cw.io.load_raw_logs(path, errors=errors)

    assert data.iloc[0]["payload"] is None
    assert bool(caught) is (errors == "warn")


def test_invalid_json_raises_by_default(tmp_path):
    path = tmp_path / "carwatch_logs_02.csv"
    path.write_text("1747282410799;local;spontaneous_awakening;{invalid}\n")

    with pytest.raises(LogParseError):
        cw.io.load_raw_logs(path)


def test_extract_sample_events_from_raw_logs_marks_swapped_tube(tmp_path):
    path = tmp_path / "carwatch_logs_02.csv"
    path.write_text(CURRENT_LOG)

    samples = cw.logs.extract_sample_events_from_raw_logs(cw.io.load_raw_logs(path))

    assert samples["scheduled_sample"].tolist() == ["B1", "B2"]
    assert samples["recorded_sample"].tolist() == ["B1", "B3"]
    assert samples.loc[0, "sampling_time"].microsecond == 0
    assert samples["sampling_event_recorded"].tolist() == [True, True]
    assert samples["sample_mismatch"].tolist() == [False, True]
    assert "barcode" not in samples
    assert "study" not in samples
    assert "event_index" not in samples


def test_extract_awakening_events_from_raw_logs_uses_first_event(tmp_path):
    content = (
        '1747282400000;local;alarm_stop;{"id":0}\n'
        '1747282410799;local;spontaneous_awakening;{"id":0}\n'
    )
    path = tmp_path / "carwatch_logs_02.csv"
    path.write_text(content)

    awakening = cw.logs.extract_awakening_events_from_raw_logs(
        cw.io.load_raw_logs(path)
    )

    assert len(awakening) == 1
    assert awakening.loc[0, "awakening_type"] == "alarm"
    assert awakening.loc[0, "awakening_time"].microsecond == 0
    assert "study" not in awakening


def test_convert_raw_logs_to_summary_matches_study_manager_example():
    data_dir = Path(__file__).parents[1] / "examples" / "data"
    raw_logs = cw.io.load_raw_logs(data_dir / "carwatch_demo_VP01_20250515.csv")
    expected = cw.io.load_study_manager_export(data_dir / "study_results.csv")

    result = cw.logs.convert_raw_logs_to_study_manager_summary(raw_logs)

    assert_frame_equal(result, expected)


def test_convert_raw_logs_to_summary_preserves_swaps(tmp_path):
    path = tmp_path / "carwatch_logs_02.csv"
    path.write_text(CURRENT_LOG)

    result = cw.logs.convert_raw_logs_to_study_manager_summary(cw.io.load_raw_logs(path))

    assert result.columns.names == ["day", "sample", "variable"]
    assert result.index.tolist() == ["02"]
    assert result.loc["02", ("D1", "day", "awakening_type")] == "self-report"
    assert result.loc["02", ("D1", "B1", "barcode")] == "0010101"
    assert result.loc["02", ("D1", "B2", "recorded_sample")] == "B3"
    assert result.loc["02", ("D1", "day", "mismatch_summary")] == "B2->B3"


def test_convert_raw_logs_to_summary_assigns_chronological_days(tmp_path):
    first = tmp_path / "carwatch_demo_01_20250515.csv"
    first.write_text(
        '1747282410799;local;spontaneous_awakening;{"id":0}\n'
        '1747282435999;local;barcode_scanned;{"barcode_value":"001","sample_expected":"B1","sample_scanned":"B1"}\n'
    )
    second = tmp_path / "carwatch_demo_01_20250516.csv"
    second.write_text(
        '1747368810799;local;alarm_stop;{"id":0}\n'
        '1747368835999;local;barcode_scanned;{"barcode_value":"002","sample_expected":"B1","sample_scanned":"B1"}\n'
    )

    result = cw.logs.convert_raw_logs_to_study_manager_summary(cw.io.load_raw_logs([second, first]))

    assert result.loc["01", ("D1", "day", "date")] == pd.Timestamp(
        "2025-05-15", tz="Europe/Berlin"
    )
    assert result.loc["01", ("D2", "day", "date")] == pd.Timestamp(
        "2025-05-16", tz="Europe/Berlin"
    )
    assert result.loc["01", ("D2", "day", "awakening_type")] == "alarm"


def test_convert_raw_logs_to_summary_advances_duplicate_legacy_samples(tmp_path):
    path = tmp_path / "carwatch_logs_01.csv"
    path.write_text(
        '1747282435999;local;barcode_scanned;{"id":0,"saliva_id":1,"barcode_value":"001"}\n'
        '1747284231091;local;barcode_scanned;{"id":1,"saliva_id":1,"barcode_value":"002"}\n'
        '1747285132626;local;barcode_scanned;{"id":815,"saliva_id":2,"barcode_value":"003"}\n'
    )

    result = cw.logs.convert_raw_logs_to_study_manager_summary(cw.io.load_raw_logs(path))

    samples = result.columns.get_level_values("sample").unique().tolist()
    assert samples == ["day", "S1", "S2", "SE"]
    assert result.loc["01", ("D1", "S2", "recorded_sample")] == "S2"


def test_convert_raw_logs_to_summary_rejects_irrelevant_logs(tmp_path):
    path = tmp_path / "carwatch_logs_01.csv"
    path.write_text('1747282435999;local;timer_set;{"id":0}\n')

    with pytest.raises(SchemaError, match="do not contain awakening or sampling"):
        cw.logs.convert_raw_logs_to_study_manager_summary(cw.io.load_raw_logs(path))


def test_extractors_validate_input_schema():
    with pytest.raises(SchemaError):
        cw.logs.extract_sample_events_from_raw_logs(pd.DataFrame({"action": []}))
