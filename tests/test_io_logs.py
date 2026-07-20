import warnings
import zipfile

import pandas as pd
import pytest

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

    data = cw.io.load_logs(path)

    assert len(data) == 3
    assert data.columns.tolist() == [
        "participant",
        "date",
        "timestamp",
        "timestamp_ms",
        "action",
        "payload",
        "source_file",
    ]
    assert data["participant"].unique().tolist() == ["02"]
    assert str(data["timestamp"].dt.tz) == "Europe/Berlin"
    assert data.iloc[1]["payload"]["barcode_value"] == "0010101"
    assert data.iloc[0]["date"] == pd.Timestamp("2025-05-15", tz="Europe/Berlin")


def test_load_multiline_json(tmp_path):
    path = tmp_path / "carwatch_test_VP_01_20260417.csv"
    path.write_text(MULTILINE_LOG)

    data = cw.io.load_logs(path)

    assert len(data) == 2
    assert data.iloc[1]["payload"]["sample_expected"] == "S1"
    assert data["participant"].unique().tolist() == ["VP_01"]
    assert data["date"].iloc[0] == pd.Timestamp("2026-04-17", tz="Europe/Berlin")


def test_load_legacy_three_column_log(tmp_path):
    path = tmp_path / "carwatch_study_VP01_20250515.csv"
    path.write_text('1747282410799;spontaneous_awakening;{"id":0}\n')

    data = cw.io.load_logs(path)

    assert data.loc[0, "action"] == "spontaneous_awakening"


def test_load_zip_ignores_hidden_files(tmp_path):
    path = tmp_path / "logs.zip"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("carwatch_logs_02.csv", CURRENT_LOG)
        archive.writestr("__MACOSX/._carwatch_logs_02.csv", CURRENT_LOG)

    data = cw.io.load_logs(path)

    assert len(data) == 3
    assert data["source_file"].str.contains("!carwatch_logs_02.csv", regex=False).all()


@pytest.mark.parametrize("errors", ["warn", "ignore"])
def test_invalid_json_can_be_retained(tmp_path, errors):
    path = tmp_path / "carwatch_logs_02.csv"
    path.write_text("1747282410799;local;spontaneous_awakening;{invalid}\n")

    with warnings.catch_warnings(record=True) as caught:
        data = cw.io.load_logs(path, errors=errors)

    assert data.loc[0, "payload"] is None
    assert bool(caught) is (errors == "warn")


def test_invalid_json_raises_by_default(tmp_path):
    path = tmp_path / "carwatch_logs_02.csv"
    path.write_text("1747282410799;local;spontaneous_awakening;{invalid}\n")

    with pytest.raises(LogParseError):
        cw.io.load_logs(path)


def test_extract_sample_events_from_raw_logs_marks_swapped_tube(tmp_path):
    path = tmp_path / "carwatch_logs_02.csv"
    path.write_text(CURRENT_LOG)

    samples = cw.logs.extract_sample_events_from_raw_logs(cw.io.load_logs(path))

    assert samples["scheduled_sample"].tolist() == ["B1", "B2"]
    assert samples["recorded_sample"].tolist() == ["B1", "B3"]
    assert samples["sample_mismatch"].tolist() == [False, True]
    assert "barcode" not in samples
    assert "study" not in samples
    assert "event_index" not in samples


def test_extract_awakening_events_from_raw_logs_prefers_self_report(tmp_path):
    content = (
        '1747282400000;local;alarm_stop;{"id":0}\n'
        '1747282410799;local;spontaneous_awakening;{"id":0}\n'
    )
    path = tmp_path / "carwatch_logs_02.csv"
    path.write_text(content)

    awakening = cw.logs.extract_awakening_events_from_raw_logs(cw.io.load_logs(path))

    assert len(awakening) == 1
    assert awakening.loc[0, "awakening_type"] == "self-report"
    assert "study" not in awakening


def test_extractors_validate_input_schema():
    with pytest.raises(SchemaError):
        cw.logs.extract_sample_events_from_raw_logs(pd.DataFrame({"action": []}))
