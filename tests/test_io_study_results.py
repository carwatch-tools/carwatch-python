from io import StringIO

import pandas as pd
import pytest

import carwatch as cw
from carwatch.exceptions import SchemaError


STUDY_RESULTS = """Study Name,Participant ID,date_D1,awakening_time_D1_app,awakening_type_D1,sample_mismatches_d1,sampling_time_D1_B1,sample_barcode_D1_B1,sample_scanned_D1_B1,sampling_time_D1_B2,sample_barcode_D1_B2,sample_scanned_D1_B2,sampling_time_D1_B3,sample_barcode_D1_B3,sample_scanned_D1_B3,sampling_time_D1_B4,sample_barcode_D1_B4,sample_scanned_D1_B4
logs,02,2025-05-15,06:13:30,self-report,B2->B3;B3->B2,06:13:55,0010101,B1,06:43:51,0010103,B3,06:58:52,0010102,B2,07:13:47,0010104,B4
"""


def _write_csv(tmp_path, content=STUDY_RESULTS):
    path = tmp_path / "study_results.csv"
    path.write_text(content)
    return path


def test_load_study_results_returns_wide_multiindex_columns(tmp_path):
    result = cw.io.load_study_results(_write_csv(tmp_path))

    assert result.index.name == "participant"
    assert result.index.tolist() == ["02"]
    assert result.columns.names == ["day", "sample", "variable"]
    assert result.columns.tolist() == [
        ("D1", "day", "date"),
        ("D1", "day", "awakening_time"),
        ("D1", "day", "awakening_type"),
        ("D1", "day", "mismatch_summary"),
        ("D1", "B1", "sampling_time"),
        ("D1", "B1", "barcode"),
        ("D1", "B1", "recorded_sample"),
        ("D1", "B2", "sampling_time"),
        ("D1", "B2", "barcode"),
        ("D1", "B2", "recorded_sample"),
        ("D1", "B3", "sampling_time"),
        ("D1", "B3", "barcode"),
        ("D1", "B3", "recorded_sample"),
        ("D1", "B4", "sampling_time"),
        ("D1", "B4", "barcode"),
        ("D1", "B4", "recorded_sample"),
    ]
    assert result[("D1", "B1", "barcode")].tolist() == ["0010101"]


def test_load_study_results_preserves_day_level_mismatch_summary(tmp_path):
    result = cw.io.load_study_results(_write_csv(tmp_path))

    assert result.loc["02", ("D1", "B2", "recorded_sample")] == "B3"
    assert result.loc["02", ("D1", "B3", "recorded_sample")] == "B2"
    assert result.loc["02", ("D1", "day", "mismatch_summary")] == "B2->B3;B3->B2"


def test_load_study_results_combines_date_and_times(tmp_path):
    result = cw.io.load_study_results(_write_csv(tmp_path))

    assert result.loc["02", ("D1", "day", "date")] == pd.Timestamp(
        "2025-05-15", tz="Europe/Berlin"
    )
    assert result.loc["02", ("D1", "day", "awakening_time")] == pd.Timestamp(
        "2025-05-15 06:13:30", tz="Europe/Berlin"
    )
    assert result.loc["02", ("D1", "B1", "sampling_time")] == pd.Timestamp(
        "2025-05-15 06:13:55", tz="Europe/Berlin"
    )


def test_extract_day_summary_from_summary_returns_one_row_per_day(tmp_path):
    result = cw.io.load_study_results(_write_csv(tmp_path))

    day_summary = cw.logs.extract_day_summary_from_summary(result)

    assert day_summary.index.names == ["participant", "day"]
    assert day_summary.index.tolist() == [("02", "D1")]
    assert day_summary.columns.tolist() == [
        "date",
        "awakening_time",
        "awakening_type",
        "mismatch_summary",
    ]
    assert day_summary.loc[("02", "D1"), "awakening_type"] == "self-report"
    assert day_summary.loc[("02", "D1"), "mismatch_summary"] == "B2->B3;B3->B2"


def test_extract_awakening_events_from_summary_returns_awakening_fields(tmp_path):
    result = cw.io.load_study_results(_write_csv(tmp_path))

    awakening = cw.logs.extract_awakening_events_from_summary(result)

    assert awakening.index.names == ["participant", "day"]
    assert awakening.columns.tolist() == [
        "date",
        "awakening_time",
        "awakening_type",
    ]
    assert awakening.loc[("02", "D1"), "awakening_type"] == "self-report"


def test_extract_sample_events_from_summary_derives_sample_fields(tmp_path):
    result = cw.io.load_study_results(_write_csv(tmp_path))

    samples = cw.logs.extract_sample_events_from_summary(result)

    assert samples.index.names == ["participant", "day", "scheduled_sample"]
    assert samples.index.get_level_values("scheduled_sample").tolist() == [
        "B1",
        "B2",
        "B3",
        "B4",
    ]
    assert samples["sample_mismatch"].tolist() == [False, True, True, False]
    assert samples.loc[("02", "D1", "B1"), "time_min"] == pytest.approx(25 / 60)
    assert "barcode" not in samples
    assert "awakening_time" not in samples
    assert "mismatch_summary" not in samples


def test_empty_sample_values_remain_unknown(tmp_path):
    data = pd.read_csv(StringIO(STUDY_RESULTS), dtype="string", keep_default_na=False)
    data.loc[
        0,
        ["sampling_time_D1_B4", "sample_barcode_D1_B4", "sample_scanned_D1_B4"],
    ] = ""
    path = tmp_path / "study_results.csv"
    data.to_csv(path, index=False)

    result = cw.io.load_study_results(path)
    samples = cw.logs.extract_sample_events_from_summary(result)
    empty = samples.loc[("02", "D1", "B4")]

    assert pd.isna(result.loc["02", ("D1", "B4", "recorded_sample")])
    assert not empty["observed"]
    assert pd.isna(empty["sampling_time"])
    assert pd.isna(empty["sample_mismatch"])


def test_load_study_results_supports_multiple_days_and_ignores_google_fit(tmp_path):
    content = """Study Name,Participant ID,date_D1,awakening_time_D1_app,awakening_time_D1_google_fit,awakening_type_D1,sampling_time_D1_S1,sample_barcode_D1_S1,sample_scanned_D1_S1,date_D2,awakening_time_D2_app,awakening_type_D2,sampling_time_D2_S1,sample_barcode_D2_S1,sample_scanned_D2_S1
study,VP_01,2025-05-15,06:00:00,05:59:00,self-report,06:00:30,0001,S1,2025-05-16,07:00:00,alarm,07:01:00,0002,S1
"""
    result = cw.io.load_study_results(_write_csv(tmp_path, content))
    awakening = cw.logs.extract_awakening_events_from_summary(result)

    assert result.columns.get_level_values("day").unique().tolist() == ["D1", "D2"]
    assert "awakening_time_google_fit" not in result.columns.get_level_values(
        "variable"
    )
    assert awakening.index.tolist() == [("VP_01", "D1"), ("VP_01", "D2")]
    assert awakening.loc[("VP_01", "D2"), "awakening_time"] == pd.Timestamp(
        "2025-05-16 07:00:00", tz="Europe/Berlin"
    )


def test_load_study_results_does_not_require_study_name(tmp_path):
    content = """Participant ID,date_D1,sampling_time_D1_S1
01,2025-01-01,06:00:00
"""

    result = cw.io.load_study_results(_write_csv(tmp_path, content))

    assert result.index.tolist() == ["01"]
    assert result.loc["01", ("D1", "S1", "sampling_time")] == pd.Timestamp(
        "2025-01-01 06:00:00", tz="Europe/Berlin"
    )


def test_load_study_results_sorts_samples_naturally(tmp_path):
    content = """Participant ID,date_D1,sampling_time_D1_B10,sampling_time_D1_B2
01,2025-01-01,06:10:00,06:02:00
"""

    result = cw.io.load_study_results(_write_csv(tmp_path, content))

    samples = result.columns.get_level_values("sample").unique().tolist()
    assert samples == ["day", "B2", "B10"]


@pytest.mark.parametrize(
    "content",
    [
        "Study Name,Participant ID\nstudy,01\n",
        "Study Name,Participant ID,date_D1\nstudy,01,2025-01-01\n",
        "date_D1,sampling_time_D1_S1\n2025-01-01,06:00:00\n",
    ],
)
def test_load_study_results_rejects_invalid_schema(tmp_path, content):
    with pytest.raises(SchemaError):
        cw.io.load_study_results(_write_csv(tmp_path, content))


def test_load_study_results_rejects_duplicate_participants(tmp_path):
    content = STUDY_RESULTS + STUDY_RESULTS.splitlines()[1] + "\n"

    with pytest.raises(SchemaError, match="duplicate"):
        cw.io.load_study_results(_write_csv(tmp_path, content))


def test_summary_extractors_validate_schema():
    with pytest.raises(SchemaError, match="MultiIndex"):
        cw.logs.extract_sample_events_from_summary(
            pd.DataFrame(index=pd.Index(["01"], name="participant"))
        )


def test_study_manager_submodule_is_not_public():
    assert "study_manager" not in cw.__all__
    with pytest.raises(AttributeError):
        _ = cw.study_manager
