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


def test_load_study_results_normalizes_samples(tmp_path):
    result = cw.io.load_study_results(_write_csv(tmp_path))

    assert result.index.names == ["study", "participant", "day", "sample"]
    assert result.index.get_level_values("participant").unique().tolist() == ["02"]
    assert result.index.get_level_values("sample").tolist() == ["B1", "B2", "B3", "B4"]
    assert result["barcode"].tolist() == ["0010101", "0010103", "0010102", "0010104"]


def test_load_study_results_preserves_swap_information(tmp_path):
    result = cw.io.load_study_results(_write_csv(tmp_path))

    assert result.loc[("logs", "02", "D1", "B2"), "sample_scanned"] == "B3"
    assert result.loc[("logs", "02", "D1", "B3"), "sample_scanned"] == "B2"
    assert result["sample_mismatch"].tolist() == [False, True, True, False]
    assert result["mismatch_summary"].unique().tolist() == ["B2->B3;B3->B2"]


def test_load_study_results_combines_date_and_times(tmp_path):
    result = cw.io.load_study_results(_write_csv(tmp_path))

    first = result.loc[("logs", "02", "D1", "B1")]
    assert first["date"] == pd.Timestamp("2025-05-15", tz="Europe/Berlin")
    assert first["awakening_time"] == pd.Timestamp("2025-05-15 06:13:30", tz="Europe/Berlin")
    assert first["sampling_time"] == pd.Timestamp("2025-05-15 06:13:55", tz="Europe/Berlin")
    assert first["time"] == pytest.approx(25 / 60)


def test_load_study_results_retains_empty_sample_slots(tmp_path):
    data = pd.read_csv(StringIO(STUDY_RESULTS), dtype="string", keep_default_na=False)
    data.loc[0, ["sampling_time_D1_B4", "sample_barcode_D1_B4", "sample_scanned_D1_B4"]] = ""
    path = tmp_path / "study_results.csv"
    data.to_csv(path, index=False)

    result = cw.io.load_study_results(path)

    empty = result.loc[("logs", "02", "D1", "B4")]
    assert not empty["observed"]
    assert pd.isna(empty["sampling_time"])
    assert empty["sample_scanned"] == "B4"


def test_load_study_results_supports_multiple_days_and_google_fit(tmp_path):
    content = """Study Name,Participant ID,date_D1,awakening_time_D1_app,awakening_time_D1_google_fit,awakening_type_D1,sampling_time_D1_S1,sample_barcode_D1_S1,sample_scanned_D1_S1,date_D2,awakening_time_D2_app,awakening_type_D2,sampling_time_D2_S1,sample_barcode_D2_S1,sample_scanned_D2_S1
study,VP_01,2025-05-15,06:00:00,05:59:00,self-report,06:00:30,0001,S1,2025-05-16,07:00:00,alarm,07:01:00,0002,S1
"""
    result = cw.io.load_study_results(_write_csv(tmp_path, content))

    assert result.index.get_level_values("day").tolist() == ["D1", "D2"]
    assert result.iloc[0]["awakening_time_google_fit"] == pd.Timestamp(
        "2025-05-15 05:59:00", tz="Europe/Berlin"
    )
    assert pd.isna(result.iloc[1]["awakening_time_google_fit"])


@pytest.mark.parametrize(
    "content",
    [
        "Study Name,Participant ID\nstudy,01\n",
        "Study Name,Participant ID,date_D1\nstudy,01,2025-01-01\n",
        "Participant ID,date_D1,sampling_time_D1_S1\n01,2025-01-01,06:00\n",
    ],
)
def test_load_study_results_rejects_invalid_schema(tmp_path, content):
    with pytest.raises(SchemaError):
        cw.io.load_study_results(_write_csv(tmp_path, content))


def test_load_study_results_rejects_duplicate_participants(tmp_path):
    content = STUDY_RESULTS + STUDY_RESULTS.splitlines()[1] + "\n"

    with pytest.raises(SchemaError, match="duplicate"):
        cw.io.load_study_results(_write_csv(tmp_path, content))
