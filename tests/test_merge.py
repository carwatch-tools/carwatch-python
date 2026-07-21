from io import StringIO

import numpy as np
import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

import carwatch as cw
from carwatch.exceptions import MergeError, SchemaError


SUMMARY = """Participant ID,date_D1,awakening_time_D1_app,awakening_type_D1,sampling_time_D1_B1,sample_barcode_D1_B1,sample_scanned_D1_B1,sampling_time_D1_B2,sample_barcode_D1_B2,sample_scanned_D1_B2,sampling_time_D1_B3,sample_barcode_D1_B3,sample_scanned_D1_B3,sampling_time_D1_B4,sample_barcode_D1_B4,sample_scanned_D1_B4
02,2025-05-15,06:00:00,self-report,06:00:00,0010101,B1,06:30:00,0010103,B3,06:45:00,0010102,B2,07:00:00,0010104,B4
"""

AVAILABILITY_SUMMARY = """Participant ID,date_D1,awakening_time_D1_app,awakening_type_D1,sampling_time_D1_B1,sample_barcode_D1_B1,sample_scanned_D1_B1,sampling_time_D1_B2,sample_barcode_D1_B2,sample_scanned_D1_B2,sampling_time_D1_B3,sample_barcode_D1_B3,sample_scanned_D1_B3,sampling_time_D1_B4,sample_barcode_D1_B4,sample_scanned_D1_B4
02,2025-05-15,06:00:00,self-report,06:15:00,0010101,B1,06:30:00,0010102,B2,,,,,,
"""

TWO_DAY_SUMMARY = """Participant ID,date_D1,awakening_time_D1_app,sampling_time_D1_B1,sample_barcode_D1_B1,sample_scanned_D1_B1,sampling_time_D1_B2,sample_barcode_D1_B2,sample_scanned_D1_B2,date_D2,awakening_time_D2_app,sampling_time_D2_B3,sample_barcode_D2_B3,sample_scanned_D2_B3,sampling_time_D2_B4,sample_barcode_D2_B4,sample_scanned_D2_B4
02,2025-05-15,06:00:00,06:00:00,0010101,B1,,,,2025-05-16,07:00:00,07:00:00,0010103,B3,,,
"""

SALIVA = """subject,sample,cortisol
02,B1,1.0
02,B2,2.0
02,B3,3.0
02,B4,4.0
"""

AVAILABILITY_SALIVA = """subject,sample,cortisol
02,B1,1.0
02,B2,
02,B3,3.0
02,B4,
"""


def _study_results(tmp_path, content=SUMMARY):
    path = tmp_path / "summary.csv"
    path.write_text(content)
    return cw.io.load_study_manager_export(path)


def _saliva(tmp_path, content=SALIVA):
    path = tmp_path / "saliva.csv"
    path.write_text(content)
    return cw.io.load_saliva(path)


def _availability_data(tmp_path):
    return (
        _study_results(tmp_path, AVAILABILITY_SUMMARY),
        _saliva(tmp_path, AVAILABILITY_SALIVA),
    )


def test_merge_saliva_corrects_swaps_from_summary(tmp_path):
    result = cw.merge_saliva(_study_results(tmp_path), _saliva(tmp_path))

    assert result.index.names == ["participant", "day", "scheduled_sample"]
    assert result["cortisol"].tolist() == [1.0, 3.0, 2.0, 4.0]
    assert result["mismatch_corrected"].tolist() == [False, True, True, False]
    assert result["sampling_event_recorded"].all()
    assert result["lab_value_available"].all()
    assert not result["sampling_time_imputed"].any()
    assert "saliva_sample" not in result
    assert "barcode" not in result
    assert "sample_id_source" not in result
    assert "time_min" in result


def test_merge_saliva_can_skip_swap_correction(tmp_path):
    result = cw.merge_saliva(
        _study_results(tmp_path),
        _saliva(tmp_path),
        correct_swaps=False,
    )

    assert result["cortisol"].tolist() == [1.0, 2.0, 3.0, 4.0]
    assert not result["mismatch_corrected"].any()


def test_merge_saliva_rejects_raw_log_sample_events(tmp_path):
    path = tmp_path / "carwatch_logs_02.csv"
    path.write_text(
        '1747282435999;local;barcode_scanned;{"sample_expected":"B1","sample_scanned":"B1"}\n'
    )
    events = cw.logs.extract_sample_events_from_raw_logs(cw.io.load_raw_logs(path))

    with pytest.raises(SchemaError, match="Study summary"):
        cw.merge_saliva(events, _saliva(tmp_path))


def test_missing_carwatch_data_ignore_is_default(tmp_path):
    study_results, saliva = _availability_data(tmp_path)

    result = cw.merge_saliva(study_results, saliva)

    missing = result.loc[("02", "D1", "B3")]
    assert not missing["sampling_event_recorded"]
    assert missing["lab_value_available"]
    assert pd.isna(missing["sampling_time"])
    assert not missing["sampling_time_imputed"]


def test_missing_carwatch_data_raise_reports_positions(tmp_path):
    study_results, saliva = _availability_data(tmp_path)

    with pytest.raises(
        MergeError,
        match="participant.*02.*day.*D1.*scheduled_sample.*B3",
    ):
        cw.merge_saliva(
            study_results,
            saliva,
            missing_carwatch_data="raise",
        )


def test_missing_carwatch_data_raise_ignores_other_availability_cases(tmp_path):
    study_results, saliva = _availability_data(tmp_path)
    saliva.loc[("02", "B3"), "cortisol"] = pd.NA

    result = cw.merge_saliva(
        study_results,
        saliva,
        missing_carwatch_data="raise",
    )

    assert not result.loc[("02", "D1", "B3"), "sampling_event_recorded"]
    assert not result.loc[("02", "D1", "B3"), "lab_value_available"]


def test_impute_relative_sampling_schedule(tmp_path):
    study_results, saliva = _availability_data(tmp_path)

    result = cw.merge_saliva(
        study_results,
        saliva,
        missing_carwatch_data="impute",
        sampling_schedule=[15, 30, 45, 60],
    )

    imputed = result.loc[("02", "D1", "B3")]
    assert imputed["sampling_time"] == pd.Timestamp(
        "2025-05-15 06:45:00", tz="Europe/Berlin"
    )
    assert imputed["time_min"] == 45
    assert imputed["sampling_time_imputed"]
    assert not imputed["sampling_event_recorded"]


@pytest.mark.parametrize(
    "sampling_schedule",
    [
        (15, 30, 45, 60),
        np.array([15, 30, 45, 60]),
    ],
)
def test_impute_accepts_tuple_and_array_schedules(sampling_schedule, tmp_path):
    study_results, saliva = _availability_data(tmp_path)

    result = cw.merge_saliva(
        study_results,
        saliva,
        missing_carwatch_data="impute",
        sampling_schedule=sampling_schedule,
    )

    assert result.loc[("02", "D1", "B3"), "sampling_time_imputed"]


def test_impute_mixed_relative_and_absolute_schedule(tmp_path):
    study_results, saliva = _availability_data(tmp_path)

    result = cw.merge_saliva(
        study_results,
        saliva,
        missing_carwatch_data="impute",
        sampling_schedule=[15, 30, "07:15", "07:30:00"],
    )

    imputed = result.loc[("02", "D1", "B3")]
    assert imputed["sampling_time"] == pd.Timestamp(
        "2025-05-15 07:15:00", tz="Europe/Berlin"
    )
    assert imputed["time_min"] == 75


def test_impute_changes_only_missing_event_with_lab_value(tmp_path):
    study_results, saliva = _availability_data(tmp_path)
    original_events = cw.logs.extract_sample_events_from_summary(study_results)

    result = cw.merge_saliva(
        study_results,
        saliva,
        missing_carwatch_data="impute",
        sampling_schedule=[15, 30, 45, 60],
    )

    assert (
        result.loc[("02", "D1", "B1"), "sampling_time"]
        == original_events.loc[("02", "D1", "B1"), "sampling_time"]
    )
    assert not result.loc[("02", "D1", "B1"), "sampling_time_imputed"]
    assert not result.loc[("02", "D1", "B2"), "sampling_time_imputed"]
    assert result.loc[("02", "D1", "B3"), "sampling_time_imputed"]
    assert pd.isna(result.loc[("02", "D1", "B4"), "sampling_time"])
    assert not result.loc[("02", "D1", "B4"), "sampling_time_imputed"]


def test_impute_list_schedule_applies_to_every_day(tmp_path):
    result = cw.merge_saliva(
        _study_results(tmp_path, TWO_DAY_SUMMARY),
        _saliva(tmp_path),
        missing_carwatch_data="impute",
        sampling_schedule=[0, 15],
    )

    assert result.loc[("02", "D1", "B2"), "sampling_time"] == pd.Timestamp(
        "2025-05-15 06:15:00", tz="Europe/Berlin"
    )
    assert result.loc[("02", "D2", "B4"), "sampling_time"] == pd.Timestamp(
        "2025-05-16 07:15:00", tz="Europe/Berlin"
    )


def test_impute_dictionary_schedule_by_day(tmp_path):
    result = cw.merge_saliva(
        _study_results(tmp_path, TWO_DAY_SUMMARY),
        _saliva(tmp_path),
        missing_carwatch_data="impute",
        sampling_schedule={"D1": [0, 20], "D2": [0, "07:45"]},
    )

    assert result.loc[("02", "D1", "B2"), "time_min"] == 20
    assert result.loc[("02", "D2", "B4"), "time_min"] == 45


def test_absolute_schedule_does_not_require_awakening_time(tmp_path):
    content = AVAILABILITY_SUMMARY.replace("06:00:00,self-report", ",self-report")
    result = cw.merge_saliva(
        _study_results(tmp_path, content),
        _saliva(tmp_path, AVAILABILITY_SALIVA),
        missing_carwatch_data="impute",
        sampling_schedule=["06:15", "06:30", "07:00", "07:15"],
    )

    imputed = result.loc[("02", "D1", "B3")]
    assert imputed["sampling_time"] == pd.Timestamp(
        "2025-05-15 07:00:00", tz="Europe/Berlin"
    )
    assert pd.isna(imputed["time_min"])


def test_relative_schedule_requires_awakening_time(tmp_path):
    content = AVAILABILITY_SUMMARY.replace("06:00:00,self-report", ",self-report")

    with pytest.raises(MergeError, match="without an awakening time"):
        cw.merge_saliva(
            _study_results(tmp_path, content),
            _saliva(tmp_path, AVAILABILITY_SALIVA),
            missing_carwatch_data="impute",
            sampling_schedule=[15, 30, 45, 60],
        )


def test_absolute_schedule_requires_study_date(tmp_path):
    content = AVAILABILITY_SUMMARY.replace("02,2025-05-15", "02,")

    with pytest.raises(MergeError, match="without a date"):
        cw.merge_saliva(
            _study_results(tmp_path, content),
            _saliva(tmp_path, AVAILABILITY_SALIVA),
            missing_carwatch_data="impute",
            sampling_schedule=["06:15", "06:30", "07:00", "07:15"],
        )


@pytest.mark.parametrize("mode", ["unknown", None, 1])
def test_merge_rejects_invalid_missing_carwatch_data(mode, tmp_path):
    with pytest.raises(ValueError, match="missing_carwatch_data"):
        cw.merge_saliva(
            _study_results(tmp_path),
            _saliva(tmp_path),
            missing_carwatch_data=mode,
        )


def test_merge_requires_schedule_only_for_imputation(tmp_path):
    study_results, saliva = _availability_data(tmp_path)

    with pytest.raises(ValueError, match="required"):
        cw.merge_saliva(
            study_results,
            saliva,
            missing_carwatch_data="impute",
        )
    with pytest.raises(ValueError, match="only allowed"):
        cw.merge_saliva(study_results, saliva, sampling_schedule=[15, 30, 45, 60])


@pytest.mark.parametrize(
    ("schedule", "message"),
    [
        ([15, 30, 45], "defines 4 samples"),
        ([[15, 30], [45, 60]], "one-dimensional"),
        ([15, 30, True, 60], "invalid value"),
        ([15, 30, float("nan"), 60], "non-finite"),
        ([15, 30, "7:00", 60], "HH:MM"),
        ([15, 30, 15, 60], "strictly increasing"),
    ],
)
def test_merge_rejects_invalid_list_schedules(schedule, message, tmp_path):
    study_results, saliva = _availability_data(tmp_path)

    with pytest.raises(SchemaError, match=message):
        cw.merge_saliva(
            study_results,
            saliva,
            missing_carwatch_data="impute",
            sampling_schedule=schedule,
        )


def test_merge_validates_dictionary_days(tmp_path):
    study_results = _study_results(tmp_path, TWO_DAY_SUMMARY)
    saliva = _saliva(tmp_path)

    with pytest.raises(SchemaError, match="unknown day IDs.*D3"):
        cw.merge_saliva(
            study_results,
            saliva,
            missing_carwatch_data="impute",
            sampling_schedule={"D1": [0, 15], "D2": [0, 15], "D3": [0, 15]},
        )
    with pytest.raises(SchemaError, match="missing affected day IDs.*D2"):
        cw.merge_saliva(
            study_results,
            saliva,
            missing_carwatch_data="impute",
            sampling_schedule={"D1": [0, 15]},
        )
    with pytest.raises(SchemaError, match="D1.*1 values.*defines 2 samples"):
        cw.merge_saliva(
            study_results,
            saliva,
            missing_carwatch_data="impute",
            sampling_schedule={"D1": [0], "D2": [0, 15]},
        )


def test_merge_accepts_extra_valid_dictionary_day(tmp_path):
    content = TWO_DAY_SUMMARY.replace(
        ",07:00:00,0010103,B3,,,\n",
        ",07:00:00,0010103,B3,07:15:00,0010104,B4\n",
    )
    result = cw.merge_saliva(
        _study_results(tmp_path, content),
        _saliva(tmp_path),
        missing_carwatch_data="impute",
        sampling_schedule={"D1": [0, 15], "D2": [0, 15]},
    )

    assert result.loc[("02", "D1", "B2"), "sampling_time_imputed"]


def test_merge_saliva_rejects_non_bijective_swaps(tmp_path):
    study_results = _study_results(tmp_path)
    study_results.loc["02", ("D1", "B2", "recorded_sample")] = "B1"

    with pytest.raises(MergeError, match="same physical saliva tube"):
        cw.merge_saliva(study_results, _saliva(tmp_path))


def test_merge_saliva_rejects_incompatible_schemas(tmp_path):
    with pytest.raises(SchemaError, match="participant"):
        cw.merge_saliva(pd.DataFrame({"scheduled_sample": ["B1"]}), _saliva(tmp_path))
    with pytest.raises(SchemaError, match="index levels"):
        cw.merge_saliva(_study_results(tmp_path), pd.DataFrame({"cortisol": [1.0]}))


def test_merge_saliva_does_not_mutate_inputs(tmp_path):
    study_results = _study_results(tmp_path)
    saliva = _saliva(tmp_path)
    expected_results = study_results.copy(deep=True)
    expected_saliva = saliva.copy(deep=True)

    cw.merge_saliva(study_results, saliva)

    assert_frame_equal(study_results, expected_results)
    assert_frame_equal(saliva, expected_saliva)


def test_merge_saliva_rejects_non_numeric_measurements(tmp_path):
    saliva = pd.read_csv(StringIO(SALIVA), dtype="string").set_index(
        ["subject", "sample"]
    )

    with pytest.raises(SchemaError, match="numeric"):
        cw.merge_saliva(_study_results(tmp_path), saliva)
