from pathlib import Path

import pytest

import carwatch as cw


@pytest.mark.parametrize(
    ("sample", "event_recorded", "lab_value_available"),
    [
        ("B1", True, True),
        ("B2", True, False),
        ("B3", False, True),
        ("B4", False, False),
    ],
)
def test_availability_example_data(
    sample,
    event_recorded,
    lab_value_available,
):
    data_dir = Path(__file__).parents[1] / "example_data"
    summary = cw.io.load_study_manager_export(data_dir / "study_manager_summary.csv")
    saliva = cw.io.load_saliva(data_dir / "saliva_samples.csv")

    merged = cw.merge_saliva(summary, saliva)
    result = merged.loc[("01", "D1", sample)]

    assert bool(result["sampling_event_recorded"]) is event_recorded
    assert bool(result["lab_value_available"]) is lab_value_available


def test_availability_example_data_can_be_imputed():
    data_dir = Path(__file__).parents[1] / "example_data"
    summary = cw.io.load_study_manager_export(data_dir / "study_manager_summary.csv")
    saliva = cw.io.load_saliva(data_dir / "saliva_samples.csv")

    merged = cw.merge_saliva(
        summary,
        saliva,
        missing_carwatch_data="impute",
        sampling_schedule=[15, 30, 45, 60],
    )

    result = merged.loc[("01", "D1", "B3")]
    assert result["sampling_time_imputed"]
    assert result["time_min"] == 45
