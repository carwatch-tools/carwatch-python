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
    summary = cw.io.load_study_results(data_dir / "availability_cases_summary.csv")
    events = cw.logs.extract_sample_events_from_summary(summary)
    saliva = cw.io.load_saliva(data_dir / "availability_cases_saliva.csv")

    merged = cw.merge_saliva(events, saliva)
    result = merged.loc[("01", "D1", sample)]

    assert bool(result["sampling_event_recorded"]) is event_recorded
    assert bool(result["lab_value_available"]) is lab_value_available
