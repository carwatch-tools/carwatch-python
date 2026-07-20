from pathlib import Path
from runpy import run_path


def test_import_example_runs_with_synthetic_data():
    example = Path(__file__).parents[1] / "examples" / "CARWatch_Import_Example.py"

    namespace = run_path(str(example))

    assert len(namespace["logs"]) == 5
    assert namespace["log_samples"]["sample_mismatch"].tolist() == [
        False,
        True,
        True,
        False,
    ]
    assert namespace["study_results"].index.name == "participant"
    assert namespace["study_results"].columns.names == ["day", "sample", "variable"]
    assert namespace["study_awakening"].index.names == ["participant", "day"]
    assert namespace["study_days"].columns.tolist() == [
        "date",
        "awakening_time",
        "awakening_type",
        "mismatch_summary",
    ]
    assert namespace["study_samples"].index.names == [
        "participant",
        "day",
        "scheduled_sample",
    ]
    assert namespace["mismatches"].index.get_level_values(
        "scheduled_sample"
    ).tolist() == ["B2", "B3"]
    assert namespace["merged_samples"]["cortisol"].tolist() == [1.0, 3.0, 2.0, 4.0]
    assert namespace["merged_samples"]["mismatch_corrected"].tolist() == [
        False,
        True,
        True,
        False,
    ]
