from pathlib import Path
from runpy import run_path


def test_import_example_runs_with_synthetic_data():
    example = Path(__file__).parents[1] / "examples" / "CARWatch_Import_Example.py"

    namespace = run_path(str(example))

    assert len(namespace["logs"]) == 5
    assert namespace["samples"]["sample_mismatch"].tolist() == [
        False,
        True,
        True,
        False,
    ]
    assert namespace["study_results"].index.names == [
        "study",
        "participant",
        "day",
        "sample",
    ]
    assert namespace["mismatches"].index.get_level_values("sample").tolist() == [
        "B2",
        "B3",
    ]
