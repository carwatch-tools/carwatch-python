"""
CARWatch saliva analysis
========================

This example loads a Study Manager export and a laboratory file, corrects two
swapped tubes during the merge, and computes common cortisol-response metrics.
"""

from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

import carwatch as cw


study_data = pd.DataFrame(
    {
        "Study Name": ["demo"],
        "Participant ID": ["02"],
        "date_D1": ["2025-05-15"],
        "awakening_time_D1_app": ["06:13:30"],
        "sampling_time_D1_B1": ["06:13:55"],
        "sample_barcode_D1_B1": ["0010101"],
        "sample_scanned_D1_B1": ["B1"],
        "sampling_time_D1_B2": ["06:43:51"],
        "sample_barcode_D1_B2": ["0010103"],
        "sample_scanned_D1_B2": ["B3"],
        "sampling_time_D1_B3": ["06:58:52"],
        "sample_barcode_D1_B3": ["0010102"],
        "sample_scanned_D1_B3": ["B2"],
        "sampling_time_D1_B4": ["07:13:47"],
        "sample_barcode_D1_B4": ["0010104"],
        "sample_scanned_D1_B4": ["B4"],
    }
)
laboratory_data = pd.DataFrame(
    {
        "participant": ["02"] * 4,
        "day": ["D1"] * 4,
        "tube": ["B1", "B2", "B3", "B4"],
        "barcode": ["0010101", "0010102", "0010103", "0010104"],
        "cortisol": [1.0, 2.0, 3.0, 4.0],
    }
)

with TemporaryDirectory() as directory:
    directory = Path(directory)
    study_path = directory / "study_results.csv"
    saliva_path = directory / "saliva.csv"
    study_data.to_csv(study_path, index=False)
    laboratory_data.to_csv(saliva_path, index=False)

    study_results = cw.io.load_study_results(study_path)
    saliva = cw.io.load_saliva(
        saliva_path,
        sample_col="tube",
        barcode_col="barcode",
        value_cols="cortisol",
    )
    merged = cw.merge_saliva(study_results, saliva, allow_unmatched=False)
    features = cw.saliva.compute_features(merged, saliva_type="cortisol")

print(
    merged[
        [
            "sample_scanned",
            "saliva_sample",
            "cortisol",
            "mismatch_corrected",
        ]
    ]
)
print(features)
