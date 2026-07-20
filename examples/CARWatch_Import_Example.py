"""
Load CARWatch logs and Study Manager results
============================================

This example demonstrates the package capabilities available after the raw-log
and Study Manager loaders. It uses synthetic data included with the repository.
"""

from pathlib import Path

import carwatch as cw


DATA_DIR = Path(__file__).parent / "data"

# Load the semicolon-delimited raw app log. Timestamps are converted from Unix
# milliseconds into timezone-aware pandas timestamps, while JSON payloads are
# retained as dictionaries.
logs = cw.io.load_logs(
    DATA_DIR / "carwatch_demo_VP01_20250515.csv",
    tz="Europe/Berlin",
)
print("Raw log events")
print(logs[["timestamp", "action", "source_file"]])

# Extract the sampling and awakening events needed for subsequent analyses.
samples = cw.logs.extract_samples(logs)
awakening = cw.logs.extract_awakening(logs)
print("\nExtracted sample scans")
print(
    samples[
        [
            "sampling_time",
            "sample",
            "sample_scanned",
            "barcode",
            "sample_mismatch",
        ]
    ]
)
print("\nExtracted awakening")
print(awakening)

# Normalize the wide Study Manager export into one row per expected sample.
study_results = cw.io.load_study_results(
    DATA_DIR / "study_results.csv",
    tz="Europe/Berlin",
)
print("\nNormalized Study Manager results")
print(
    study_results[
        [
            "sampling_time",
            "time",
            "barcode",
            "sample_scanned",
            "sample_mismatch",
        ]
    ]
)

# Keep the recorded mismatch information visible for audit and later correction.
mismatches = study_results.loc[study_results["sample_mismatch"]]
print("\nRecorded sample swaps")
print(mismatches[["barcode", "sample_scanned", "mismatch_summary"]])
