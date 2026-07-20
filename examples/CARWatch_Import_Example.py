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
log_samples = cw.logs.extract_sample_events_from_raw_logs(logs)
log_awakening = cw.logs.extract_awakening_events_from_raw_logs(logs)
print("\nExtracted sample scans")
print(
    log_samples[
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
print(log_awakening)

# Load the Study Manager export with one participant per row. The columns
# explicitly encode day, sample, and variable.
study_results = cw.io.load_study_results(
    DATA_DIR / "study_results.csv",
    tz="Europe/Berlin",
)
print("\nWide Study Manager results")
print(study_results)

# Extract focused tables only when day- or sample-level analysis is needed.
study_awakening = cw.logs.extract_awakening_events_from_summary(study_results)
study_days = cw.logs.extract_day_summary_from_summary(study_results)
study_samples = cw.logs.extract_sample_events_from_summary(study_results)
print("\nStudy Manager awakening information")
print(study_awakening)
print("\nStudy Manager sample information")
print(study_samples)

# Keep the recorded mismatch information visible for audit and later correction.
mismatches = study_samples.loc[study_samples["sample_mismatch"].fillna(False)]
print("\nRecorded sample swaps")
print(mismatches[["barcode", "sample_scanned"]])
print("\nRecorded daily mismatch summary")
print(study_days[["mismatch_summary"]])

# Merge laboratory values by the physical tube that was scanned. The values
# for B2 and B3 are therefore assigned to their actual sampling positions.
saliva = cw.io.load_saliva(DATA_DIR / "saliva_samples.csv")
merged_samples = cw.merge_saliva(study_samples, saliva)
print("\nSample information merged with saliva measurements")
print(
    merged_samples[
        [
            "sampling_time",
            "sample_scanned",
            "saliva_sample",
            "cortisol",
            "mismatch_corrected",
        ]
    ]
)
