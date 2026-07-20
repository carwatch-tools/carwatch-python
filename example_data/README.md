# Sampling and laboratory availability cases

The two CSV files contain all four combinations of recorded CARWatch sampling
events and available laboratory values:

| Sample | Sampling event recorded | Laboratory value available |
|---|---:|---:|
| B1 | yes | yes |
| B2 | yes | no |
| B3 | no | yes |
| B4 | no | no |

Load and merge the files from a notebook started in the repository root:

```python
from pathlib import Path

import carwatch as cw

data_dir = Path("example_data")

summary = cw.io.load_study_results(data_dir / "availability_cases_summary.csv")
sample_events = cw.logs.extract_sample_events_from_summary(summary)
saliva = cw.io.load_saliva(data_dir / "availability_cases_saliva.csv")
merged = cw.merge_saliva(sample_events, saliva)

merged[
    [
        "sampling_time",
        "cortisol",
        "sampling_event_recorded",
        "lab_value_available",
    ]
]
```

The files only encode the four states. They do not prescribe filtering,
imputation, exclusion, or error handling.
