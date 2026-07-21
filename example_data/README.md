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

summary = cw.io.load_study_manager_export(data_dir / "study_manager_summary.csv")
saliva = cw.io.load_saliva(data_dir / "saliva_samples.csv")
merged = cw.merge_saliva(summary, saliva)

merged[
    [
        "sampling_time",
        "cortisol",
        "sampling_event_recorded",
        "lab_value_available",
    ]
]
```

Impute the missing CARWatch time for B3 with a theoretical schedule relative
to awakening:

```python
imputed = cw.merge_saliva(
    summary,
    saliva,
    missing_carwatch_data="impute",
    sampling_schedule=[15, 30, 45, 60],
)
```

Only B3 is imputed. B1 already has complete information; B2 and B4 have no
laboratory value.
