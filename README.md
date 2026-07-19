# carwatch

`carwatch` loads CARWatch app logs, normalizes CARWatch Study Manager exports,
imports laboratory saliva measurements, corrects accidentally swapped samples,
and computes common saliva-response metrics.

## Installation

The package currently targets Python 3.10 or newer. Install the repository for
development with [uv](https://docs.astral.sh/uv/):

```bash
git clone https://github.com/carwatch-tools/carwatch-python.git
cd carwatch-python
uv sync
```

## Analysis workflow

Load the Study Manager export. Its wide columns are converted to one row per
expected sample, indexed by study, participant, day, and sample.

```python
import carwatch as cw

study_results = cw.io.load_study_results("study_results.csv")
```

Load laboratory results in long format:

```python
saliva = cw.io.load_saliva(
    "saliva_long.csv",
    participant_col="Participant",
    day_col="Day",
    sample_col="Tube",
    barcode_col="Barcode",
    value_cols="Cortisol",
    day_map={"1": "D1"},
)
saliva = saliva.rename(columns={"Cortisol": "cortisol"})
```

Wide laboratory files are supported through an explicit mapping from source
columns to physical tube labels:

```python
saliva = cw.io.load_saliva(
    "saliva_wide.csv",
    format="wide",
    participant_col="Participant",
    day_col="Day",
    sample_columns={"Cort_1": "B1", "Cort_2": "B2", "Cort_3": "B3"},
    value_name="cortisol",
    day_map={"1": "D1"},
)
```

Merge and compute features:

```python
merged = cw.merge_saliva(study_results, saliva)
features = cw.saliva.compute_features(merged, saliva_type="cortisol")
```

`merge_saliva()` matches by barcode first and by the physical tube recorded in
`sample_scanned` second. Laboratory values from swapped tubes are therefore
placed at the intended sample positions and inherit the correct sampling times.
The output records `match_method`, `merge_status`, and `mismatch_corrected` for
auditing. Use `match_by="expected_sample"` only when the laboratory data have
already been corrected.

## Raw app logs

Raw CSV files, ZIP archives, directories, and sequences of paths can be loaded
through the same API:

```python
logs = cw.io.load_logs("raw_logs.zip", tz="Europe/Berlin")
samples = cw.logs.extract_samples(logs)
awakening = cw.logs.extract_awakening(logs)
```

## Saliva metrics

The `carwatch.saliva` module provides `auc`, `slope`, `initial_value`,
`max_value`, `max_increase`, `standard_features`, `mean_se`, and
`compute_features`. AUC calculations implement the trapezoidal formulas from
Pruessner et al. (2003) and accept the participant-specific `time` values from
CARWatch or explicit sampling times.

## Development

```bash
uv run poe format
uv run poe ci_check
uv run poe test
uv run poe docs_clean
```

The local `playground/` directory is intentionally ignored because it contains
study data. Tests and documentation use synthetic records.
