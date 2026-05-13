# Preprocessing

Pipeline based on `EDA/EDA.ipynb`:

- drop `Unnamed: 0`
- drop rows without `position` for supervised modeling
- keep `session_key`, `meeting_key`, `driver_number`, and `year` only for traceability
- drop likely leakage race-state columns by default
- normalize `feature_most_used_compound`
- fill missing std features with `0`, then median-impute numeric features
- clip numeric outliers with train-set 1% and 99% quantiles
- standard-scale numeric features
- one-hot encode categorical features
- split train/test by `session_key` to avoid session overlap

Run:

```powershell
python preprocessing/preprocess.py
```

Outputs:

- `data/processed/train_processed.csv`
- `data/processed/test_processed.csv`
- `data/processed/preprocessor.joblib`
- `data/processed/metadata.json`

Use `--keep-leakage` only if your modeling goal intentionally uses in-race features.
