from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ID_COLUMNS = ["session_key", "meeting_key", "driver_number", "year"]
TARGET_COLUMN = "position"
SAMPLE_WEIGHT_COLUMN = "sample_weight"

# These columns are observed during/after the race. Keep them out by default so
# the model does not learn the finishing position from race-state proxies.
LEAKAGE_COLUMNS = [
    "feature_avg_race_position",
    "feature_best_race_position",
    "feature_worst_race_position",
    "feature_avg_gap_to_leader",
    "feature_avg_interval_to_car_ahead",
]


@dataclass(frozen=True)
class PreprocessConfig:
    input_path: str = "data/final_data.csv"
    output_dir: str = "data/processed"
    target_column: str = TARGET_COLUMN
    test_size: float = 0.2
    random_state: int = 42
    drop_leakage: bool = True
    clip_lower_quantile: float = 0.01
    clip_upper_quantile: float = 0.99


class StdMissingToZero(BaseEstimator, TransformerMixin):
    """Fill missing std-like features with zero before median imputation."""

    def __init__(self, std_columns: list[str] | None = None):
        self.std_columns = std_columns or []

    def fit(self, X: pd.DataFrame, y: pd.Series | None = None):
        self.columns_ = list(X.columns)
        self.std_columns_ = [c for c in self.std_columns if c in self.columns_]
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X, columns=self.columns_).copy()
        if self.std_columns_:
            X[self.std_columns_] = X[self.std_columns_].fillna(0)
        return X

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray(getattr(self, "columns_", []), dtype=object)
        return np.asarray(input_features, dtype=object)


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Winsorize numeric features with train-set quantile bounds."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X: np.ndarray, y: pd.Series | None = None):
        X_df = pd.DataFrame(X)
        self.lower_bounds_ = X_df.quantile(self.lower_quantile)
        self.upper_bounds_ = X_df.quantile(self.upper_quantile)
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X_df = pd.DataFrame(X)
        X_df = X_df.clip(self.lower_bounds_, self.upper_bounds_, axis=1)
        return X_df.to_numpy()

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.asarray([], dtype=object)
        return np.asarray(input_features, dtype=object)


def normalize_compound(series: pd.Series) -> pd.Series:
    return (
        series.astype("string")
        .str.strip()
        .str.upper()
        .replace({"NAN": pd.NA, "": pd.NA, "UNKNOWN": pd.NA})
        .fillna("UNKNOWN")
    )


def load_data(input_path: str | Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)
    return df.drop(columns=["Unnamed: 0"], errors="ignore")


def make_feature_frame(df: pd.DataFrame, config: PreprocessConfig) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    df = df.copy()
    df = df.dropna(subset=[config.target_column]).reset_index(drop=True)

    if "feature_most_used_compound" in df.columns:
        df["feature_most_used_compound"] = normalize_compound(df["feature_most_used_compound"])

    drop_columns = [config.target_column]
    if config.drop_leakage:
        drop_columns.extend([c for c in LEAKAGE_COLUMNS if c in df.columns])

    X = df.drop(columns=drop_columns, errors="ignore")
    y = df[config.target_column].astype(int)
    ids = df[[c for c in ID_COLUMNS if c in df.columns]].copy()
    return X, y, ids


def build_preprocessor(X: pd.DataFrame, config: PreprocessConfig) -> ColumnTransformer:
    feature_id_cols = [c for c in ID_COLUMNS if c in X.columns]
    categorical_cols = X.select_dtypes(include=["object", "category", "string"]).columns.tolist()
    numeric_cols = [
        c
        for c in X.select_dtypes(include="number").columns.tolist()
        if c not in feature_id_cols
    ]
    std_cols = [c for c in numeric_cols if c.endswith("_std")]

    numeric_pipeline = Pipeline(
        steps=[
            ("std_missing_to_zero", StdMissingToZero(std_columns=std_cols)),
            ("imputer", SimpleImputer(strategy="median")),
            (
                "clipper",
                QuantileClipper(
                    lower_quantile=config.clip_lower_quantile,
                    upper_quantile=config.clip_upper_quantile,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="UNKNOWN")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_cols),
            ("cat", categorical_pipeline, categorical_cols),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def split_by_session(
    X: pd.DataFrame,
    y: pd.Series,
    ids: pd.DataFrame,
    config: PreprocessConfig,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, pd.DataFrame]:
    if "session_key" in ids.columns:
        groups = ids["session_key"]
    else:
        groups = np.arange(len(X))

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=config.test_size,
        random_state=config.random_state,
    )
    train_idx, test_idx = next(splitter.split(X, y, groups=groups))

    return (
        X.iloc[train_idx].reset_index(drop=True),
        X.iloc[test_idx].reset_index(drop=True),
        y.iloc[train_idx].reset_index(drop=True),
        y.iloc[test_idx].reset_index(drop=True),
        ids.iloc[train_idx].reset_index(drop=True),
        ids.iloc[test_idx].reset_index(drop=True),
    )


def compute_balanced_sample_weights(y: pd.Series) -> pd.Series:
    """Compute inverse-frequency class weights for imbalanced targets."""
    class_counts = y.value_counts()
    n_samples = len(y)
    n_classes = class_counts.size
    weights = y.map(lambda cls: n_samples / (n_classes * class_counts.loc[cls]))
    return weights.astype(float)


def target_distribution(y: pd.Series) -> dict[str, int]:
    return {str(k): int(v) for k, v in y.value_counts().sort_index().items()}


def transform_to_frame(
    preprocessor: ColumnTransformer,
    X: pd.DataFrame,
    y: pd.Series,
    ids: pd.DataFrame,
) -> pd.DataFrame:
    features = preprocessor.transform(X)
    feature_names = preprocessor.get_feature_names_out()
    feature_df = pd.DataFrame(features, columns=feature_names, index=X.index)
    return pd.concat([ids.reset_index(drop=True), feature_df, y.rename(TARGET_COLUMN)], axis=1)


def run_preprocessing(config: PreprocessConfig) -> dict[str, object]:
    input_path = Path(config.input_path)
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_data(input_path)
    X, y, ids = make_feature_frame(raw_df, config)
    X_train, X_test, y_train, y_test, ids_train, ids_test = split_by_session(X, y, ids, config)

    preprocessor = build_preprocessor(X_train, config)
    preprocessor.fit(X_train, y_train)

    train_processed = transform_to_frame(preprocessor, X_train, y_train, ids_train)
    test_processed = transform_to_frame(preprocessor, X_test, y_test, ids_test)
    train_processed[SAMPLE_WEIGHT_COLUMN] = compute_balanced_sample_weights(y_train).reset_index(drop=True)
    test_processed[SAMPLE_WEIGHT_COLUMN] = 1.0

    train_path = output_dir / "train_processed.csv"
    test_path = output_dir / "test_processed.csv"
    pipeline_path = output_dir / "preprocessor.joblib"
    metadata_path = output_dir / "metadata.json"

    train_processed.to_csv(train_path, index=False)
    test_processed.to_csv(test_path, index=False)
    joblib.dump(preprocessor, pipeline_path)

    metadata = {
        "config": asdict(config),
        "raw_shape": list(raw_df.shape),
        "modeling_shape_before_transform": [len(X), X.shape[1]],
        "train_shape": list(train_processed.shape),
        "test_shape": list(test_processed.shape),
        "dropped_target_rows": int(raw_df[config.target_column].isna().sum()),
        "dropped_leakage_columns": [c for c in LEAKAGE_COLUMNS if config.drop_leakage and c in raw_df.columns],
        "id_columns_kept_for_traceability": [c for c in ID_COLUMNS if c in raw_df.columns],
        "imbalance_strategy": "balanced_sample_weight_train_only",
        "sample_weight_column": SAMPLE_WEIGHT_COLUMN,
        "train_target_distribution": target_distribution(y_train),
        "test_target_distribution": target_distribution(y_test),
        "feature_columns_after_transform": preprocessor.get_feature_names_out().tolist(),
    }

    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return metadata


def parse_args() -> PreprocessConfig:
    parser = argparse.ArgumentParser(description="Preprocess DS102 F1 final_data.csv")
    parser.add_argument("--input-path", default=PreprocessConfig.input_path)
    parser.add_argument("--output-dir", default=PreprocessConfig.output_dir)
    parser.add_argument("--test-size", type=float, default=PreprocessConfig.test_size)
    parser.add_argument("--random-state", type=int, default=PreprocessConfig.random_state)
    parser.add_argument("--keep-leakage", action="store_true")
    parser.add_argument("--clip-lower-quantile", type=float, default=PreprocessConfig.clip_lower_quantile)
    parser.add_argument("--clip-upper-quantile", type=float, default=PreprocessConfig.clip_upper_quantile)
    args = parser.parse_args()

    return PreprocessConfig(
        input_path=args.input_path,
        output_dir=args.output_dir,
        test_size=args.test_size,
        random_state=args.random_state,
        drop_leakage=not args.keep_leakage,
        clip_lower_quantile=args.clip_lower_quantile,
        clip_upper_quantile=args.clip_upper_quantile,
    )


if __name__ == "__main__":
    result = run_preprocessing(parse_args())
    print(json.dumps({k: v for k, v in result.items() if k != "feature_columns_after_transform"}, indent=2))
