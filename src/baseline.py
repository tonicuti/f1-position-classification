from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold, KFold

ROOT_DIR = Path(__file__).resolve().parents[1]
TARGET_COLUMN = "position"
SAMPLE_WEIGHT_COLUMN = "sample_weight"
RANDOM_STATE = 42


class CFG:
    train_path = ROOT_DIR / "data" / "processed" / "train_processed.csv"
    test_path = ROOT_DIR / "data" / "processed" / "test_processed.csv"
    metadata_path = ROOT_DIR / "data" / "processed" / "metadata.json"
    model_path = ROOT_DIR / "models" / "random_forest_position_model.joblib"
    metrics_path = ROOT_DIR / "reports" / "random_forest_metrics.json"
    predictions_path = ROOT_DIR / "reports" / "random_forest_test_predictions.csv"


def load_feature_columns(metadata_path: Path, train_df: pd.DataFrame) -> list[str]:
    """Use preprocessing metadata when available, otherwise infer usable features."""
    if metadata_path.exists():
        with metadata_path.open("r", encoding="utf-8") as file:
            metadata = json.load(file)
        feature_columns = metadata.get("feature_columns_after_transform", [])
        missing_columns = [col for col in feature_columns if col not in train_df.columns]
        if missing_columns:
            raise ValueError(
                "Metadata references columns that are missing in training data: "
                f"{missing_columns}"
            )
        return feature_columns

    ignored_columns = {
        TARGET_COLUMN,
        SAMPLE_WEIGHT_COLUMN,
        "session_key",
        "meeting_key",
        "driver_number",
        "year",
    }
    return [col for col in train_df.columns if col not in ignored_columns]


def build_model() -> RandomForestRegressor:
    return RandomForestRegressor(
        n_estimators=500,
        max_depth=None,
        min_samples_split=4,
        min_samples_leaf=2,
        max_features="sqrt",
        bootstrap=True,
        n_jobs=-1,
        random_state=RANDOM_STATE,
    )


def make_cv(train_df: pd.DataFrame, n_splits: int = 5):
    if "session_key" in train_df.columns and train_df["session_key"].nunique() >= n_splits:
        return GroupKFold(n_splits=n_splits), train_df["session_key"]
    return KFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE), None


def clip_round_position(predictions: np.ndarray) -> np.ndarray:
    return np.clip(np.rint(predictions), 1, 20).astype(int)


def score_predictions(y_true: pd.Series, raw_pred: np.ndarray) -> dict[str, float]:
    rounded_pred = clip_round_position(raw_pred)
    return {
        "mae_raw": float(mean_absolute_error(y_true, raw_pred)),
        "rmse_raw": float(root_mean_squared_error(y_true, raw_pred)),
        "mae_rounded": float(mean_absolute_error(y_true, rounded_pred)),
        "accuracy_rounded": float(accuracy_score(y_true, rounded_pred)),
    }


def cross_validate_model(
    model: RandomForestRegressor,
    x_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: pd.Series | None,
    train_df: pd.DataFrame,
) -> dict:
    cv, groups = make_cv(train_df)
    fold_metrics: list[dict] = []
    oof_predictions = np.zeros(len(x_train), dtype=float)

    split_iterator = cv.split(x_train, y_train, groups=groups)
    for fold, (train_idx, valid_idx) in enumerate(split_iterator, start=1):
        x_fold_train = x_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        x_fold_valid = x_train.iloc[valid_idx]
        y_fold_valid = y_train.iloc[valid_idx]
        fold_weight = sample_weight.iloc[train_idx] if sample_weight is not None else None

        fitted_model = clone(model)
        fitted_model.fit(x_fold_train, y_fold_train, sample_weight=fold_weight)
        oof_predictions[valid_idx] = fitted_model.predict(x_fold_valid)

        fold_metrics.append(
            {
                "fold": fold,
                "random_forest": score_predictions(
                    y_fold_valid, oof_predictions[valid_idx]
                ),
            }
        )

    return {
        "cv_type": type(cv).__name__,
        "cv_splits": cv.get_n_splits(),
        "fold_metrics": fold_metrics,
        "oof_metrics": {
            "random_forest": score_predictions(y_train, oof_predictions),
        },
    }


def train_random_forest_baseline(
    train_path: Path,
    test_path: Path,
    metadata_path: Path,
    model_path: Path,
    metrics_path: Path,
    predictions_path: Path,
) -> dict:
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    if TARGET_COLUMN not in train_df.columns or TARGET_COLUMN not in test_df.columns:
        raise ValueError(f"Both train and test files must contain '{TARGET_COLUMN}'.")

    feature_columns = load_feature_columns(metadata_path, train_df)
    x_train = train_df[feature_columns]
    y_train = train_df[TARGET_COLUMN].astype(int)
    x_test = test_df[feature_columns]
    y_test = test_df[TARGET_COLUMN].astype(int)
    sample_weight = (
        train_df[SAMPLE_WEIGHT_COLUMN]
        if SAMPLE_WEIGHT_COLUMN in train_df.columns
        else None
    )

    model = build_model()
    cv_results = cross_validate_model(model, x_train, y_train, sample_weight, train_df)

    fitted_model = clone(model)
    fitted_model.fit(x_train, y_train, sample_weight=sample_weight)
    test_raw_predictions = fitted_model.predict(x_test)
    test_predictions = clip_round_position(test_raw_predictions)

    metrics = {
        "model": "RandomForestRegressor baseline",
        "target": TARGET_COLUMN,
        "features_used": feature_columns,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "sample_weight_used": sample_weight is not None,
        "cross_validation": cv_results,
        "test_metrics": {
            "random_forest": score_predictions(y_test, test_raw_predictions),
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": fitted_model,
            "feature_columns": feature_columns,
            "target_column": TARGET_COLUMN,
        },
        model_path,
    )

    with metrics_path.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    prediction_columns = [
        col
        for col in ["session_key", "meeting_key", "driver_number", "year", TARGET_COLUMN]
        if col in test_df.columns
    ]
    predictions = test_df[prediction_columns].copy()
    predictions["random_forest_raw_prediction"] = test_raw_predictions
    predictions["predicted_position"] = test_predictions
    predictions["absolute_error"] = (
        predictions[TARGET_COLUMN] - predictions["predicted_position"]
    ).abs()
    predictions.to_csv(predictions_path, index=False)

    return metrics


def main() -> None:
    metrics = train_random_forest_baseline(
        train_path=CFG.train_path,
        test_path=CFG.test_path,
        metadata_path=CFG.metadata_path,
        model_path=CFG.model_path,
        metrics_path=CFG.metrics_path,
        predictions_path=CFG.predictions_path,
    )

    cv_metrics = metrics["cross_validation"]["oof_metrics"]["random_forest"]
    test_metrics = metrics["test_metrics"]["random_forest"]

    print("Training finished.")
    print(f"CV type: {metrics['cross_validation']['cv_type']}")
    print(f"CV splits: {metrics['cross_validation']['cv_splits']}")
    print(f"OOF Random Forest MAE: {cv_metrics['mae_rounded']:.4f}")
    print(f"OOF Random Forest accuracy: {cv_metrics['accuracy_rounded']:.4f}")
    print(f"Test Random Forest MAE: {test_metrics['mae_rounded']:.4f}")
    print(f"Test Random Forest accuracy: {test_metrics['accuracy_rounded']:.4f}")
    print(f"Model saved to: {CFG.model_path}")
    print(f"Metrics saved to: {CFG.metrics_path}")
    print(f"Predictions saved to: {CFG.predictions_path}")


if __name__ == "__main__":
    main()
