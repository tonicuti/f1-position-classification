from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from lightgbm import LGBMRegressor
from sklearn.base import clone
from sklearn.metrics import (
    accuracy_score,
    mean_absolute_error,
    root_mean_squared_error,
)
from sklearn.model_selection import GroupKFold, KFold
from xgboost import XGBRegressor

ROOT_DIR = Path(__file__).resolve().parents[1]
TARGET_COLUMN = "position"
SAMPLE_WEIGHT_COLUMN = "sample_weight"
RANDOM_STATE = 42
ENSEMBLE_WEIGHTS = {"lightgbm": 0.6, "xgboost": 0.4}


class CFG:
    train_path = ROOT_DIR / "data" / "processed" / "train_processed.csv"
    test_path = ROOT_DIR / "data" / "processed" / "test_processed.csv"
    metadata_path = ROOT_DIR / "data" / "processed" / "metadata.json"
    model_path = ROOT_DIR / "models" / "lgbm_xgb_ensemble_position_model.joblib"
    metrics_path = ROOT_DIR / "reports" / "lgbm_xgb_ensemble_metrics.json"
    predictions_path = ROOT_DIR / "reports" / "lgbm_xgb_ensemble_test_predictions.csv"


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


def build_models() -> dict[str, object]:
    return {
        "lightgbm": LGBMRegressor(
            objective="regression",
            n_estimators=400,
            learning_rate=0.03,
            num_leaves=15,
            max_depth=4,
            min_child_samples=10,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            verbosity=-1,
        ),
        "xgboost": XGBRegressor(
            objective="reg:squarederror",
            n_estimators=400,
            learning_rate=0.03,
            max_depth=3,
            min_child_weight=3,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            eval_metric="mae",
        ),
    }


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


def cross_validate_models(
    models: dict[str, object],
    x_train: pd.DataFrame,
    y_train: pd.Series,
    sample_weight: pd.Series | None,
    train_df: pd.DataFrame,
) -> dict:
    cv, groups = make_cv(train_df)
    fold_metrics: list[dict] = []
    oof_predictions = {
        "lightgbm": np.zeros(len(x_train), dtype=float),
        "xgboost": np.zeros(len(x_train), dtype=float),
        "ensemble": np.zeros(len(x_train), dtype=float),
    }

    split_iterator = cv.split(x_train, y_train, groups=groups)
    for fold, (train_idx, valid_idx) in enumerate(split_iterator, start=1):
        x_fold_train = x_train.iloc[train_idx]
        y_fold_train = y_train.iloc[train_idx]
        x_fold_valid = x_train.iloc[valid_idx]
        y_fold_valid = y_train.iloc[valid_idx]
        fold_weight = sample_weight.iloc[train_idx] if sample_weight is not None else None

        fitted_models = {}
        for model_name, model in models.items():
            fitted_model = clone(model)
            fitted_model.fit(x_fold_train, y_fold_train, sample_weight=fold_weight)
            fitted_models[model_name] = fitted_model
            oof_predictions[model_name][valid_idx] = fitted_model.predict(x_fold_valid)

        oof_predictions["ensemble"][valid_idx] = (
            ENSEMBLE_WEIGHTS["lightgbm"] * oof_predictions["lightgbm"][valid_idx]
            + ENSEMBLE_WEIGHTS["xgboost"] * oof_predictions["xgboost"][valid_idx]
        )

        fold_metrics.append(
            {
                "fold": fold,
                "lightgbm": score_predictions(
                    y_fold_valid, oof_predictions["lightgbm"][valid_idx]
                ),
                "xgboost": score_predictions(
                    y_fold_valid, oof_predictions["xgboost"][valid_idx]
                ),
                "ensemble": score_predictions(
                    y_fold_valid, oof_predictions["ensemble"][valid_idx]
                ),
            }
        )

    return {
        "cv_type": type(cv).__name__,
        "cv_splits": cv.get_n_splits(),
        "fold_metrics": fold_metrics,
        "oof_metrics": {
            model_name: score_predictions(y_train, predictions)
            for model_name, predictions in oof_predictions.items()
        },
    }


def train_lightgbm_xgboost_ensemble(
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

    models = build_models()
    cv_results = cross_validate_models(models, x_train, y_train, sample_weight, train_df)

    fitted_models = {}
    test_raw_predictions = {}
    for model_name, model in models.items():
        fitted_model = clone(model)
        fitted_model.fit(x_train, y_train, sample_weight=sample_weight)
        fitted_models[model_name] = fitted_model
        test_raw_predictions[model_name] = fitted_model.predict(x_test)

    ensemble_raw = (
        ENSEMBLE_WEIGHTS["lightgbm"] * test_raw_predictions["lightgbm"]
        + ENSEMBLE_WEIGHTS["xgboost"] * test_raw_predictions["xgboost"]
    )
    ensemble_pred = clip_round_position(ensemble_raw)

    metrics = {
        "model": "LightGBMRegressor + XGBRegressor ensemble",
        "target": TARGET_COLUMN,
        "ensemble_weights": ENSEMBLE_WEIGHTS,
        "features_used": feature_columns,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "sample_weight_used": sample_weight is not None,
        "cross_validation": cv_results,
        "test_metrics": {
            "lightgbm": score_predictions(y_test, test_raw_predictions["lightgbm"]),
            "xgboost": score_predictions(y_test, test_raw_predictions["xgboost"]),
            "ensemble": score_predictions(y_test, ensemble_raw),
        },
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "models": fitted_models,
            "ensemble_weights": ENSEMBLE_WEIGHTS,
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
    predictions["lightgbm_raw_prediction"] = test_raw_predictions["lightgbm"]
    predictions["xgboost_raw_prediction"] = test_raw_predictions["xgboost"]
    predictions["ensemble_raw_prediction"] = ensemble_raw
    predictions["predicted_position"] = ensemble_pred
    predictions["absolute_error"] = (
        predictions[TARGET_COLUMN] - predictions["predicted_position"]
    ).abs()
    predictions.to_csv(predictions_path, index=False)

    return metrics


def main() -> None:
    metrics = train_lightgbm_xgboost_ensemble(
        train_path=CFG.train_path,
        test_path=CFG.test_path,
        metadata_path=CFG.metadata_path,
        model_path=CFG.model_path,
        metrics_path=CFG.metrics_path,
        predictions_path=CFG.predictions_path,
    )

    cv_metrics = metrics["cross_validation"]["oof_metrics"]
    test_metrics = metrics["test_metrics"]

    print("Training finished.")
    print(f"CV type: {metrics['cross_validation']['cv_type']}")
    print(f"CV splits: {metrics['cross_validation']['cv_splits']}")
    print(f"OOF LightGBM MAE: {cv_metrics['lightgbm']['mae_rounded']:.4f}")
    print(f"OOF XGBoost MAE: {cv_metrics['xgboost']['mae_rounded']:.4f}")
    print(f"OOF Ensemble MAE: {cv_metrics['ensemble']['mae_rounded']:.4f}")
    print(f"Test LightGBM MAE: {test_metrics['lightgbm']['mae_rounded']:.4f}")
    print(f"Test XGBoost MAE: {test_metrics['xgboost']['mae_rounded']:.4f}")
    print(f"Test Ensemble MAE: {test_metrics['ensemble']['mae_rounded']:.4f}")
    print(f"Model saved to: {CFG.model_path}")
    print(f"Metrics saved to: {CFG.metrics_path}")
    print(f"Predictions saved to: {CFG.predictions_path}")


if __name__ == "__main__":
    main()
