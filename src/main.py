from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.svm import SVC


ROOT_DIR = Path(__file__).resolve().parents[1]
TARGET_COLUMN = "position"


class CFG:
    train_path = ROOT_DIR / "data" / "processed" / "train_processed.csv"
    test_path = ROOT_DIR / "data" / "processed" / "test_processed.csv"
    metadata_path = ROOT_DIR / "data" / "processed" / "metadata.json"
    model_path = ROOT_DIR / "models" / "svm_position_model.joblib"
    metrics_path = ROOT_DIR / "reports" / "svm_metrics.json"
    predictions_path = ROOT_DIR / "reports" / "svm_test_predictions.csv"


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

    ignored_columns = {TARGET_COLUMN, "session_key", "meeting_key", "driver_number", "year"}
    return [col for col in train_df.columns if col not in ignored_columns]


def train_svm(
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

    min_class_count = int(y_train.value_counts().min())
    if min_class_count < 2:
        raise ValueError("Each position class needs at least 2 samples for CV training.")

    param_grid = {
        "C": [0.1, 1, 10, 100],
        "gamma": ["scale", 0.01, 0.1, 1],
        "kernel": ["rbf"],
        "class_weight": [None, "balanced"],
    }
    cv_splits = min(5, min_class_count)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    search = GridSearchCV(
        estimator=SVC(random_state=42),
        param_grid=param_grid,
        scoring="f1_macro",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(x_train, y_train)

    best_model = search.best_estimator_
    y_pred = best_model.predict(x_test)

    report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
    metrics = {
        "model": "SVC",
        "target": TARGET_COLUMN,
        "features_used": feature_columns,
        "n_train": int(len(train_df)),
        "n_test": int(len(test_df)),
        "cv_splits": cv_splits,
        "best_params": search.best_params_,
        "best_cv_f1_macro": float(search.best_score_),
        "test_accuracy": float(accuracy_score(y_test, y_pred)),
        "test_f1_macro": float(f1_score(y_test, y_pred, average="macro", zero_division=0)),
        "test_f1_weighted": float(
            f1_score(y_test, y_pred, average="weighted", zero_division=0)
        ),
        "test_mae_position": float(mean_absolute_error(y_test, y_pred)),
        "classification_report": report,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    model_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)

    joblib.dump(
        {
            "model": best_model,
            "feature_columns": feature_columns,
            "target_column": TARGET_COLUMN,
            "best_params": search.best_params_,
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
    predictions["predicted_position"] = y_pred
    predictions["absolute_error"] = (
        predictions[TARGET_COLUMN] - predictions["predicted_position"]
    ).abs()
    predictions.to_csv(predictions_path, index=False)

    return metrics


def main() -> None:
    metrics = train_svm(
        train_path=CFG.train_path,
        test_path=CFG.test_path,
        metadata_path=CFG.metadata_path,
        model_path=CFG.model_path,
        metrics_path=CFG.metrics_path,
        predictions_path=CFG.predictions_path,
    )

    print("Training finished.")
    print(f"Best params: {metrics['best_params']}")
    print(f"CV macro F1: {metrics['best_cv_f1_macro']:.4f}")
    print(f"Test accuracy: {metrics['test_accuracy']:.4f}")
    print(f"Test macro F1: {metrics['test_f1_macro']:.4f}")
    print(f"Test MAE position: {metrics['test_mae_position']:.4f}")


if __name__ == "__main__":
    main()
