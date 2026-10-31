"""Leakage-aware late-order classification, evaluation, and inference."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    fbeta_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import (
    RandomizedSearchCV,
    StratifiedKFold,
    train_test_split,
)
from sklearn.pipeline import Pipeline
from sklearn.base import clone

from config import (
    CLASSIFICATION_FALSE_NEGATIVE_COST,
    CLASSIFICATION_FALSE_POSITIVE_COST,
    CLASSIFICATION_IMPORTANCE_PATH,
    CLASSIFICATION_MODEL_PATH,
    CLASSIFICATION_RANDOM_SEED,
    CLASSIFICATION_REPORT_PATH,
    CLASSIFICATION_TARGET_COLUMN,
    CLASSIFICATION_TEST_PREDICTIONS_PATH,
    CLASSIFICATION_THRESHOLD_BETA,
    CLASSIFICATION_VALIDATION_PREDICTIONS_PATH,
    PROCESSED_DATA_PATH,
    REGRESSION_FEATURE_COLUMNS,
    REGRESSION_FEATURE_POLICY,
    SLA_THRESHOLD_MINUTES,
    TARGET_COLUMN,
)
from src.data.feature_engineering import (
    label_late_orders,
    normalize_traffic,
    normalize_weather,
)
from src.data.loader import load_orders, sha256_file
from src.models.regression import (
    MISSING_CATEGORY,
    build_order_features,
    create_preprocessor,
    load_regression_dataset,
)

MODEL_VERSION = "1.0.0"
DEFAULT_TEST_FRACTION = 0.2
DEFAULT_VALIDATION_FRACTION = 0.2
DEFAULT_CV_FOLDS = 3
DEFAULT_SEARCH_ITERATIONS = 3
CALIBRATION_CV_FOLDS = 3
THRESHOLD_GRID = tuple(round(value, 2) for value in np.arange(0.05, 1.0, 0.05))
_MISSING_TEXT = frozenset({"", "nan", "na", "n/a", "null", "none"})

try:
    from xgboost import XGBClassifier
except ImportError:
    XGBClassifier = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class ClassificationDataset:
    """Order-placement features, SLA labels, and retrospective context."""

    features: pd.DataFrame
    target: pd.Series
    source_row_numbers: pd.Series
    context: pd.DataFrame


def load_classification_dataset(
    path: Path = PROCESSED_DATA_PATH,
) -> ClassificationDataset:
    """Build the binary target from the configured SLA and reuse PS4 features."""
    regression_data = load_regression_dataset(path)
    labels = label_late_orders(
        regression_data.target, sla_threshold_minutes=SLA_THRESHOLD_MINUTES
    )
    valid_labels = labels.notna()
    if not valid_labels.all():
        raise ValueError("Classification target contains missing delivery outcomes.")

    source = load_orders(path)
    positions = regression_data.source_row_numbers.to_numpy(dtype="int64") - 2
    source_context = source.iloc[positions]
    context = pd.DataFrame(
        {
            "weather": normalize_weather(source_context["Weatherconditions"]),
            "traffic": normalize_traffic(source_context["Road_traffic_density"]),
        }
    ).reset_index(drop=True)
    for column in context.columns:
        values = context[column].astype("string").str.strip()
        missing = values.isna() | values.str.casefold().isin(_MISSING_TEXT)
        context[column] = values.mask(missing, MISSING_CATEGORY).astype("string")

    return ClassificationDataset(
        features=regression_data.features.reset_index(drop=True),
        target=labels.astype("int8").reset_index(drop=True).rename(
            CLASSIFICATION_TARGET_COLUMN
        ),
        source_row_numbers=regression_data.source_row_numbers.reset_index(drop=True),
        context=context,
    )


def stratified_train_test_indices(
    target: pd.Series | np.ndarray,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    random_seed: int = CLASSIFICATION_RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return reproducible disjoint positions while preserving class proportions."""
    labels = np.asarray(target, dtype="int8")
    if labels.ndim != 1 or len(labels) < 4:
        raise ValueError("At least four one-dimensional target rows are required.")
    if not set(np.unique(labels)).issubset({0, 1}) or len(np.unique(labels)) != 2:
        raise ValueError("The classification target must contain both binary classes.")
    if not 0.0 < test_fraction < 1.0:
        raise ValueError("test_fraction must be between zero and one.")
    positions = np.arange(len(labels))
    try:
        return train_test_split(
            positions,
            test_size=test_fraction,
            random_state=random_seed,
            shuffle=True,
            stratify=labels,
        )
    except ValueError as exc:
        raise ValueError(f"Unable to create a stratified split: {exc}") from exc


def stratified_development_validation_indices(
    target: pd.Series | np.ndarray,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    random_seed: int = CLASSIFICATION_RANDOM_SEED + 1,
) -> tuple[np.ndarray, np.ndarray]:
    """Split training rows for tuning and threshold selection without test access."""
    return stratified_train_test_indices(target, validation_fraction, random_seed)


def build_classification_pipeline(name: str, estimator: Any) -> Pipeline:
    """Reuse the regression imputation and category-encoding policy."""
    preprocessor = create_preprocessor(
        name,
        scale_numeric=name == "Logistic Regression",
    )
    return Pipeline(
        [
            ("preprocessor", preprocessor),
            ("classifier", estimator),
        ]
    )


def _curve_records(
    actual: pd.Series | np.ndarray, probabilities: pd.Series | np.ndarray
) -> dict[str, list[float]]:
    false_positive_rate, true_positive_rate, _ = roc_curve(actual, probabilities)
    precision, recall, _ = precision_recall_curve(actual, probabilities)
    return {
        "roc_false_positive_rate": false_positive_rate.astype(float).tolist(),
        "roc_true_positive_rate": true_positive_rate.astype(float).tolist(),
        "precision_recall_precision": precision.astype(float).tolist(),
        "precision_recall_recall": recall.astype(float).tolist(),
    }


def _candidate_models(positive_weight: float) -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    candidates: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "Logistic Regression": (
            LogisticRegression(
                solver="liblinear",
                max_iter=1000,
                random_state=CLASSIFICATION_RANDOM_SEED,
            ),
            {
                "classifier__C": [0.1, 1.0, 10.0],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "Random Forest": (
            RandomForestClassifier(
                random_state=CLASSIFICATION_RANDOM_SEED,
                n_jobs=4,
            ),
            {
                "classifier__n_estimators": [100, 160],
                "classifier__max_depth": [None, 16],
                "classifier__min_samples_leaf": [1, 3],
                "classifier__class_weight": [None, "balanced"],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingClassifier(random_state=CLASSIFICATION_RANDOM_SEED),
            {
                "classifier__n_estimators": [100, 160],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [2, 3],
            },
        ),
    }
    if XGBClassifier is not None:
        candidates["XGBoost"] = (
            XGBClassifier(
                objective="binary:logistic",
                eval_metric="logloss",
                random_state=CLASSIFICATION_RANDOM_SEED,
                n_jobs=4,
                tree_method="hist",
                verbosity=0,
            ),
            {
                "classifier__n_estimators": [100, 160],
                "classifier__learning_rate": [0.05, 0.1],
                "classifier__max_depth": [3, 5],
                "classifier__scale_pos_weight": [1.0, positive_weight],
            },
        )
    return candidates


def _positive_probability(estimator: Any, features: pd.DataFrame) -> np.ndarray:
    """Return validated probabilities for class 1."""
    probabilities = np.asarray(estimator.predict_proba(features), dtype="float64")
    if probabilities.ndim != 2 or probabilities.shape[1] != 2:
        raise ValueError("Classifier must return probabilities for both binary classes.")
    positive_index = list(estimator.classes_).index(1)
    positive = probabilities[:, positive_index]
    if not np.isfinite(positive).all() or ((positive < 0) | (positive > 1)).any():
        raise ValueError("Classifier returned an invalid positive-class probability.")
    return positive


def expected_decision_cost(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
    *,
    false_negative_cost: float,
    false_positive_cost: float,
) -> dict[str, float]:
    """Calculate caller-configured per-order costs; no project costs are assumed."""
    if (
        not math.isfinite(false_negative_cost)
        or not math.isfinite(false_positive_cost)
        or false_negative_cost < 0
        or false_positive_cost < 0
    ):
        raise ValueError("Decision costs must be finite, non-negative values.")
    actual_values = np.asarray(actual, dtype="int8")
    predicted_values = np.asarray(predicted, dtype="int8")
    if (
        actual_values.ndim != 1
        or predicted_values.ndim != 1
        or len(actual_values) == 0
        or len(actual_values) != len(predicted_values)
    ):
        raise ValueError("Actual and predicted labels must be aligned non-empty vectors.")
    _, fp, fn, _ = confusion_matrix(
        actual_values, predicted_values, labels=[0, 1]
    ).ravel()
    total_cost = float(fn * false_negative_cost + fp * false_positive_cost)
    return {
        "false_negative_cost": float(false_negative_cost),
        "false_positive_cost": float(false_positive_cost),
        "total_cost": total_cost,
        "mean_cost_per_order": total_cost / len(actual_values),
    }


def _classification_metrics(
    actual: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Calculate thresholded operational metrics and probability-ranking metrics."""
    actual_values = np.asarray(actual, dtype="int8")
    probability_values = np.asarray(probabilities, dtype="float64")
    if (
        actual_values.ndim != 1
        or probability_values.ndim != 1
        or len(actual_values) == 0
        or len(actual_values) != len(probability_values)
    ):
        raise ValueError("Actual labels and probabilities must be aligned non-empty vectors.")
    if not set(np.unique(actual_values)).issubset({0, 1}):
        raise ValueError("Actual labels must contain only 0 and 1.")
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("threshold must be between zero and one.")
    if not np.isfinite(probability_values).all() or (
        (probability_values < 0) | (probability_values > 1)
    ).any():
        raise ValueError("Probabilities must be finite values between zero and one.")
    predicted = (probability_values >= threshold).astype("int8")
    tn, fp, fn, tp = confusion_matrix(
        actual_values, predicted, labels=[0, 1]
    ).ravel()
    specificity = float(tn / (tn + fp)) if tn + fp else 0.0
    return {
        "threshold": float(threshold),
        "order_count": int(len(actual_values)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "flagged_orders": int(predicted.sum()),
        "accuracy": float(accuracy_score(actual_values, predicted)),
        "precision": float(precision_score(actual_values, predicted, zero_division=0)),
        "recall": float(recall_score(actual_values, predicted, zero_division=0)),
        "f1": float(f1_score(actual_values, predicted, zero_division=0)),
        "f_beta": float(
            fbeta_score(
                actual_values,
                predicted,
                beta=CLASSIFICATION_THRESHOLD_BETA,
                zero_division=0,
            )
        ),
        "specificity": specificity,
        "balanced_accuracy": float(balanced_accuracy_score(actual_values, predicted)),
        "roc_auc": float(roc_auc_score(actual_values, probability_values)),
        "pr_auc": float(average_precision_score(actual_values, probability_values)),
        "brier_score": float(brier_score_loss(actual_values, probability_values)),
        "confusion_matrix": {
            "labels": ["On time", "Late"],
            "values": [[int(tn), int(fp)], [int(fn), int(tp)]],
        },
        "classification_report": classification_report(
            actual_values,
            predicted,
            labels=[0, 1],
            target_names=["On time", "Late"],
            output_dict=True,
            zero_division=0,
        ),
    }


def evaluate_classification_metrics(
    actual: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    threshold: float,
) -> dict[str, Any]:
    """Public metric entry point for validation and interactive dashboard use."""
    return _classification_metrics(actual, probabilities, threshold)


def threshold_comparison(
    actual: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    thresholds: tuple[float, ...] = THRESHOLD_GRID,
) -> list[dict[str, Any]]:
    """Measure intervention volume and error trade-offs across thresholds."""
    if not thresholds:
        raise ValueError("At least one threshold is required.")
    rows = []
    for threshold in thresholds:
        metrics = _classification_metrics(actual, probabilities, threshold)
        rows.append(
            {
                key: metrics[key]
                for key in (
                    "threshold",
                    "flagged_orders",
                    "true_positives",
                    "false_positives",
                    "false_negatives",
                    "precision",
                    "recall",
                    "f1",
                    "f_beta",
                    "specificity",
                )
            }
        )
    return rows


def select_operating_threshold(
    actual: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    *,
    beta: float = CLASSIFICATION_THRESHOLD_BETA,
    false_negative_cost: float | None = None,
    false_positive_cost: float | None = None,
) -> tuple[float, list[dict[str, Any]]]:
    """Choose a validation threshold by F-beta or explicitly supplied error costs."""
    if not math.isfinite(beta) or beta <= 0:
        raise ValueError("beta must be a positive finite number.")
    if (false_negative_cost is None) != (false_positive_cost is None):
        raise ValueError("Supply both false-negative and false-positive costs, or neither.")
    rows = threshold_comparison(actual, probabilities)
    if false_negative_cost is not None and false_positive_cost is not None:
        def rank(row: dict[str, Any]) -> tuple[float, float, float]:
            costs = expected_decision_cost(
                actual,
                np.asarray(probabilities) >= row["threshold"],
                false_negative_cost=false_negative_cost,
                false_positive_cost=false_positive_cost,
            )
            return (
                costs["mean_cost_per_order"],
                -row["recall"],
                row["flagged_orders"],
            )

        selected = min(rows, key=rank)
    else:
        for row in rows:
            row["selection_f_beta"] = float(
                fbeta_score(
                    actual,
                    np.asarray(probabilities) >= row["threshold"],
                    beta=beta,
                    zero_division=0,
                )
            )
        selected = max(
            rows,
            key=lambda row: (
                row["selection_f_beta"],
                row["precision"],
                -row["flagged_orders"],
                row["threshold"],
            ),
        )
    return float(selected["threshold"]), rows


def _calibration_bins(
    actual: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
    bin_count: int = 10,
) -> list[dict[str, float | int]]:
    actual_values = np.asarray(actual, dtype="int8")
    probability_values = np.asarray(probabilities, dtype="float64")
    effective_bins = min(bin_count, len(np.unique(probability_values)))
    if effective_bins < 1:
        return []
    try:
        bins = pd.qcut(
            probability_values,
            q=effective_bins,
            duplicates="drop",
        )
    except ValueError:
        return []
    frame = pd.DataFrame(
        {"actual": actual_values, "probability": probability_values, "bin": bins}
    )
    grouped = (
        frame.groupby("bin", observed=True)
        .agg(
            order_count=("actual", "size"),
            mean_predicted_probability=("probability", "mean"),
            observed_late_rate=("actual", "mean"),
        )
        .reset_index(drop=True)
    )
    return [
        {
            "order_count": int(row.order_count),
            "mean_predicted_probability": float(row.mean_predicted_probability),
            "observed_late_rate": float(row.observed_late_rate),
        }
        for row in grouped.itertuples(index=False)
    ]


def _calibration_summary(
    actual: pd.Series | np.ndarray,
    probabilities: pd.Series | np.ndarray,
) -> dict[str, Any]:
    """Summarize Brier score and quantile-bin reliability without overclaiming."""
    actual_values = np.asarray(actual, dtype="int8")
    probability_values = np.asarray(probabilities, dtype="float64")
    bins = _calibration_bins(actual_values, probability_values)
    if bins:
        weighted_error = sum(
            row["order_count"]
            * abs(
                row["mean_predicted_probability"] - row["observed_late_rate"]
            )
            for row in bins
        ) / len(actual_values)
    else:
        weighted_error = None
    prevalence = float(actual_values.mean())
    return {
        "brier_score": float(brier_score_loss(actual_values, probability_values)),
        "prevalence_only_brier_score": float(
            brier_score_loss(actual_values, np.full(len(actual_values), prevalence))
        ),
        "weighted_calibration_error": (
            float(weighted_error) if weighted_error is not None else None
        ),
        "bins": bins,
        "interpretation": (
            "Reliability is estimated on a held-out training validation set. "
            "It is not a guarantee of calibration in production."
        ),
    }


def _class_counts(target: pd.Series | np.ndarray) -> dict[str, int | float]:
    labels = np.asarray(target, dtype="int8")
    total = int(len(labels))
    late = int(labels.sum())
    return {
        "orders": total,
        "on_time": total - late,
        "late": late,
        "late_rate": float(late / total) if total else 0.0,
    }


def _decision_cost_configuration(
    false_negative_cost: float | None,
    false_positive_cost: float | None,
) -> dict[str, Any]:
    if false_negative_cost is None and false_positive_cost is None:
        return {
            "false_negative_cost": None,
            "false_positive_cost": None,
            "status": "No monetary/operational cost values were provided.",
            "threshold_policy": (
                f"Maximize F{CLASSIFICATION_THRESHOLD_BETA:g} on training-only validation; "
                "this prioritizes recall without inventing costs."
            ),
        }
    if false_negative_cost is None or false_positive_cost is None:
        raise ValueError("Both decision costs must be set together.")
    if (
        not math.isfinite(false_negative_cost)
        or not math.isfinite(false_positive_cost)
        or false_negative_cost < 0
        or false_positive_cost < 0
    ):
        raise ValueError("Decision costs must be finite and non-negative.")
    return {
        "false_negative_cost": float(false_negative_cost),
        "false_positive_cost": float(false_positive_cost),
        "status": "Caller-supplied costs; not empirical project costs.",
        "threshold_policy": "Minimize validation mean decision cost.",
    }


def train_classification_models(
    dataset: ClassificationDataset,
    *,
    model_path: Path = CLASSIFICATION_MODEL_PATH,
    report_path: Path = CLASSIFICATION_REPORT_PATH,
    test_predictions_path: Path = CLASSIFICATION_TEST_PREDICTIONS_PATH,
    validation_predictions_path: Path = CLASSIFICATION_VALIDATION_PREDICTIONS_PATH,
    importance_path: Path = CLASSIFICATION_IMPORTANCE_PATH,
    source_path: Path = PROCESSED_DATA_PATH,
    test_fraction: float = DEFAULT_TEST_FRACTION,
    validation_fraction: float = DEFAULT_VALIDATION_FRACTION,
    cv_folds: int = DEFAULT_CV_FOLDS,
    search_iterations: int = DEFAULT_SEARCH_ITERATIONS,
    permutation_repeats: int = 3,
    false_negative_cost: float | None = CLASSIFICATION_FALSE_NEGATIVE_COST,
    false_positive_cost: float | None = CLASSIFICATION_FALSE_POSITIVE_COST,
) -> dict[str, Any]:
    """Tune on development folds, set threshold on validation, evaluate test once."""
    if len(dataset.target) < 20:
        raise ValueError("At least 20 labeled orders are required for classification.")
    if (
        tuple(dataset.features.columns) != REGRESSION_FEATURE_COLUMNS
        or len(dataset.features) != len(dataset.target)
        or len(dataset.context) != len(dataset.target)
        or len(dataset.source_row_numbers) != len(dataset.target)
    ):
        raise ValueError("Classification features, labels, and row metadata are inconsistent.")
    if cv_folds < 2 or search_iterations < 1 or permutation_repeats < 1:
        raise ValueError("CV folds, search iterations, and repeats must be positive.")
    costs = _decision_cost_configuration(false_negative_cost, false_positive_cost)

    train_positions, test_positions = stratified_train_test_indices(
        dataset.target, test_fraction
    )
    train_local, validation_local = stratified_development_validation_indices(
        dataset.target.iloc[train_positions],
        validation_fraction,
        CLASSIFICATION_RANDOM_SEED + 1,
    )
    development_positions = train_positions[train_local]
    validation_positions = train_positions[validation_local]
    X_development = dataset.features.iloc[development_positions]
    y_development = dataset.target.iloc[development_positions]
    X_validation = dataset.features.iloc[validation_positions]
    y_validation = dataset.target.iloc[validation_positions]
    X_training = dataset.features.iloc[train_positions]
    y_training = dataset.target.iloc[train_positions]
    X_test = dataset.features.iloc[test_positions]
    y_test = dataset.target.iloc[test_positions]

    positive_weight = float((y_development == 0).sum() / max((y_development == 1).sum(), 1))
    folds = min(cv_folds, int(y_development.value_counts().min()))
    if folds < 2:
        raise ValueError("Each class needs at least two development rows for CV.")
    cross_validation = StratifiedKFold(
        n_splits=folds,
        shuffle=True,
        random_state=CLASSIFICATION_RANDOM_SEED,
    )

    candidates = _candidate_models(positive_weight)
    fitted_development: dict[str, Any] = {}
    model_results: list[dict[str, Any]] = []
    selected_validation_probabilities: dict[str, np.ndarray] = {}
    for name, (estimator, parameter_space) in candidates.items():
        search = RandomizedSearchCV(
            estimator=build_classification_pipeline(name, estimator),
            param_distributions=parameter_space,
            n_iter=min(search_iterations, max(1, math.prod(map(len, parameter_space.values())))),
            scoring="average_precision",
            cv=cross_validation,
            refit=True,
            random_state=CLASSIFICATION_RANDOM_SEED,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(X_development, y_development)
        best_pipeline: Pipeline = search.best_estimator_
        raw_validation_probabilities = _positive_probability(best_pipeline, X_validation)
        raw_brier = float(brier_score_loss(y_validation, raw_validation_probabilities))

        calibrated_pipeline: CalibratedClassifierCV | None = None
        calibrated_validation_probabilities: np.ndarray | None = None
        if len(y_development.value_counts()) == 2:
            calibration_cv = StratifiedKFold(
                n_splits=min(CALIBRATION_CV_FOLDS, int(y_development.value_counts().min())),
                shuffle=True,
                random_state=CLASSIFICATION_RANDOM_SEED + 2,
            )
            calibrated_pipeline = CalibratedClassifierCV(
                estimator=best_pipeline,
                method="sigmoid",
                cv=calibration_cv,
            )
            calibrated_pipeline.fit(X_development, y_development)
            calibrated_validation_probabilities = _positive_probability(
                calibrated_pipeline, X_validation
            )
        calibrated_brier = (
            float(brier_score_loss(y_validation, calibrated_validation_probabilities))
            if calibrated_validation_probabilities is not None
            else math.inf
        )
        use_calibration = calibrated_brier < raw_brier
        validation_estimator: Any = (
            calibrated_pipeline if use_calibration else best_pipeline
        )
        validation_probabilities = (
            calibrated_validation_probabilities
            if use_calibration
            else raw_validation_probabilities
        )
        assert validation_probabilities is not None

        threshold, threshold_rows = select_operating_threshold(
            y_validation,
            validation_probabilities,
            beta=CLASSIFICATION_THRESHOLD_BETA,
            false_negative_cost=false_negative_cost,
            false_positive_cost=false_positive_cost,
        )
        validation_metrics = _classification_metrics(
            y_validation, validation_probabilities, threshold
        )
        result = {
            "model": name,
            "cv_average_precision": float(search.best_score_),
            "best_parameters": search.best_params_,
            "validation_metrics": validation_metrics,
            "selected_threshold": threshold,
            "validation_threshold_comparison": threshold_rows,
            "probability_calibration": {
                "method": "sigmoid" if use_calibration else "none",
                "raw_brier_score": raw_brier,
                "calibrated_brier_score": (
                    calibrated_brier if calibrated_validation_probabilities is not None else None
                ),
                "selected_brier_score": (
                    calibrated_brier if use_calibration else raw_brier
                ),
            },
        }
        model_results.append(result)
        selected_validation_probabilities[name] = validation_probabilities
        fitted_development[name] = validation_estimator

    selected_result = max(
        model_results,
        key=lambda result: (
            result["validation_metrics"]["f_beta"],
            result["validation_metrics"]["recall"],
            result["validation_metrics"]["precision"],
            result["validation_metrics"]["pr_auc"],
        ),
    )
    selected_name = str(selected_result["model"])
    selected_threshold = float(selected_result["selected_threshold"])
    selected_validation_model = fitted_development[selected_name]
    validation_probabilities = selected_validation_probabilities[selected_name]
    selected_validation_metrics = _classification_metrics(
        y_validation, validation_probabilities, selected_threshold
    )

    estimator, _ = candidates[selected_name]
    best_parameters = selected_result["best_parameters"]
    final_base = build_classification_pipeline(selected_name, clone(estimator))
    final_base.set_params(**best_parameters)
    use_calibration = (
        selected_result["probability_calibration"]["method"] == "sigmoid"
    )
    if use_calibration:
        final_cv = StratifiedKFold(
            n_splits=min(CALIBRATION_CV_FOLDS, int(y_training.value_counts().min())),
            shuffle=True,
            random_state=CLASSIFICATION_RANDOM_SEED + 2,
        )
        final_model: Any = CalibratedClassifierCV(
            estimator=final_base,
            method="sigmoid",
            cv=final_cv,
        )
        final_model.fit(X_training, y_training)
    else:
        final_model = final_base.fit(X_training, y_training)

    test_probabilities = _positive_probability(final_model, X_test)
    test_metrics = _classification_metrics(y_test, test_probabilities, selected_threshold)
    test_calibration = _calibration_summary(y_test, test_probabilities)
    validation_calibration = _calibration_summary(
        y_validation, validation_probabilities
    )
    importance = permutation_importance(
        selected_validation_model,
        X_validation,
        y_validation,
        scoring="average_precision",
        n_repeats=permutation_repeats,
        random_state=CLASSIFICATION_RANDOM_SEED,
        n_jobs=1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": REGRESSION_FEATURE_COLUMNS,
            "pr_auc_decrease": importance.importances_mean,
            "standard_deviation": importance.importances_std,
        }
    ).sort_values("pr_auc_decrease", ascending=False, ignore_index=True)

    test_predictions = _prediction_frame(
        dataset, test_positions, y_test, test_probabilities, selected_threshold
    )
    validation_predictions = _prediction_frame(
        dataset,
        validation_positions,
        y_validation,
        validation_probabilities,
        selected_threshold,
    )
    metadata = {
        "model_version": MODEL_VERSION,
        "selected_model": selected_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_column": CLASSIFICATION_TARGET_COLUMN,
        "target_definition": (
            f"1 when {TARGET_COLUMN} > configured SLA "
            f"({SLA_THRESHOLD_MINUTES} minutes); otherwise 0."
        ),
        "sla_threshold_minutes": SLA_THRESHOLD_MINUTES,
        "sla_status": "provisional",
        "feature_columns": list(REGRESSION_FEATURE_COLUMNS),
        "feature_availability_policy": {
            feature: reason
            for feature, reason in REGRESSION_FEATURE_POLICY.items()
        },
        "excluded_prediction_fields": {
            "Time_taken(min)": "Outcome used only to construct the label.",
            "ID": "Order identifier; not a predictor.",
            "Delivery_person_ID": "Rider assignment timing is unknown.",
            "Time_Order_picked": "Known after order placement.",
            "rider_and_vehicle_fields": "Assignment timing is unverified.",
            "weather_normalized/Road_traffic_density": (
                "Availability at order placement is unverified; included only as "
                "retrospective dashboard context, never as model features."
            ),
        },
        "prediction_scenario": (
            "Order placement. Rider-assignment and post-pickup models are not combined."
        ),
        "training_configuration": {
            "random_seed": CLASSIFICATION_RANDOM_SEED,
            "test_fraction": test_fraction,
            "validation_fraction_of_training": validation_fraction,
            "cv_folds": folds,
            "cv_scoring": "average precision",
            "search_iterations_per_model": search_iterations,
            "test_rows": int(len(test_positions)),
            "training_rows": int(len(train_positions)),
            "development_rows": int(len(development_positions)),
            "validation_rows": int(len(validation_positions)),
            "input_rows": int(len(dataset.target)),
            "class_distribution": {
                "all_data": _class_counts(dataset.target),
                "training": _class_counts(y_training),
                "development": _class_counts(y_development),
                "validation": _class_counts(y_validation),
                "test": _class_counts(y_test),
            },
            "test_set_used_for_tuning_or_threshold_selection": False,
            "threshold_selection": (
                f"Maximize F{CLASSIFICATION_THRESHOLD_BETA:g} on the stratified "
                "training validation split, with precision/intervention-volume tie breaks."
                if false_negative_cost is None
                else "Minimize caller-supplied mean validation error cost."
            ),
            "resampling": (
                "None. Class-weight options are compared during model tuning; "
                "synthetic resampling is not used."
            ),
            "preprocessing_fit_scope": (
                "Imputation, encoding, scaling, and calibration are fit within "
                "training folds or the training partition only."
            ),
            "decision_costs": costs,
        },
        "processed_dataset": {
            "path": str(source_path),
            "sha256": sha256_file(source_path) if source_path.is_file() else None,
        },
        "package_versions": _package_versions(),
    }
    report = {
        "metadata": metadata,
        "class_distribution": {
            "on_time": int((dataset.target == 0).sum()),
            "late": int(dataset.target.sum()),
            "total": int(len(dataset.target)),
            "late_rate": float(dataset.target.mean()),
        },
        "model_comparison": [
            {
                "model": row["model"],
                "cv_average_precision": row["cv_average_precision"],
                "best_parameters": row["best_parameters"],
                "validation_threshold": row["selected_threshold"],
                "validation_metrics": _metrics_without_report(row["validation_metrics"]),
                "probability_calibration": row["probability_calibration"],
            }
            for row in model_results
        ],
        "selected_model": {
            "name": selected_name,
            "selection_criterion": (
                f"Highest F{CLASSIFICATION_THRESHOLD_BETA:g} on the stratified validation split; "
                "ties favor recall, precision, and PR-AUC."
            ),
            "threshold": selected_threshold,
            "threshold_beta": CLASSIFICATION_THRESHOLD_BETA,
            "validation_metrics": _metrics_without_report(selected_validation_metrics),
            "validation_threshold_comparison": selected_result[
                "validation_threshold_comparison"
            ],
            "test_metrics": _metrics_without_report(test_metrics),
            "test_classification_report": test_metrics["classification_report"],
            "validation_calibration": validation_calibration,
            "test_calibration": test_calibration,
            "validation_curves": _curve_records(y_validation, validation_probabilities),
            "test_curves": _curve_records(y_test, test_probabilities),
        },
        "feature_importance": _records(importance_frame),
        "artifacts": {
            "model": str(model_path),
            "test_predictions": str(test_predictions_path),
            "validation_predictions": str(validation_predictions_path),
            "feature_importance": str(importance_path),
        },
        "interpretation": (
            "False negatives are genuinely late orders not flagged; false positives are "
            "on-time orders unnecessarily flagged. Feature importance and category "
            "breakdowns describe associations, not causes. Weather and traffic fields "
            "are retrospective context only, not classifier inputs."
        ),
    }
    _save_json(report, report_path)
    test_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    test_predictions.to_csv(test_predictions_path, index=False)
    validation_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    validation_predictions.to_csv(validation_predictions_path, index=False)
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    importance_frame.to_csv(importance_path, index=False)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {
            "pipeline": final_model,
            "metadata": {
                **metadata,
                "operating_threshold": selected_threshold,
                "calibration_method": "sigmoid" if use_calibration else None,
            },
        },
        model_path,
    )
    return report


def _prediction_frame(
    dataset: ClassificationDataset,
    positions: np.ndarray,
    actual: pd.Series,
    probabilities: np.ndarray,
    threshold: float,
) -> pd.DataFrame:
    frame = dataset.features.iloc[positions].reset_index(drop=True).copy()
    frame.insert(
        0,
        "source_row_number",
        dataset.source_row_numbers.iloc[positions].to_numpy(),
    )
    frame["actual_is_late"] = np.asarray(actual, dtype="int8")
    frame["risk_score"] = probabilities
    frame["operating_threshold"] = threshold
    frame["predicted_is_late"] = (probabilities >= threshold).astype("int8")
    frame["weather_context"] = dataset.context.iloc[positions]["weather"].to_numpy()
    frame["traffic_context"] = dataset.context.iloc[positions]["traffic"].to_numpy()
    return frame


def _metrics_without_report(metrics: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in metrics.items() if key != "classification_report"}


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return [
        {
            key: value.item() if isinstance(value, np.generic) else value
            for key, value in row.items()
        }
        for row in frame.to_dict("records")
    ]


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, (np.ndarray,)):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NA or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return value


def _save_json(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def load_classifier_bundle(path: Path = CLASSIFICATION_MODEL_PATH) -> dict[str, Any]:
    """Load a persisted classifier and reject artifacts with a mismatched policy."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Classifier not found at {path}. Run `python -m src.models.classification` first."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not {"pipeline", "metadata"}.issubset(bundle):
        raise ValueError(f"Classifier artifact at {path} has an unsupported structure.")
    metadata = bundle["metadata"]
    if tuple(metadata.get("feature_columns", ())) != REGRESSION_FEATURE_COLUMNS:
        raise ValueError("Classifier feature configuration does not match the project.")
    if metadata.get("sla_threshold_minutes") != SLA_THRESHOLD_MINUTES:
        raise ValueError("Classifier SLA does not match the configured project SLA.")
    return bundle


def predict_late_order(
    order: Mapping[str, Any],
    model_path: Path = CLASSIFICATION_MODEL_PATH,
) -> dict[str, Any]:
    """Return a calibrated risk score and late/on-time status for an order."""
    bundle = load_classifier_bundle(model_path)
    features, warnings = build_order_features(order)
    risk_score = float(_positive_probability(bundle["pipeline"], features)[0])
    threshold = float(bundle["metadata"]["operating_threshold"])
    return {
        "risk_score": risk_score,
        "classification_threshold": threshold,
        "predicted_status": "late" if risk_score >= threshold else "on_time",
        "model": bundle["metadata"]["selected_model"],
        "model_version": bundle["metadata"]["model_version"],
        "sla_threshold_minutes": bundle["metadata"]["sla_threshold_minutes"],
        "warnings": warnings,
    }


def _package_versions() -> dict[str, str | None]:
    return {
        package: _version(package)
        for package in ("scikit-learn", "xgboost", "joblib", "pandas", "numpy")
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the PS5 late-order risk classifiers."
    )
    parser.add_argument("--input", type=Path, default=PROCESSED_DATA_PATH)
    args = parser.parse_args()
    dataset = load_classification_dataset(args.input)
    report = train_classification_models(dataset, source_path=args.input)
    selected = report["selected_model"]
    print(
        f"Late orders: {report['class_distribution']['late']:,}/"
        f"{report['class_distribution']['total']:,} "
        f"({report['class_distribution']['late_rate']:.1%})"
    )
    print(
        f"Selected model: {report['selected_model']['name']} "
        f"at threshold {selected['threshold']:.2f}"
    )
    print(
        f"Test recall: {selected['test_metrics']['recall']:.3f}; "
        f"precision: {selected['test_metrics']['precision']:.3f}; "
        f"F1: {selected['test_metrics']['f1']:.3f}; "
        f"ROC-AUC: {selected['test_metrics']['roc_auc']:.3f}"
    )
    print(f"Model: {CLASSIFICATION_MODEL_PATH}")
    print(f"Metrics: {CLASSIFICATION_REPORT_PATH}")


if __name__ == "__main__":
    main()
