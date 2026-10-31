"""Leakage-aware training, evaluation, persistence, and inference for ETA regression."""

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
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import KFold, RandomizedSearchCV, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from config import (
    COORDINATE_COLUMNS,
    PROCESSED_DATA_PATH,
    REGRESSION_CATEGORICAL_FEATURE_COLUMNS,
    REGRESSION_FEATURE_COLUMNS,
    REGRESSION_FEATURE_POLICY,
    REGRESSION_IMPORTANCE_PATH,
    REGRESSION_MODEL_PATH,
    REGRESSION_NUMERIC_FEATURE_COLUMNS,
    REGRESSION_PREDICTIONS_PATH,
    REGRESSION_RANDOM_SEED,
    REGRESSION_REPORT_PATH,
    SLA_THRESHOLD_MINUTES,
    TARGET_COLUMN,
)
from src.data.cleaning import extract_delivery_time_minutes
from src.data.feature_engineering import engineer_features, get_pre_delivery_features, parse_timestamps
from src.data.loader import load_orders, sha256_file

MODEL_VERSION = "1.0.0"
MISSING_CATEGORY = "(Missing)"
DEFAULT_TEST_SIZE = 0.2
DEFAULT_CV_SPLITS = 3
DEFAULT_SEARCH_ITERATIONS = 3
MIN_TEMPORAL_DATES = 4
TEMPORAL_VALIDATION_FRACTION = 0.2
RIDER_AND_OUTCOME_FIELDS = {
    "ID",
    "Delivery_person_ID",
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Vehicle_condition",
    "Type_of_vehicle",
    "multiple_deliveries",
    "Time_Order_picked",
    TARGET_COLUMN,
    "is_late_order",
}
PREDICTION_INPUT_COLUMNS = frozenset(
    {
        "Order_Date",
        "Time_Orderd",
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
        "City",
        "Type_of_order",
    }
)
_ENGINEERING_CONTEXT_COLUMNS = (
    "Order_Date",
    "Time_Orderd",
    "Time_Order_picked",
    "Weatherconditions",
    "Road_traffic_density",
    "Delivery_person_Age",
    "Festival",
    "City",
    "Type_of_order",
    *COORDINATE_COLUMNS.values(),
)
_MISSING_TEXT = frozenset({"", "nan", "na", "n/a", "null", "none"})

try:
    from xgboost import XGBRegressor
except ImportError:
    XGBRegressor = None  # type: ignore[assignment,misc]


@dataclass(frozen=True)
class RegressionDataset:
    """Prepared predictors, target, dates, and source row counts."""

    features: pd.DataFrame
    target: pd.Series
    order_dates: pd.Series
    source_row_numbers: pd.Series
    excluded_target_rows: int


def _normalize_features(features: pd.DataFrame) -> pd.DataFrame:
    """Convert Phase 1 string-typed columns into numeric and nullable categories."""
    normalized = features.loc[:, list(REGRESSION_FEATURE_COLUMNS)].copy()
    for column in REGRESSION_NUMERIC_FEATURE_COLUMNS:
        values = normalized[column].astype("string").str.strip()
        if column in {"is_weekend", "is_peak_hour"}:
            numeric = values.str.casefold().map(
                {"true": 1.0, "false": 0.0, "1": 1.0, "0": 0.0}
            )
        else:
            numeric = pd.to_numeric(values, errors="coerce")
        normalized[column] = pd.to_numeric(numeric, errors="coerce").astype("float64")

    for column in REGRESSION_CATEGORICAL_FEATURE_COLUMNS:
        values = normalized[column].astype("string").str.strip()
        is_missing = (
            values.isna()
            | values.str.casefold().isin(_MISSING_TEXT)
        )
        object_values = values.to_numpy(dtype=object, na_value=np.nan)
        object_values[is_missing.to_numpy()] = np.nan
        normalized[column] = object_values
    return normalized


def load_regression_dataset(path: Path = PROCESSED_DATA_PATH) -> RegressionDataset:
    """Load Phase 1 features and the configured target without dropping predictors."""
    frame = load_orders(path)
    if TARGET_COLUMN not in frame:
        raise ValueError(f"Processed dataset must contain the target `{TARGET_COLUMN}`.")

    raw_target = frame[TARGET_COLUMN].astype("string").str.strip()
    target = pd.to_numeric(raw_target, errors="coerce")
    formatted = raw_target.str.contains(r"\(min\)", case=False, na=False)
    if formatted.any():
        target.loc[formatted] = extract_delivery_time_minutes(raw_target.loc[formatted])
    target = target.where(target.gt(0))
    valid_target = target.notna() & np.isfinite(target.astype("float64"))
    selected_features = get_pre_delivery_features(frame).loc[
        :, list(REGRESSION_FEATURE_COLUMNS)
    ]
    features = _normalize_features(selected_features.loc[valid_target])
    order_dates = parse_timestamps(frame.loc[valid_target])["Order_Date"].reset_index(
        drop=True
    )
    return RegressionDataset(
        features=features.reset_index(drop=True),
        target=target.loc[valid_target].astype("float64").reset_index(drop=True),
        order_dates=order_dates,
        source_row_numbers=pd.Series(
            np.flatnonzero(valid_target.to_numpy()) + 2,
            name="source_row_number",
            dtype="int64",
        ),
        excluded_target_rows=int((~valid_target).sum()),
    )


def create_preprocessor(
    model_name: str, *, scale_numeric: bool | None = None
) -> ColumnTransformer:
    """Build train-fitted imputers/encoder, optionally scaling numeric features."""
    numeric_steps: list[tuple[str, Any]] = [
        ("imputer", SimpleImputer(strategy="median", add_indicator=True))
    ]
    if scale_numeric is None:
        scale_numeric = model_name == "Linear Regression"
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))
    numeric = Pipeline(numeric_steps)
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="constant", fill_value=MISSING_CATEGORY)),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    return ColumnTransformer(
        [
            ("numeric", numeric, list(REGRESSION_NUMERIC_FEATURE_COLUMNS)),
            ("categorical", categorical, list(REGRESSION_CATEGORICAL_FEATURE_COLUMNS)),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def _candidate_models() -> dict[str, tuple[Any, dict[str, list[Any]]]]:
    """Return small, deterministic search spaces for the requested regressors."""
    candidates: dict[str, tuple[Any, dict[str, list[Any]]]] = {
        "Linear Regression": (
            LinearRegression(),
            {"model__fit_intercept": [True, False]},
        ),
        "Random Forest": (
            RandomForestRegressor(
                random_state=REGRESSION_RANDOM_SEED,
                n_jobs=4,
            ),
            {
                "model__n_estimators": [100, 160],
                "model__max_depth": [None, 16],
                "model__min_samples_leaf": [1, 3],
                "model__max_features": [0.8, 1.0],
            },
        ),
        "Gradient Boosting": (
            GradientBoostingRegressor(random_state=REGRESSION_RANDOM_SEED),
            {
                "model__n_estimators": [100, 160],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [2, 3],
            },
        ),
    }
    if XGBRegressor is not None:
        candidates["XGBoost"] = (
            XGBRegressor(
                objective="reg:squarederror",
                random_state=REGRESSION_RANDOM_SEED,
                n_jobs=4,
                tree_method="hist",
                verbosity=0,
            ),
            {
                "model__n_estimators": [100, 160],
                "model__learning_rate": [0.05, 0.1],
                "model__max_depth": [3, 5],
                "model__subsample": [0.8, 1.0],
            },
        )
    return candidates


def build_regression_pipeline(model_name: str, estimator: Any) -> Pipeline:
    """Create the single preprocessing-plus-estimator pipeline used for fit and inference."""
    return Pipeline(
        [
            ("preprocessor", create_preprocessor(model_name)),
            ("model", estimator),
        ]
    )


def _make_temporal_split(
    training_dates: pd.Series,
) -> tuple[np.ndarray, np.ndarray, str | None, str | None]:
    """Hold out the latest distinct dates from the training partition only."""
    parsed_dates = pd.to_datetime(training_dates, errors="coerce")
    unique_dates = pd.Index(parsed_dates.dropna().unique()).sort_values()
    if len(unique_dates) < MIN_TEMPORAL_DATES:
        return (
            np.arange(len(training_dates)),
            np.array([], dtype="int64"),
            None,
            f"At least {MIN_TEMPORAL_DATES} distinct valid dates are required.",
        )
    validation_date_count = max(
        1, math.ceil(len(unique_dates) * TEMPORAL_VALIDATION_FRACTION)
    )
    first_validation_date = unique_dates[-validation_date_count]
    validation_mask = parsed_dates.ge(first_validation_date).fillna(False).to_numpy()
    validation_positions = np.flatnonzero(validation_mask)
    development_positions = np.flatnonzero(~validation_mask)
    if not len(development_positions) or not len(validation_positions):
        return (
            np.arange(len(training_dates)),
            np.array([], dtype="int64"),
            None,
            "Unable to form non-empty chronological development and validation sets.",
        )
    return (
        development_positions,
        validation_positions,
        pd.Timestamp(first_validation_date).date().isoformat(),
        None,
    )


def split_train_test_indices(
    row_count: int,
    test_size: float = DEFAULT_TEST_SIZE,
    random_seed: int = REGRESSION_RANDOM_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Create reproducible disjoint row positions for train and held-out test."""
    if row_count < 2:
        raise ValueError("At least two rows are required for a train-test split.")
    if not 0.0 < test_size < 1.0:
        raise ValueError("test_size must be between zero and one.")
    return train_test_split(
        np.arange(row_count),
        test_size=test_size,
        random_state=random_seed,
        shuffle=True,
    )


def calculate_regression_metrics(
    actual: pd.Series | np.ndarray,
    predicted: pd.Series | np.ndarray,
) -> dict[str, float]:
    """Calculate standard ETA errors after validating finite, aligned observations."""
    actual_values = np.asarray(actual, dtype="float64")
    predicted_values = np.asarray(predicted, dtype="float64")
    if actual_values.ndim != 1 or predicted_values.ndim != 1:
        raise ValueError("Actual and predicted values must be one-dimensional.")
    if not len(actual_values) or len(actual_values) != len(predicted_values):
        raise ValueError("Actual and predicted values must have matching non-empty lengths.")
    if not np.isfinite(actual_values).all() or not np.isfinite(predicted_values).all():
        raise ValueError("Actual and predicted values must all be finite.")
    return {
        "mae_minutes": float(mean_absolute_error(actual_values, predicted_values)),
        "rmse_minutes": float(np.sqrt(mean_squared_error(actual_values, predicted_values))),
        "r2": float(r2_score(actual_values, predicted_values)),
        "median_absolute_error_minutes": float(
            median_absolute_error(actual_values, predicted_values)
        ),
    }


def _bounded_predictions(model: Pipeline, features: pd.DataFrame) -> np.ndarray:
    """Prevent negative ETA outputs while preserving the regression model's minutes unit."""
    return np.maximum(np.asarray(model.predict(features), dtype="float64"), 0.0)


def _error_by_delivery_range(
    actual: pd.Series,
    predicted: np.ndarray,
) -> list[dict[str, Any]]:
    """Summarize absolute and squared errors across actual delivery-time ranges."""
    frame = pd.DataFrame(
        {
            "actual": np.asarray(actual, dtype="float64"),
            "predicted": predicted,
        }
    )
    frame["absolute_error"] = (frame["actual"] - frame["predicted"]).abs()
    frame["squared_error"] = (frame["actual"] - frame["predicted"]) ** 2
    frame["delivery_range"] = pd.cut(
        frame["actual"],
        bins=[0, 20, 30, 40, np.inf],
        labels=["10–19", "20–29", "30–39", "40+"],
        right=False,
    )
    grouped = (
        frame.groupby("delivery_range", observed=True)
        .agg(
            order_count=("actual", "size"),
            mae_minutes=("absolute_error", "mean"),
            mean_squared_error=("squared_error", "mean"),
        )
        .reset_index()
    )
    grouped["rmse_minutes"] = np.sqrt(grouped.pop("mean_squared_error"))
    return _json_records(grouped)


def _residual_summary(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    residuals = np.asarray(actual, dtype="float64") - predicted
    return {
        "mean_minutes": float(np.mean(residuals)),
        "median_minutes": float(np.median(residuals)),
        "standard_deviation_minutes": float(np.std(residuals, ddof=1))
        if len(residuals) > 1
        else 0.0,
        "p95_absolute_minutes": float(np.quantile(np.abs(residuals), 0.95)),
    }


def _json_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        records.append(
            {
                key: (
                    value.item()
                    if isinstance(value, np.generic)
                    else value.isoformat()
                    if isinstance(value, (pd.Timestamp, datetime))
                    else value
                )
                for key, value in row.items()
            }
        )
    return records


def _json_value(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, (pd.Timestamp, datetime)):
        return value.isoformat()
    if value is pd.NA or (isinstance(value, float) and not math.isfinite(value)):
        return None
    return value


def _version(package: str) -> str | None:
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _save_json(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_value(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def train_regression_models(
    dataset: RegressionDataset,
    *,
    model_path: Path = REGRESSION_MODEL_PATH,
    report_path: Path = REGRESSION_REPORT_PATH,
    predictions_path: Path = REGRESSION_PREDICTIONS_PATH,
    importance_path: Path = REGRESSION_IMPORTANCE_PATH,
    source_path: Path = PROCESSED_DATA_PATH,
    cv_splits: int = DEFAULT_CV_SPLITS,
    search_iterations: int = DEFAULT_SEARCH_ITERATIONS,
    permutation_repeats: int = 3,
) -> dict[str, Any]:
    """Tune on development folds, validate chronologically, and test once."""
    if len(dataset.features) < 10:
        raise ValueError("At least 10 valid target rows are required for regression.")
    if cv_splits < 2 or search_iterations < 1 or permutation_repeats < 1:
        raise ValueError("CV folds, search iterations, and permutation repeats must be positive.")
    if tuple(dataset.features.columns) != REGRESSION_FEATURE_COLUMNS:
        raise ValueError("Regression feature columns do not match the configured allowlist.")

    training_positions, test_positions = split_train_test_indices(
        len(dataset.features)
    )
    training_dates = dataset.order_dates.iloc[training_positions].reset_index(drop=True)
    (
        development_local,
        temporal_local,
        temporal_start,
        temporal_skip_reason,
    ) = _make_temporal_split(training_dates)
    development_positions = training_positions[development_local]
    temporal_positions = training_positions[temporal_local]
    X_development = dataset.features.iloc[development_positions]
    y_development = dataset.target.iloc[development_positions]
    X_training = dataset.features.iloc[training_positions]
    y_training = dataset.target.iloc[training_positions]
    X_test = dataset.features.iloc[test_positions]
    y_test = dataset.target.iloc[test_positions]

    folds = min(cv_splits, len(X_development))
    if folds < 2:
        raise ValueError("Not enough development rows for cross-validation.")
    cross_validation = KFold(
        n_splits=folds,
        shuffle=True,
        random_state=REGRESSION_RANDOM_SEED,
    )
    baseline_cv_pipeline = Pipeline(
        [
            ("preprocessor", create_preprocessor("Mean Baseline")),
            ("model", DummyRegressor(strategy="mean")),
        ]
    )
    baseline_cv_scores = cross_val_score(
        baseline_cv_pipeline,
        X_development,
        y_development,
        cv=cross_validation,
        scoring="neg_mean_absolute_error",
        n_jobs=1,
    )
    candidates = _candidate_models()
    tuned_estimators: dict[str, Pipeline] = {}
    model_results: list[dict[str, Any]] = []

    for name, (estimator, parameter_space) in candidates.items():
        pipeline = build_regression_pipeline(name, estimator)
        search = RandomizedSearchCV(
            estimator=pipeline,
            param_distributions=parameter_space,
            n_iter=min(search_iterations, max(1, _parameter_count(parameter_space))),
            scoring="neg_mean_absolute_error",
            cv=cross_validation,
            refit=True,
            random_state=REGRESSION_RANDOM_SEED,
            n_jobs=1,
            error_score="raise",
            return_train_score=False,
        )
        search.fit(X_development, y_development)
        best_index = int(search.best_index_)
        cv_mae = float(-search.best_score_)
        cv_mae_std = float(search.cv_results_["std_test_score"][best_index])
        selected_pipeline = build_regression_pipeline(
            name,
            _estimator_with_params(estimator, search.best_params_),
        )
        selected_pipeline.fit(X_training, y_training)
        test_predictions = _bounded_predictions(selected_pipeline, X_test)
        result: dict[str, Any] = {
            "model": name,
            "cv_mae_minutes": cv_mae,
            "cv_mae_standard_deviation_minutes": cv_mae_std,
            "best_parameters": search.best_params_,
            "test_metrics": calculate_regression_metrics(y_test, test_predictions),
            "temporal_validation_metrics": None,
        }
        if len(temporal_positions):
            temporal_predictions = _bounded_predictions(
                search.best_estimator_,
                dataset.features.iloc[temporal_positions],
            )
            result["temporal_validation_metrics"] = calculate_regression_metrics(
                dataset.target.iloc[temporal_positions],
                temporal_predictions,
            )
        model_results.append(result)
        tuned_estimators[name] = selected_pipeline

    selected_result = min(
        model_results,
        key=lambda result: (result["cv_mae_minutes"], result["model"]),
    )
    selected_name = str(selected_result["model"])
    selected_model = tuned_estimators[selected_name]
    transformed_feature_names = (
        selected_model.named_steps["preprocessor"].get_feature_names_out().tolist()
    )

    baseline_test_prediction = np.full(
        len(y_test),
        float(y_training.mean()),
        dtype="float64",
    )
    baseline_temporal_prediction = np.full(
        len(temporal_positions),
        float(y_development.mean()),
        dtype="float64",
    )
    baseline_metrics = calculate_regression_metrics(y_test, baseline_test_prediction)
    baseline_temporal_metrics = (
        calculate_regression_metrics(
            dataset.target.iloc[temporal_positions],
            baseline_temporal_prediction,
        )
        if len(temporal_positions)
        else None
    )
    selected_test_predictions = _bounded_predictions(selected_model, X_test)
    selected_test_metrics = calculate_regression_metrics(y_test, selected_test_predictions)
    test_mae_improvement = (
        baseline_metrics["mae_minutes"] - selected_test_metrics["mae_minutes"]
    )
    improvement_percent = (
        100.0 * test_mae_improvement / baseline_metrics["mae_minutes"]
        if baseline_metrics["mae_minutes"]
        else None
    )

    importance = permutation_importance(
        selected_model,
        X_test,
        y_test,
        scoring="neg_mean_absolute_error",
        n_repeats=permutation_repeats,
        random_state=REGRESSION_RANDOM_SEED,
        n_jobs=1,
    )
    importance_frame = pd.DataFrame(
        {
            "feature": REGRESSION_FEATURE_COLUMNS,
            "mae_increase_minutes": importance.importances_mean,
            "standard_deviation_minutes": importance.importances_std,
        }
    ).sort_values("mae_increase_minutes", ascending=False, ignore_index=True)

    comparison: list[dict[str, Any]] = [
        {
            "model": "Mean Baseline",
            "cv_mae_minutes": float(-baseline_cv_scores.mean()),
            "cv_mae_standard_deviation_minutes": float(baseline_cv_scores.std(ddof=1))
            if len(baseline_cv_scores) > 1
            else 0.0,
            "best_parameters": {},
            "test_metrics": baseline_metrics,
            "temporal_validation_metrics": baseline_temporal_metrics,
        },
        *model_results,
    ]
    test_predictions_frame = pd.DataFrame(
        {
            "source_row_number": dataset.source_row_numbers.iloc[test_positions].to_numpy(),
            "order_date": dataset.order_dates.iloc[test_positions]
            .dt.strftime("%Y-%m-%d")
            .to_numpy(),
            "actual_minutes": y_test.to_numpy(),
            "predicted_minutes": selected_test_predictions,
        }
    )
    test_predictions_frame["residual_minutes"] = (
        test_predictions_frame["actual_minutes"]
        - test_predictions_frame["predicted_minutes"]
    )
    test_predictions_frame["absolute_error_minutes"] = (
        test_predictions_frame["residual_minutes"].abs()
    )

    metadata: dict[str, Any] = {
        "model_version": MODEL_VERSION,
        "selected_model": selected_name,
        "trained_at_utc": datetime.now(timezone.utc).isoformat(),
        "target_column": TARGET_COLUMN,
        "target_unit": "minutes",
        "sla_threshold_minutes": SLA_THRESHOLD_MINUTES,
        "sla_status": "provisional",
        "feature_columns": list(REGRESSION_FEATURE_COLUMNS),
        "transformed_feature_names": transformed_feature_names,
        "feature_availability_policy": REGRESSION_FEATURE_POLICY,
        "excluded_feature_policy": {
            key: value
            for key, value in REGRESSION_FEATURE_POLICY.items()
            if key not in REGRESSION_FEATURE_COLUMNS
        },
        "training_configuration": {
            "random_seed": REGRESSION_RANDOM_SEED,
            "random_test_fraction": DEFAULT_TEST_SIZE,
            "cv_folds": folds,
            "cv_scoring": "negative mean absolute error",
            "search_iterations_per_model": search_iterations,
            "temporal_validation_fraction_of_training_dates": TEMPORAL_VALIDATION_FRACTION,
            "temporal_validation_start_date": temporal_start,
            "temporal_validation_skip_reason": temporal_skip_reason,
            "test_rows": int(len(test_positions)),
            "training_rows": int(len(training_positions)),
            "development_rows": int(len(development_positions)),
            "temporal_validation_rows": int(len(temporal_positions)),
            "input_rows": int(len(dataset.features) + dataset.excluded_target_rows),
            "valid_target_rows": int(len(dataset.target)),
            "excluded_invalid_target_rows": dataset.excluded_target_rows,
            "selection_criterion": (
                "Lowest training-only cross-validated MAE; chronological validation and "
                "held-out test metrics are reported separately."
            ),
            "test_set_used_for_tuning": False,
            "preprocessing_fit_scope": (
                "Each cross-validation fold fits preprocessing on that fold's training rows; "
                "final inference preprocessing is refit on the full random training split."
            ),
        },
        "dataset": {
            "processed_path": str(source_path),
            "processed_sha256": sha256_file(source_path) if source_path.is_file() else None,
        },
        "package_versions": {
            package: _version(package)
            for package in ("scikit-learn", "xgboost", "joblib", "pandas", "numpy")
        },
    }
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(
        {"pipeline": selected_model, "metadata": _json_value(metadata)},
        model_path,
        compress=3,
    )

    report: dict[str, Any] = {
        "metadata": metadata,
        "baseline_test_metrics": baseline_metrics,
        "selected_model_test_metrics": selected_test_metrics,
        "selected_model_mae_improvement_minutes": test_mae_improvement,
        "selected_model_mae_improvement_percent": improvement_percent,
        "selected_model_residual_summary": _residual_summary(
            y_test, selected_test_predictions
        ),
        "model_comparison": comparison,
        "error_by_delivery_range": _error_by_delivery_range(
            y_test, selected_test_predictions
        ),
        "feature_importance": _json_records(importance_frame),
        "temporal_baseline_metrics": baseline_temporal_metrics,
        "temporal_selected_model_metrics": selected_result[
            "temporal_validation_metrics"
        ],
        "artifacts": {
            "model": str(model_path),
            "test_predictions": str(predictions_path),
            "feature_importance": str(importance_path),
        },
        "interpretation": (
            "Permutation importance measures the change in held-out MAE when a feature is "
            "shuffled. It is an association with model performance, not a causal effect. "
            "The dataset does not confirm real-time feature availability."
        ),
    }
    _save_json(report, report_path)
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    test_predictions_frame.to_csv(predictions_path, index=False)
    importance_path.parent.mkdir(parents=True, exist_ok=True)
    importance_frame.to_csv(importance_path, index=False)
    return report


def _parameter_count(parameter_space: dict[str, list[Any]]) -> int:
    count = 1
    for values in parameter_space.values():
        count *= len(values)
    return count


def _estimator_with_params(estimator: Any, best_parameters: dict[str, Any]) -> Any:
    parameters = {
        key.removeprefix("model__"): value
        for key, value in best_parameters.items()
    }
    return estimator.set_params(**parameters)


def load_model_bundle(path: Path = REGRESSION_MODEL_PATH) -> dict[str, Any]:
    """Load the selected model and metadata produced by this training module."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Trained model not found at {path}. Run `python -m src.models.regression` first."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not {"pipeline", "metadata"}.issubset(bundle):
        raise ValueError(f"Model artifact at {path} has an unsupported structure.")
    if tuple(bundle["metadata"].get("feature_columns", ())) != REGRESSION_FEATURE_COLUMNS:
        raise ValueError("Model feature configuration does not match the current project.")
    return bundle


def build_order_features(order: Mapping[str, Any]) -> tuple[pd.DataFrame, list[str]]:
    """Derive safe model fields from raw order-placement inputs."""
    unsupported = set(order).difference(PREDICTION_INPUT_COLUMNS)
    if unsupported:
        blocked = unsupported.intersection(RIDER_AND_OUTCOME_FIELDS)
        fields = sorted(blocked or unsupported)
        raise ValueError(
            "Unsupported or unavailable prediction input field(s): "
            + ", ".join(fields)
        )
    if not order or not any(
        value is not None and str(value).strip().casefold() not in _MISSING_TEXT
        for value in order.values()
    ):
        raise ValueError("Provide at least one non-missing order-placement input.")

    raw_record = {column: pd.NA for column in _ENGINEERING_CONTEXT_COLUMNS}
    raw_record.update(order)
    raw = pd.DataFrame([raw_record])
    engineered = engineer_features(raw)
    selected = get_pre_delivery_features(engineered).loc[
        :, list(REGRESSION_FEATURE_COLUMNS)
    ]
    features = _normalize_features(selected)
    warnings: list[str] = []
    missing_features = features.columns[features.iloc[0].isna()].tolist()
    if missing_features:
        warnings.append(
            "Missing or invalid inputs will be imputed for: "
            + ", ".join(missing_features)
            + "."
        )
    if "delivery_distance_km" in missing_features:
        warnings.append(
            "Distance could not be calculated from valid regional coordinates."
        )
    return features, warnings


def predict_delivery_time(
    order: Mapping[str, Any],
    model_path: Path = REGRESSION_MODEL_PATH,
) -> dict[str, Any]:
    """Return a non-negative ETA in minutes using the saved training pipeline."""
    bundle = load_model_bundle(model_path)
    features, warnings = build_order_features(order)
    prediction = float(_bounded_predictions(bundle["pipeline"], features)[0])
    return {
        "predicted_delivery_time_minutes": prediction,
        "model": bundle["metadata"]["selected_model"],
        "model_version": bundle["metadata"]["model_version"],
        "warnings": warnings,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train and evaluate the PS4 delivery-time regression models."
    )
    parser.add_argument("--input", type=Path, default=PROCESSED_DATA_PATH)
    args = parser.parse_args()
    dataset = load_regression_dataset(args.input)
    report = train_regression_models(dataset, source_path=args.input)
    print(f"Training rows: {report['metadata']['training_configuration']['training_rows']:,}")
    print(f"Selected model: {report['metadata']['selected_model']}")
    print(
        f"Test MAE: {report['selected_model_test_metrics']['mae_minutes']:.2f} min; "
        f"RMSE: {report['selected_model_test_metrics']['rmse_minutes']:.2f} min; "
        f"R²: {report['selected_model_test_metrics']['r2']:.3f}"
    )
    print(f"Model: {report['artifacts']['model']}")
    print(f"Metrics: {REGRESSION_REPORT_PATH}")


if __name__ == "__main__":
    main()
