"""Tests for PS5 late-order target, validation, thresholding, and inference."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from config import (
    CLASSIFICATION_TARGET_COLUMN,
    REGRESSION_FEATURE_COLUMNS,
    SLA_THRESHOLD_MINUTES,
)
from src.data.feature_engineering import label_late_orders
from src.models.classification import (
    ClassificationDataset,
    _calibration_summary,
    evaluate_classification_metrics,
    expected_decision_cost,
    load_classifier_bundle,
    predict_late_order,
    select_operating_threshold,
    stratified_development_validation_indices,
    stratified_train_test_indices,
    train_classification_models,
)


def small_classification_dataset(rows: int = 120) -> ClassificationDataset:
    """Create deterministic orders with both classes and realistic feature types."""
    row = np.arange(rows)
    distance = 1.0 + row % 12
    hour = row % 24
    weekday = row % 7
    features = pd.DataFrame(
        {
            "delivery_distance_km": distance.astype(float),
            "order_hour": hour.astype(float),
            "order_day": row % 28 + 1,
            "day_of_week": weekday,
            "is_weekend": (weekday >= 5).astype(float),
            "is_peak_hour": np.isin(hour, [11, 12, 13, 17, 18, 19, 20, 21]).astype(float),
            "distance_band": pd.cut(
                distance,
                bins=[0, 3, 7, 12, np.inf],
                labels=["0-3 km", "3-7 km", "7-12 km", "12+ km"],
                right=False,
            ).astype(str),
            "City": np.where(row % 3, "Urban", "Rural"),
            "Type_of_order": np.where(row % 2, "Meal", "Snack"),
        }
    ).loc[:, list(REGRESSION_FEATURE_COLUMNS)]
    labels = ((distance > 7) | np.isin(hour, [11, 12, 13, 17, 18, 19, 20, 21])).astype(
        "int8"
    )
    return ClassificationDataset(
        features=features,
        target=pd.Series(labels),
        source_row_numbers=pd.Series(row + 2, dtype="int64"),
        context=pd.DataFrame(
            {
                "weather": np.where(row % 2, "Sunny", "Rain"),
                "traffic": np.where(row % 3, "High", "Low"),
            }
        ),
    )


class ClassificationTests(unittest.TestCase):
    def test_sla_target_is_strictly_greater_than_configured_threshold(self) -> None:
        labels = label_late_orders(pd.Series([SLA_THRESHOLD_MINUTES - 1, SLA_THRESHOLD_MINUTES, SLA_THRESHOLD_MINUTES + 1, pd.NA]))
        self.assertEqual(labels.iloc[:3].astype(int).tolist(), [0, 0, 1])
        self.assertTrue(pd.isna(labels.iloc[3]))
        self.assertEqual(CLASSIFICATION_TARGET_COLUMN, "is_late")

    def test_class_distribution_and_stratified_splits(self) -> None:
        dataset = small_classification_dataset()
        train, test = stratified_train_test_indices(dataset.target)
        dev, validation = stratified_development_validation_indices(dataset.target.iloc[train])
        self.assertFalse(set(train).intersection(test))
        self.assertFalse(set(dev).intersection(validation))
        self.assertAlmostEqual(
            dataset.target.iloc[train].mean(), dataset.target.iloc[test].mean(), delta=0.02
        )
        self.assertEqual(int(dataset.target.sum()) + int((dataset.target == 0).sum()), len(dataset.target))

    def test_leakage_fields_are_not_model_predictors(self) -> None:
        dataset = small_classification_dataset()
        self.assertEqual(tuple(dataset.features.columns), REGRESSION_FEATURE_COLUMNS)
        self.assertTrue(
            {
                "ID",
                "Delivery_person_ID",
                "Time_Order_picked",
                "Time_taken(min)",
                "is_late",
                "weather",
                "traffic",
            }.isdisjoint(dataset.features.columns)
        )

    def test_threshold_selection_reports_false_positives_and_negatives(self) -> None:
        actual = np.array([0, 0, 0, 1, 1, 1])
        probabilities = np.array([0.1, 0.35, 0.8, 0.3, 0.6, 0.9])
        threshold, rows = select_operating_threshold(actual, probabilities)
        self.assertIn(threshold, [row["threshold"] for row in rows])
        metrics = evaluate_classification_metrics(actual, probabilities, threshold)
        self.assertEqual(
            metrics["true_positives"] + metrics["false_negatives"],
            int(actual.sum()),
        )
        self.assertEqual(
            metrics["true_negatives"] + metrics["false_positives"],
            int((actual == 0).sum()),
        )
        self.assertTrue(all("flagged_orders" in row for row in rows))

    def test_decision_costs_require_explicit_nonnegative_inputs(self) -> None:
        costs = expected_decision_cost(
            [0, 0, 1, 1],
            [1, 0, 0, 1],
            false_negative_cost=10,
            false_positive_cost=2,
        )
        self.assertEqual(costs["total_cost"], 12)
        with self.assertRaises(ValueError):
            expected_decision_cost(
                [0, 1], [1, 0], false_negative_cost=-1, false_positive_cost=2
            )

    def test_calibration_summary_compares_prevalence_reference(self) -> None:
        summary = _calibration_summary(
            [0, 0, 1, 1],
            [0.1, 0.4, 0.6, 0.9],
        )
        self.assertGreaterEqual(summary["brier_score"], 0)
        self.assertGreaterEqual(summary["prevalence_only_brier_score"], 0)
        self.assertTrue(summary["bins"])

    def test_training_persistence_metrics_and_prediction(self) -> None:
        dataset = small_classification_dataset()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "processed.csv"
            source.write_text("test data source", encoding="utf-8")
            model_path = root / "models" / "classifier.joblib"
            report_path = root / "metrics" / "report.json"
            test_path = root / "metrics" / "test.csv"
            validation_path = root / "metrics" / "validation.csv"
            importance_path = root / "metrics" / "importance.csv"
            report = train_classification_models(
                dataset,
                model_path=model_path,
                report_path=report_path,
                test_predictions_path=test_path,
                validation_predictions_path=validation_path,
                importance_path=importance_path,
                source_path=source,
                cv_folds=2,
                search_iterations=1,
                permutation_repeats=1,
            )
            self.assertTrue(model_path.is_file())
            self.assertTrue(report_path.is_file())
            self.assertTrue(test_path.is_file())
            self.assertTrue(validation_path.is_file())
            self.assertTrue(importance_path.is_file())
            self.assertGreaterEqual(len(report["model_comparison"]), 3)
            self.assertFalse(
                report["metadata"]["training_configuration"][
                    "test_set_used_for_tuning_or_threshold_selection"
                ]
            )
            self.assertEqual(
                report["metadata"]["training_configuration"]["resampling"],
                "None. Class-weight options are compared during model tuning; synthetic resampling is not used.",
            )
            saved = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(len(pd.read_csv(test_path)), 24)
            self.assertEqual(saved["selected_model"]["threshold"], report["selected_model"]["threshold"])
            bundle = load_classifier_bundle(model_path)
            self.assertEqual(
                bundle["metadata"]["operating_threshold"],
                report["selected_model"]["threshold"],
            )
            prediction = predict_late_order(
                {
                    "Order_Date": "01-03-2022",
                    "Time_Orderd": "11:30:00",
                    "Restaurant_latitude": 12.0,
                    "Restaurant_longitude": 77.0,
                    "Delivery_location_latitude": 12.01,
                    "Delivery_location_longitude": 77.01,
                    "City": "Urban",
                    "Type_of_order": "Snack",
                },
                model_path=model_path,
            )
            self.assertGreaterEqual(prediction["risk_score"], 0)
            self.assertLessEqual(prediction["risk_score"], 1)
            self.assertEqual(
                prediction["predicted_status"],
                "late"
                if prediction["risk_score"] >= prediction["classification_threshold"]
                else "on_time",
            )
            self.assertEqual(prediction["model_version"], "1.0.0")
            with self.assertRaises(ValueError):
                predict_late_order(
                    {"Time_taken(min)": 22},
                    model_path=model_path,
                )


if __name__ == "__main__":
    unittest.main()
