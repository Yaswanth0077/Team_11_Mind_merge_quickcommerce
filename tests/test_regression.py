"""Tests for PS4 regression splits, preprocessing, persistence, and inference."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from config import (
    REGRESSION_CATEGORICAL_FEATURE_COLUMNS,
    REGRESSION_FEATURE_COLUMNS,
    REGRESSION_NUMERIC_FEATURE_COLUMNS,
    TARGET_COLUMN,
)
from src.models.regression import (
    RegressionDataset,
    build_order_features,
    build_regression_pipeline,
    calculate_regression_metrics,
    load_model_bundle,
    predict_delivery_time,
    split_train_test_indices,
    train_regression_models,
)


def small_regression_dataset(rows: int = 100) -> RegressionDataset:
    """Build deterministic in-memory orders for model lifecycle tests."""
    row = np.arange(rows)
    distance = 1.0 + (row % 12)
    hour = row % 24
    order_day = row % 28 + 1
    weekday = row % 7
    features = pd.DataFrame(
        {
            "delivery_distance_km": distance.astype(float),
            "order_hour": hour.astype(float),
            "order_day": order_day.astype(float),
            "day_of_week": weekday.astype(float),
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
    target = pd.Series(
        13 + distance * 0.9 + np.isin(hour, [11, 12, 13, 17, 18, 19, 20, 21]) * 4
        + (row % 5) * 0.2,
        dtype="float64",
    )
    dates = pd.Series(
        pd.Timestamp("2022-03-01") + pd.to_timedelta(row % 10, unit="D")
    )
    return RegressionDataset(
        features=features,
        target=target,
        order_dates=dates,
        source_row_numbers=pd.Series(row + 2, dtype="int64"),
        excluded_target_rows=0,
    )


class RegressionTests(unittest.TestCase):
    def test_feature_allowlist_excludes_leakage_and_unverified_columns(self) -> None:
        self.assertEqual(len(REGRESSION_FEATURE_COLUMNS), 9)
        self.assertTrue(
            set(REGRESSION_FEATURE_COLUMNS).isdisjoint(
                {
                    "ID",
                    "Delivery_person_ID",
                    "Delivery_person_Age",
                    "Delivery_person_Ratings",
                    "Time_Order_picked",
                    TARGET_COLUMN,
                    "is_late_order",
                    "weather_normalized",
                    "traffic_normalized",
                    "Festival",
                }
            )
        )
        self.assertEqual(
            set(REGRESSION_NUMERIC_FEATURE_COLUMNS)
            | set(REGRESSION_CATEGORICAL_FEATURE_COLUMNS),
            set(REGRESSION_FEATURE_COLUMNS),
        )

    def test_train_test_split_is_reproducible_and_disjoint(self) -> None:
        train_a, test_a = split_train_test_indices(100)
        train_b, test_b = split_train_test_indices(100)
        np.testing.assert_array_equal(train_a, train_b)
        np.testing.assert_array_equal(test_a, test_b)
        self.assertFalse(set(train_a).intersection(test_a))
        self.assertEqual(len(train_a) + len(test_a), 100)
        with self.assertRaises(ValueError):
            split_train_test_indices(1)

    def test_preprocessing_fits_on_training_data_and_ignores_new_categories(self) -> None:
        dataset = small_regression_dataset()
        train = dataset.features.iloc[:80].copy()
        train.loc[:, "delivery_distance_km"] = 2.0
        train.loc[:, "City"] = "Known city"
        train.loc[0, "distance_band"] = pd.NA
        train.loc[0, "City"] = pd.NA
        pipeline = build_regression_pipeline("Linear Regression", LinearRegression())
        pipeline.fit(train, dataset.target.iloc[:80])

        imputer = pipeline.named_steps["preprocessor"].named_transformers_["numeric"].named_steps[
            "imputer"
        ]
        self.assertEqual(imputer.statistics_[0], 2.0)
        encoder = pipeline.named_steps["preprocessor"].named_transformers_[
            "categorical"
        ].named_steps["encoder"]
        city_categories = encoder.categories_[
            list(REGRESSION_CATEGORICAL_FEATURE_COLUMNS).index("City")
        ].tolist()
        self.assertEqual(city_categories, ["(Missing)", "Known city"])
        self.assertTrue(np.isfinite(pipeline.predict(dataset.features.iloc[[90]])).all())

    def test_metric_calculation_matches_manual_values(self) -> None:
        metrics = calculate_regression_metrics([10, 20, 30], [12, 18, 30])
        self.assertAlmostEqual(metrics["mae_minutes"], 4 / 3)
        self.assertAlmostEqual(metrics["rmse_minutes"], np.sqrt(8 / 3))
        self.assertAlmostEqual(metrics["median_absolute_error_minutes"], 2.0)
        self.assertAlmostEqual(metrics["r2"], 0.96)
        with self.assertRaises(ValueError):
            calculate_regression_metrics([1, 2], [1])
        with self.assertRaises(ValueError):
            calculate_regression_metrics([1, np.nan], [1, 2])

    def test_training_saves_comparison_model_and_test_predictions(self) -> None:
        dataset = small_regression_dataset()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "processed.csv"
            source.write_text("test data source", encoding="utf-8")
            model_path = root / "models" / "eta.joblib"
            report_path = root / "metrics" / "report.json"
            predictions_path = root / "metrics" / "predictions.csv"
            importance_path = root / "metrics" / "importance.csv"
            report = train_regression_models(
                dataset,
                model_path=model_path,
                report_path=report_path,
                predictions_path=predictions_path,
                importance_path=importance_path,
                source_path=source,
                cv_splits=2,
                search_iterations=1,
                permutation_repeats=1,
            )

            self.assertTrue(model_path.is_file())
            self.assertTrue(report_path.is_file())
            self.assertTrue(predictions_path.is_file())
            self.assertTrue(importance_path.is_file())
            self.assertEqual(
                json.loads(report_path.read_text(encoding="utf-8"))["metadata"][
                    "training_configuration"
                ]["test_set_used_for_tuning"],
                False,
            )
            self.assertEqual(len(report["model_comparison"]), 5)
            self.assertEqual(report["metadata"]["training_configuration"]["test_rows"], 20)
            self.assertGreater(
                report["metadata"]["training_configuration"]["temporal_validation_rows"],
                0,
            )
            self.assertEqual(len(pd.read_csv(predictions_path)), 20)

            bundle = load_model_bundle(model_path)
            self.assertEqual(bundle["metadata"]["selected_model"], report["metadata"]["selected_model"])
            prediction = predict_delivery_time(
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
            self.assertGreaterEqual(
                prediction["predicted_delivery_time_minutes"], 0.0
            )
            self.assertEqual(prediction["model_version"], "1.0.0")

    def test_prediction_validates_inputs_and_reports_missing_predictors(self) -> None:
        with self.assertRaisesRegex(ValueError, "Delivery_person_ID"):
            build_order_features({"Delivery_person_ID": "rider-1"})
        with self.assertRaises(ValueError):
            build_order_features({"City": " "})
        features, warnings = build_order_features(
            {
                "Order_Date": "not-a-date",
                "Time_Orderd": "25:90:00",
                "City": "Urban",
            }
        )
        self.assertEqual(tuple(features.columns), REGRESSION_FEATURE_COLUMNS)
        self.assertTrue(warnings)
        self.assertTrue(features["delivery_distance_km"].isna().all())


if __name__ == "__main__":
    unittest.main()
