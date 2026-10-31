"""Unit tests for the Phase 1 data audit and feature pipeline."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from config import FEATURE_AVAILABILITY_POLICY, PRE_DELIVERY_FEATURE_COLUMNS
from src.data.cleaning import (
    clean_orders,
    extract_delivery_time_minutes,
    normalize_missing_values,
)
from src.data.feature_engineering import (
    coordinate_validity,
    engineer_features,
    get_pre_delivery_features,
    haversine_km,
    label_late_orders,
    outlier_flags,
    parse_timestamps,
)
from src.data.pipeline import run_data_pipeline


def sample_orders() -> pd.DataFrame:
    """Return a small schema-complete fixture with normal and invalid examples."""
    return pd.DataFrame(
        {
            "ID": ["a", "b", "c"],
            "Delivery_person_ID": ["r1", "r2", "r3"],
            "Delivery_person_Age": ["25", "NaN ", "40"],
            "Delivery_person_Ratings": ["4.8", "5.5", "3.9"],
            "Restaurant_latitude": [12.0, 0.0, 12.0],
            "Restaurant_longitude": [77.0, 0.0, 77.0],
            "Delivery_location_latitude": [12.01, 12.02, 12.01],
            "Delivery_location_longitude": [77.01, 77.02, 77.01],
            "Order_Date": ["01-03-2022", "bad-date", "05-03-2022"],
            "Time_Orderd": ["11:30:00", "NaN ", "23:55:00"],
            "Time_Order_picked": ["11:45:00", "12:05:00", "00:05:00"],
            "Weatherconditions": ["conditions Sunny", "conditions NaN", "Cloudy"],
            "Road_traffic_density": ["High ", "NaN ", " Low"],
            "Vehicle_condition": ["2", "1", "3"],
            "Type_of_order": ["Snack", "Drinks", "Meal"],
            "Type_of_vehicle": ["motorcycle", "scooter", "motorcycle"],
            "multiple_deliveries": ["0", "NaN ", "2"],
            "Festival": ["No ", "Yes", "No"],
            "City": ["Urban", "NaN ", "Metropolitian"],
            "Time_taken(min)": ["(min) 30", "(min) 31", "(min) 20"],
        },
        dtype="string",
    )


class DeliveryTimeTests(unittest.TestCase):
    def test_delivery_time_extraction(self) -> None:
        values = pd.Series(["(min) 24", "(min) 30", "(min) 9", "bad", pd.NA])
        actual = extract_delivery_time_minutes(values)
        self.assertEqual(actual.iloc[:3].tolist(), [24, 30, 9])
        self.assertTrue(actual.iloc[3:].isna().all())

    def test_missing_value_handling_and_cleaning(self) -> None:
        cleaned, operations = clean_orders(sample_orders())
        self.assertTrue(pd.isna(cleaned.loc[1, "Delivery_person_Age"]))
        self.assertTrue(pd.isna(cleaned.loc[1, "Weatherconditions"]))
        self.assertEqual(cleaned.loc[0, "Weatherconditions"], "Sunny")
        self.assertEqual(cleaned.loc[0, "Road_traffic_density"], "High")
        self.assertEqual(cleaned["Time_taken(min)"].dtype.name, "Int64")
        self.assertEqual(len(cleaned), 3)
        self.assertTrue(all(operation["records_excluded"] == 0 for operation in operations))

    def test_unparseable_numeric_values_are_reported_not_deleted(self) -> None:
        frame = sample_orders()
        frame.loc[0, "Delivery_person_Age"] = "unknown"
        frame.loc[0, "Time_taken(min)"] = "invalid"
        cleaned, operations = clean_orders(frame)
        numeric_log = next(
            operation
            for operation in operations
            if operation["operation"] == "Convert numeric fields"
        )
        self.assertTrue(pd.isna(cleaned.loc[0, "Delivery_person_Age"]))
        self.assertTrue(pd.isna(cleaned.loc[0, "Time_taken(min)"]))
        self.assertEqual(
            numeric_log["unparseable_nonmissing_values_by_column"][
                "Delivery_person_Age"
            ],
            1,
        )
        self.assertEqual(
            numeric_log["unparseable_nonmissing_values_by_column"][
                "Time_taken(min)"
            ],
            1,
        )
        self.assertEqual(len(cleaned), len(frame))

    def test_all_supported_missing_representations_are_normalized(self) -> None:
        frame = pd.DataFrame(
            {
                "ordinary": ["", "   ", "NaN ", None, "null"],
                "Weatherconditions": [
                    "conditions NaN",
                    " Sunny ",
                    "Fog",
                    "Cloudy",
                    "Stormy",
                ],
            },
            dtype="string",
        )
        cleaned, _ = normalize_missing_values(frame)
        self.assertTrue(cleaned["ordinary"].isna().all())
        self.assertTrue(pd.isna(cleaned.loc[0, "Weatherconditions"]))
        self.assertEqual(cleaned.loc[1, "Weatherconditions"], " Sunny ")

    def test_haversine_distance(self) -> None:
        distance = haversine_km(0.0, 0.0, 0.0, 1.0)
        self.assertAlmostEqual(float(distance), 111.195, places=2)
        vector = haversine_km(
            pd.Series([12.0, 0.0, 91.0]),
            pd.Series([77.0, 0.0, 77.0]),
            pd.Series([12.01, 12.02, 12.0]),
            pd.Series([77.01, 77.02, 77.0]),
        )
        self.assertGreater(float(vector.iloc[0]), 0)
        self.assertGreater(float(vector.iloc[1]), 0)
        self.assertTrue(pd.isna(vector.iloc[2]))

    def test_invalid_coordinates_are_flagged_and_distance_is_missing(self) -> None:
        cleaned, _ = clean_orders(sample_orders())
        enriched = engineer_features(cleaned)
        self.assertTrue(bool(enriched.loc[1, "zero_coordinate_pair"]))
        self.assertFalse(bool(enriched.loc[1, "coordinates_valid"]))
        self.assertTrue(pd.isna(enriched.loc[1, "delivery_distance_km"]))
        self.assertGreater(enriched.loc[1, "raw_haversine_distance_km"], 100)
        self.assertTrue(bool(enriched.loc[1, "distance_over_review_threshold"]))
        self.assertTrue(bool(enriched.loc[0, "coordinates_valid"]))
        range_valid, region_valid, _ = coordinate_validity(cleaned)
        self.assertTrue(bool(range_valid.loc[1]))
        self.assertFalse(bool(region_valid.loc[1]))
        self.assertTrue(np.isnan(haversine_km(91.0, 0.0, 0.0, 0.0)))

    def test_calendar_and_peak_features(self) -> None:
        cleaned, _ = clean_orders(sample_orders())
        enriched = engineer_features(cleaned)
        self.assertEqual(enriched.loc[0, "order_hour"], 11)
        self.assertEqual(enriched.loc[0, "day_of_week"], 1)
        self.assertFalse(bool(enriched.loc[0, "is_weekend"]))
        self.assertTrue(bool(enriched.loc[0, "is_peak_hour"]))
        self.assertTrue(bool(enriched.loc[2, "is_weekend"]))
        self.assertFalse(bool(enriched.loc[2, "is_peak_hour"]))
        self.assertEqual(enriched.loc[0, "distance_band"], "0-3 km")

    def test_timestamp_parsing_uses_declared_formats(self) -> None:
        cleaned, _ = clean_orders(sample_orders())
        parsed = parse_timestamps(cleaned)
        self.assertEqual(str(parsed["Order_Date"].iloc[0].date()), "2022-03-01")
        self.assertTrue(pd.isna(parsed["Order_Date"].iloc[1]))
        self.assertTrue(pd.isna(parsed["Time_Orderd"].iloc[1]))
        self.assertEqual(parsed["Time_Order_picked"].notna().sum(), 3)

    def test_sla_classification_is_strictly_greater_than_threshold(self) -> None:
        labels = label_late_orders(pd.Series([29, 30, 31, pd.NA]), 30)
        self.assertEqual(labels.iloc[:3].tolist(), [False, False, True])
        self.assertTrue(pd.isna(labels.iloc[3]))
        with self.assertRaises(ValueError):
            label_late_orders(pd.Series([1]), 0)

    def test_outlier_flags_report_thresholds_without_removing_values(self) -> None:
        values = pd.Series([1, 2, 3, 4, 50, pd.NA])
        iqr_flags, percentile_flags, thresholds = outlier_flags(values)
        self.assertTrue(bool(iqr_flags.iloc[4]))
        self.assertTrue(bool(percentile_flags.iloc[4]))
        self.assertTrue(pd.isna(iqr_flags.iloc[5]))
        self.assertEqual(thresholds["valid_count"], 5)
        self.assertEqual(len(values), 6)

    def test_leakage_exclusion(self) -> None:
        cleaned, _ = clean_orders(sample_orders())
        enriched = engineer_features(cleaned)
        selected = get_pre_delivery_features(enriched)
        self.assertEqual(tuple(selected.columns), PRE_DELIVERY_FEATURE_COLUMNS)
        self.assertTrue(
            set(PRE_DELIVERY_FEATURE_COLUMNS).issubset(FEATURE_AVAILABILITY_POLICY)
        )
        forbidden = {
            "ID",
            "Delivery_person_ID",
            "Time_Order_picked",
            "Time_taken(min)",
            "is_late_order",
            "raw_haversine_distance_km",
            "distance_over_review_threshold",
            "Delivery_person_Age",
            "Delivery_person_Ratings",
            "Vehicle_condition",
            "Type_of_vehicle",
            "multiple_deliveries",
        }
        self.assertTrue(forbidden.isdisjoint(selected.columns))

    def test_cleaning_pipeline_consistency_and_report(self) -> None:
        fixture = sample_orders()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            input_path = root / "train.csv"
            fixture.to_csv(input_path, index=False)
            original_bytes = input_path.read_bytes()
            result = run_data_pipeline(
                input_path=input_path,
                processed_path=root / "processed" / "orders.csv",
                report_dir=root / "reports",
            )
            self.assertEqual(input_path.read_bytes(), original_bytes)
            self.assertEqual(result.input_records, 3)
            self.assertEqual(result.retained_records, 3)
            self.assertEqual(result.excluded_records, 0)
            processed = pd.read_csv(result.processed_path)
            self.assertEqual(len(processed), 3)
            self.assertIn("delivery_distance_km", processed.columns)
            self.assertIn("raw_haversine_distance_km", processed.columns)
            report = json.loads(result.json_report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["records"]["excluded_records"], 0)
            self.assertEqual(report["delay_lens"]["sla_threshold_minutes"], 30)
            self.assertEqual(
                report["delay_lens"]["observed_outcomes"]["late_orders"], 1
            )
            self.assertEqual(
                report["delay_lens"]["observed_outcomes"]["not_late_orders"], 2
            )
            self.assertEqual(
                report["outlier_analysis"]["distance_km"][
                    "pairs_over_review_threshold"
                ],
                1,
            )
            self.assertTrue(result.markdown_report_path.is_file())


if __name__ == "__main__":
    unittest.main()
