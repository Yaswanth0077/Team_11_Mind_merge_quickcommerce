"""Tests for PS3 descriptive analytics and shared dashboard filters."""

from __future__ import annotations

import json
import unittest
from datetime import date

import pandas as pd

from config import LATE_LABEL_COLUMN, TARGET_COLUMN
from src.analytics.delay_analysis import (
    DelayFilters,
    analyze_by_category,
    analyze_by_day,
    analyze_by_hour,
    analyze_distance_bands,
    analyze_ratings,
    analyze_weather_traffic,
    analysis_quality_summary,
    apply_delay_filters,
    benchmark_comparison_to_network,
    build_analysis_report,
    delivery_time_summary,
    network_benchmark_metrics,
    prepare_delay_data,
    rider_performance,
)


def delay_orders() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ID": [f"order-{index}" for index in range(8)],
            "Delivery_person_ID": ["r1", "r1", "r1", "r1", "r2", "r2", "r2", "r2"],
            "Delivery_person_Age": ["25"] * 8,
            "Delivery_person_Ratings": ["4.0", "4.5", "5.0", "5.5", "", "3.0", "4.0", "4.0"],
            "Restaurant_latitude": [12.0] * 8,
            "Restaurant_longitude": [77.0] * 8,
            "Delivery_location_latitude": [12.01, 12.02, 12.03, 12.04, 12.05, 0, 12.07, 12.08],
            "Delivery_location_longitude": [77.01, 77.02, 77.03, 77.04, 77.05, 0, 77.07, 77.08],
            "Order_Date": [
                "01-03-2022",
                "01-03-2022",
                "02-03-2022",
                "03-03-2022",
                "04-03-2022",
                "05-03-2022",
                "",
                "07-03-2022",
            ],
            "Time_Orderd": [
                "09:00:00",
                "09:30:00",
                "12:00:00",
                "18:00:00",
                "",
                "10:00:00",
                "11:00:00",
                "bad-time",
            ],
            "Time_Order_picked": ["09:10:00"] * 8,
            "Weatherconditions": [
                "conditions Sunny",
                "Sunny",
                "Cloudy",
                "Stormy",
                "conditions NaN",
                "Sunny",
                "Cloudy",
                "Stormy",
            ],
            "Road_traffic_density": ["Low", "Low", "High", "Jam", "Medium", "Low", "High", "Jam"],
            "Type_of_order": ["Snack", "Snack", "Meal", "Meal", "Drinks", "Snack", "Meal", "Drinks"],
            "City": ["Urban", "Urban", "Urban", "Semi-Urban", "Urban", "Urban", "Semi-Urban", "Urban"],
            TARGET_COLUMN: ["(min) 20", "(min) 30", "(min) 31", "(min) 40", "(min) 25", "(min) 50", "bad", "(min) 29"],
            "delivery_distance_km": [2.0, 2.5, 4.0, 8.0, 2.0, 0.0, 13.0, 4.0],
        },
        dtype="string",
    )


class DelayAnalysisTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.data = prepare_delay_data(delay_orders(), sla_threshold_minutes=30)

    def test_delivery_time_and_late_rate_use_valid_target_rows(self) -> None:
        summary = delivery_time_summary(self.data)
        self.assertEqual(summary["order_count"], 7)
        self.assertEqual(summary["late_orders"], 3)
        self.assertAlmostEqual(summary["late_rate"], 3 / 7)
        self.assertEqual(self.data[LATE_LABEL_COLUMN].iloc[1], False)
        self.assertEqual(self.data[LATE_LABEL_COLUMN].iloc[2], True)
        self.assertTrue(pd.isna(self.data[LATE_LABEL_COLUMN].iloc[6]))

    def test_hourly_summary_excludes_missing_time_but_keeps_sample_counts(self) -> None:
        hourly = analyze_by_hour(self.data)
        self.assertEqual(hourly["order_count"].sum(), 5)
        nine_am = hourly.loc[hourly["hour"] == 9].iloc[0]
        self.assertEqual(nine_am["order_count"], 2)
        self.assertEqual(nine_am["late_rate"], 0)
        noon = hourly.loc[hourly["hour"] == 12].iloc[0]
        self.assertEqual(noon["late_rate"], 1)

    def test_day_summary_orders_weekdays_and_reports_volumes(self) -> None:
        daily = analyze_by_day(self.data)
        self.assertEqual(daily["order_count"].sum(), 7)
        self.assertEqual(
            daily["day_name"].tolist(),
            ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"],
        )
        self.assertEqual(daily["order_count"].tolist(), [1, 2, 1, 1, 1, 1])

    def test_weather_and_traffic_summary_retains_missing_categories(self) -> None:
        weather = analyze_by_category(self.data, "weather")
        missing_weather = weather.loc[weather["analysis_weather"] == "(Missing)"].iloc[0]
        self.assertEqual(missing_weather["order_count"], 1)
        traffic = analyze_by_category(self.data, "traffic")
        self.assertEqual(
            traffic["analysis_traffic"].tolist(),
            ["Low", "Medium", "High", "Jam"],
        )
        comparison = analyze_weather_traffic(self.data)
        self.assertEqual(comparison["order_count"].sum(), 7)
        self.assertTrue(comparison["late_rate"].between(0, 1).all())

    def test_distance_bands_use_only_valid_coordinate_distances(self) -> None:
        bands = analyze_distance_bands(self.data)
        self.assertEqual(bands["order_count"].sum(), 6)
        counts = dict(zip(bands["analysis_distance_band"], bands["order_count"]))
        self.assertEqual(counts["0-3 km"], 3)
        self.assertEqual(counts["3-7 km"], 2)
        self.assertEqual(counts["7-12 km"], 1)
        self.assertEqual(counts.get("12+ km", 0), 0)
        self.assertEqual(
            bands["analysis_distance_band"].tolist(),
            ["0-3 km", "3-7 km", "7-12 km"],
        )

    def test_rating_and_rider_summaries_apply_validity_and_minimum_sample(self) -> None:
        ratings = analyze_ratings(self.data)
        self.assertNotIn(5.5, ratings["rating"].tolist())
        self.assertEqual(ratings["order_count"].sum(), 5)
        riders = rider_performance(self.data, minimum_orders=4)
        self.assertEqual(riders["meets_minimum_sample"].sum(), 1)
        self.assertEqual(riders["order_count"].sum(), 7)
        self.assertFalse(rider_performance(self.data, minimum_orders=5)["meets_minimum_sample"].any())

    def test_quality_summary_separates_missing_and_invalid_values(self) -> None:
        quality = analysis_quality_summary(self.data)
        self.assertEqual(quality["records_retained_by_preparation"], 8)
        self.assertEqual(quality["records_excluded_by_preparation"], 0)
        self.assertEqual(quality["invalid_coordinate_records"], 1)
        self.assertEqual(quality["missing_coordinate_records"], 0)
        self.assertEqual(quality["missing_rating_records"], 1)
        self.assertEqual(quality["invalid_rating_records"], 1)
        self.assertEqual(quality["missing_weather_records"], 1)

    def test_default_and_filtered_selections_are_consistent(self) -> None:
        default = apply_delay_filters(self.data)
        self.assertEqual(len(default), 7)
        self.assertEqual(default["analysis_hour"].isna().sum(), 2)
        selected = apply_delay_filters(
            self.data,
            DelayFilters(
                start_date=date(2022, 3, 2),
                end_date=date(2022, 3, 3),
                first_hour=10,
                last_hour=13,
                weather=("Cloudy",),
                traffic=("High",),
                order_type=("Meal",),
                city=("Urban",),
                distance_km=(3.0, 7.0),
            ),
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected.iloc[0][TARGET_COLUMN], 31)
        no_categories = apply_delay_filters(self.data, DelayFilters(weather=()))
        self.assertTrue(no_categories.empty)

    def test_empty_data_and_invalid_distance_ranges_are_safe(self) -> None:
        empty = prepare_delay_data(delay_orders().iloc[0:0])
        self.assertIsNone(delivery_time_summary(empty)["late_rate"])
        self.assertTrue(analyze_by_hour(empty).empty)
        self.assertTrue(analyze_weather_traffic(empty).empty)
        with self.assertRaises(ValueError):
            apply_delay_filters(self.data, DelayFilters(distance_km=(5.0, 2.0)))

    def test_fixed_network_benchmark_reports_absolute_and_rate_difference(self) -> None:
        selected = self.data.iloc[:2]
        benchmark = benchmark_comparison_to_network(
            selected,
            network_benchmark_metrics(self.data),
        ).set_index("metric")
        late = benchmark.loc["Late-order rate (%)"]
        self.assertAlmostEqual(late["selected"], 0.0)
        self.assertAlmostEqual(late["overall_network"], 3 / 7 * 100)
        self.assertAlmostEqual(late["difference"], -3 / 7 * 100)
        self.assertEqual(late["difference_unit"], "percentage points")

    def test_report_contains_json_serializable_values_and_actual_sla(self) -> None:
        report = build_analysis_report(self.data, sla_threshold_minutes=30)
        encoded = json.dumps(report, allow_nan=False)
        self.assertIn('"sla_threshold_minutes": 30', encoded)
        self.assertEqual(report["quality"]["valid_delivery_time_records"], 7)
        self.assertIn("associations", report["interpretation"])


if __name__ == "__main__":
    unittest.main()
