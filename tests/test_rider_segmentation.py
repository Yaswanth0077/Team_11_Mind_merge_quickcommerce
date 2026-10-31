"""PS7 rider aggregation, clustering, persistence, and filtering tests."""

from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from config import RIDER_MIN_COMPLETED_DELIVERIES, SLA_THRESHOLD_MINUTES
from src.analytics.rider_segmentation import (
    FEATURE_COLUMNS,
    RiderDataset,
    _select_cluster_count,
    aggregate_rider_performance,
    build_rider_preprocessor,
    evaluate_segment_counts,
    fit_rider_segments,
    load_rider_segment_model,
    run_rider_segmentation,
)
from src.visualization.rider_segmentation_dashboard import (
    ELIGIBLE_RIDERS,
    INSUFFICIENT_RIDERS,
    build_segment_support_table,
    filter_rider_assignments,
    summarize_selected_segments,
)


def prepared_rider_orders(
    *,
    riders: int = 8,
    deliveries_per_rider: int = 40,
    include_low_volume: bool = True,
) -> pd.DataFrame:
    """Create reproducible prepared records with distinct observed performance profiles."""
    data: list[dict[str, object]] = []
    for rider_number in range(riders):
        slow_group = rider_number >= riders // 2
        count = deliveries_per_rider
        for order_number in range(count):
            delivery = (
                36 if slow_group and order_number % 3 == 0 else
                31 if slow_group and order_number % 4 == 0 else
                24 if not slow_group and order_number % 5 == 0 else 26
            )
            data.append(
                {
                    "ID": f"order-{rider_number}-{order_number}",
                    "Delivery_person_ID": f"rider-{rider_number}",
                    "Time_taken(min)": delivery,
                    "analysis_valid_order": True,
                    "analysis_distance_km": (
                        np.nan
                        if rider_number == 0
                        else (4.0 if not slow_group else 9.0)
                    ),
                    "analysis_rating": (
                        np.nan
                        if rider_number == 1
                        else (4.8 if not slow_group else 4.1)
                    ),
                    "multiple_deliveries": (
                        np.nan
                        if rider_number == 2
                        else (0 if not slow_group else 2)
                    ),
                }
            )
    if include_low_volume:
        for order_number in range(10):
            data.append(
                {
                    "ID": f"low-order-{order_number}",
                    "Delivery_person_ID": "low-volume-rider",
                    "Time_taken(min)": 22,
                    "analysis_valid_order": True,
                    "analysis_distance_km": 3.0,
                    "analysis_rating": 4.5,
                    "multiple_deliveries": 0,
                }
            )
    return pd.DataFrame(data)


def small_rider_dataset() -> RiderDataset:
    riders = prepared_rider_orders()
    return aggregate_rider_performance(
        riders,
        minimum_deliveries=30,
        minimum_rating_observations=5,
    )


class RiderSegmentationTests(unittest.TestCase):
    def test_rider_aggregation_and_missing_values_are_transparent(self) -> None:
        fixture = prepared_rider_orders()
        fixture.loc[0, "Time_taken(min)"] = SLA_THRESHOLD_MINUTES
        result = aggregate_rider_performance(fixture)
        self.assertEqual(len(result.riders), 9)
        self.assertEqual(result.audit["completed_deliveries_with_valid_rider_id"], 330)
        self.assertEqual(result.audit["riders_meeting_minimum_deliveries"], 8)
        self.assertEqual(result.audit["riders_flagged_insufficient_deliveries"], 1)
        self.assertFalse(
            result.riders.loc[
                result.riders["rider_id"].eq("low-volume-rider"),
                "sufficient_observations",
            ].item()
        )
        self.assertTrue(
            pd.isna(
                result.riders.loc[
                    result.riders["rider_id"].eq("rider-1"),
                    "mean_rider_rating",
                ].item()
            )
        )
        self.assertTrue(
            result.riders.loc[
                result.riders["rider_id"].eq("rider-1"),
                "rating_observation_count",
            ].item()
            < 5
        )
        self.assertEqual(
            result.audit["feature_missing_values_among_eligible_riders"][
                "mean_delivery_distance_km"
            ],
            1,
        )
        self.assertTrue(
            result.riders["late_rate_ci_low"].le(result.riders["late_rate"]).all()
        )
        self.assertTrue(
            result.riders["late_rate_ci_high"].ge(result.riders["late_rate"]).all()
        )
        first_rider = result.riders.loc[
            result.riders["rider_id"].eq("rider-0")
        ].iloc[0]
        self.assertEqual(
            int(first_rider["late_delivery_count"]),
            int(
                result.orders.loc[
                    result.orders["rider_id"].eq("rider-0"),
                    "delivery_minutes",
                ]
                .gt(SLA_THRESHOLD_MINUTES)
                .sum()
            ),
        )

    def test_minimum_delivery_requirement_is_inclusive_and_configurable(self) -> None:
        fixture = prepared_rider_orders(riders=3, deliveries_per_rider=30)
        result = aggregate_rider_performance(
            fixture,
            minimum_deliveries=30,
        )
        self.assertEqual(int(result.riders["sufficient_observations"].sum()), 3)
        lower = aggregate_rider_performance(
            fixture,
            minimum_deliveries=31,
        )
        self.assertEqual(int(lower.riders["insufficient_observations"].sum()), 4)
        self.assertEqual(RIDER_MIN_COMPLETED_DELIVERIES, 30)

    def test_completed_rows_without_valid_rider_id_are_audited_not_grouped(self) -> None:
        fixture = prepared_rider_orders()
        fixture.loc[0, "Delivery_person_ID"] = " null "
        result = aggregate_rider_performance(fixture)
        self.assertEqual(
            result.audit["completed_deliveries_missing_or_invalid_rider_id"],
            1,
        )
        self.assertEqual(len(result.orders), len(fixture) - 1)
        self.assertNotIn("null", set(result.riders["rider_id"]))

    def test_feature_selection_and_training_only_scaling_pipeline(self) -> None:
        dataset = small_rider_dataset()
        eligible = dataset.riders.loc[dataset.riders["sufficient_observations"]]
        self.assertEqual(FEATURE_COLUMNS, (
            "log_delivery_count",
            "mean_delivery_minutes",
            "late_rate",
            "mean_delivery_distance_km",
            "mean_rider_rating",
            "mean_multiple_deliveries",
        ))
        self.assertNotIn("rider_id", FEATURE_COLUMNS)
        self.assertNotIn("Delivery_person_Age", FEATURE_COLUMNS)
        self.assertNotIn("ID", FEATURE_COLUMNS)
        preprocessor = build_rider_preprocessor()
        scaled = preprocessor.fit_transform(
            eligible.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype="float64")
        )
        self.assertTrue(np.isfinite(scaled).all())
        self.assertTrue(np.allclose(scaled.mean(axis=0), 0.0, atol=1e-10))
        self.assertEqual(scaled.shape[0], len(eligible))
        self.assertEqual(scaled.shape[1], len(FEATURE_COLUMNS))

    def test_k_diagnostics_and_fixed_seed_reproducibility(self) -> None:
        dataset = small_rider_dataset()
        eligible = dataset.riders.loc[dataset.riders["sufficient_observations"]]
        preprocessor = build_rider_preprocessor()
        scaled = preprocessor.fit_transform(
            eligible.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype="float64")
        )
        first, sample_a = evaluate_segment_counts(
            scaled,
            cluster_counts=(2, 3, 4),
            silhouette_sample_size=8,
            random_seed=42,
        )
        second, sample_b = evaluate_segment_counts(
            scaled,
            cluster_counts=(2, 3, 4),
            silhouette_sample_size=8,
            random_seed=42,
        )
        pd.testing.assert_frame_equal(first, second)
        self.assertEqual(sample_a, sample_b)
        self.assertTrue(first["inertia"].ge(0).all())
        self.assertTrue(first["silhouette_score"].between(-1, 1).all())
        self.assertEqual(first["cluster_count"].tolist(), [2, 3, 4])

    def test_k_selection_balances_elbow_silhouette_and_segment_size(self) -> None:
        evaluations = pd.DataFrame(
            {
                "cluster_count": [2, 3, 4],
                "inertia": [100.0, 50.0, 20.0],
                "silhouette_score": [0.20, 0.195, 0.18],
                "smallest_segment_riders": [200, 80, 31],
                "largest_segment_riders": [500, 300, 250],
            }
        )
        selected, explanation = _select_cluster_count(evaluations)
        self.assertEqual(selected, 3)
        self.assertIn("inertia-elbow K", explanation)
        evaluations.loc[1, "silhouette_score"] = 0.15
        selected, _ = _select_cluster_count(evaluations)
        self.assertEqual(selected, 2)

    def test_cluster_assignments_and_segment_metrics(self) -> None:
        report, riders, order_assignments, _ = fit_rider_segments(
            small_rider_dataset(),
            cluster_counts=(2, 3, 4),
            silhouette_sample_size=8,
        )
        self.assertEqual(report["metadata"]["selected_cluster_count"], 2)
        self.assertEqual(riders["segment_id"].notna().sum(), 8)
        self.assertEqual(order_assignments["segment_id"].notna().sum(), 320)
        segments = pd.DataFrame(report["segments"])
        self.assertEqual(segments["rider_count"].sum(), 8)
        self.assertEqual(segments["total_deliveries"].sum(), 320)
        self.assertTrue(segments["late_rate"].between(0, 1).all())
        self.assertAlmostEqual(segments["rider_share"].sum(), 1.0)
        self.assertTrue(segments["operational_description"].str.len().gt(0).all())
        self.assertTrue(segments["small_segment_flag"].any())
        self.assertTrue(
            segments.loc[segments["small_segment_flag"], "rider_count"].lt(30).all()
        )
        for segment in segments.itertuples(index=False):
            segment_orders = order_assignments.loc[
                order_assignments["segment_id"].eq(segment.segment_id)
            ]
            self.assertAlmostEqual(
                segment.mean_delivery_minutes,
                segment_orders["delivery_minutes"].mean(),
            )
            self.assertAlmostEqual(
                segment.median_delivery_minutes,
                segment_orders["delivery_minutes"].median(),
            )
            self.assertAlmostEqual(
                segment.late_rate,
                segment_orders["is_late"].mean(),
            )
        self.assertEqual(
            int(
                riders.loc[
                    riders["rider_id"].eq("low-volume-rider"),
                    "segment_id",
                ].isna().sum()
            ),
            1,
        )

    def test_support_hypotheses_follow_filtered_segment_metrics(self) -> None:
        metrics = pd.DataFrame(
            [
                {
                    "segment_id": 0,
                    "rider_count": 264,
                    "late_rate": 0.348,
                    "mean_delivery_minutes": 27.35,
                    "median_deliveries_per_rider": 57.0,
                },
                {
                    "segment_id": 2,
                    "rider_count": 243,
                    "late_rate": 0.252,
                    "mean_delivery_minutes": 25.33,
                    "median_deliveries_per_rider": 45.0,
                },
            ]
        )
        table = build_segment_support_table(metrics)
        self.assertEqual(table["Segment (not a rank)"].tolist(), [0, 2])
        self.assertIn("34.8% observed late rate", table.iloc[0]["Observed group profile"])
        self.assertIn("not proof", table.iloc[1]["Potential operational support"])
        self.assertTrue(table["Suggested training / process review"].str.len().gt(0).all())
        self.assertTrue(table["Roster / workload consideration"].str.len().gt(0).all())
        self.assertTrue(table["Monitoring KPI"].str.len().gt(0).all())
        self.assertTrue(table["Limitations / validation"].str.len().gt(0).all())

    def test_dashboard_segment_and_volume_filters_recompute_metrics(self) -> None:
        _, riders, order_assignments, _ = fit_rider_segments(
            small_rider_dataset(),
            cluster_counts=(2, 3),
            silhouette_sample_size=8,
        )
        eligible = filter_rider_assignments(riders, segment=ELIGIBLE_RIDERS)
        insufficient = filter_rider_assignments(
            riders,
            segment=INSUFFICIENT_RIDERS,
        )
        self.assertEqual(len(eligible), 8)
        self.assertEqual(len(insufficient), 1)
        first_id = int(eligible["segment_id"].iloc[0])
        selected = filter_rider_assignments(riders, segment=f"Segment {first_id}")
        filtered = filter_rider_assignments(
            riders,
            segment=ELIGIBLE_RIDERS,
            delivery_count_range=(40, 40),
        )
        self.assertLessEqual(len(selected), len(eligible))
        self.assertEqual(len(filtered), 8)
        summary = summarize_selected_segments(selected, order_assignments)
        self.assertEqual(int(summary["rider_count"].sum()), len(selected))
        self.assertEqual(int(summary["total_deliveries"].sum()), len(selected) * 40)
        with self.assertRaises(ValueError):
            filter_rider_assignments(riders, segment=99)

    def test_model_and_output_persistence(self) -> None:
        dataset = small_rider_dataset()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed.csv"
            processed.write_text("test data", encoding="utf-8")
            model_path = root / "model.joblib"
            report_path = root / "report.json"
            riders_path = root / "riders.csv"
            orders_path = root / "orders.csv"
            with patch(
                "src.analytics.rider_segmentation.load_rider_dataset",
                return_value=dataset,
            ):
                report = run_rider_segmentation(
                    processed,
                    model_path=model_path,
                    report_path=report_path,
                    assignments_path=riders_path,
                    order_assignments_path=orders_path,
                    cluster_counts=(2, 3),
                    silhouette_sample_size=8,
                )
            self.assertTrue(model_path.is_file())
            self.assertTrue(report_path.is_file())
            self.assertTrue(riders_path.is_file())
            self.assertTrue(orders_path.is_file())
            loaded = load_rider_segment_model(model_path)
            self.assertEqual(loaded["metadata"]["selected_cluster_count"], 2)
            self.assertEqual(report["metadata"]["selected_cluster_count"], 2)
            eligible = dataset.riders.loc[dataset.riders["sufficient_observations"]]
            transformed = loaded["preprocessor"].transform(
                eligible.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype="float64")
            )
            self.assertEqual(
                loaded["model"].predict(transformed).shape,
                (len(eligible),),
            )
            self.assertEqual(len(pd.read_csv(riders_path)), 9)
            self.assertEqual(len(pd.read_csv(orders_path)), 330)
            parsed = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(len(parsed["segments"]), 2)


if __name__ == "__main__":
    unittest.main()
