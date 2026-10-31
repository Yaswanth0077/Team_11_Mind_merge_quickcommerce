"""PS6 geographic validation, clustering, evidence, and dashboard tests."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from src.analytics.demand_zones import (
    GeographicDataset,
    _cluster_summary,
    _fit_weighted_kmeans,
    _project_coordinates,
    assign_demand_zone,
    fit_demand_zones,
    load_demand_zone_model,
    load_geographic_dataset,
    run_demand_zone_analysis,
)
from src.visualization.demand_zone_dashboard import (
    aggregate_customer_locations,
    filter_zone_assignments,
)


def small_geographic_dataset(rows_per_zone: int = 40) -> GeographicDataset:
    """Make two deterministic, spatially separated groups of order observations."""
    rng = np.random.default_rng(42)
    centers = np.array([[12.96, 77.58], [13.08, 77.68]])
    coordinates = np.vstack(
        [
            center + rng.normal(0, 0.008, size=(rows_per_zone, 2))
            for center in centers
        ]
    )
    count = len(coordinates)
    labels = np.repeat([0, 1], rows_per_zone)
    orders = pd.DataFrame(
        {
            "source_row_number": np.arange(count) + 2,
            "customer_latitude": coordinates[:, 0],
            "customer_longitude": coordinates[:, 1],
            "delivery_minutes": np.where(labels == 0, 25, 36).astype(float),
            "is_late": labels.astype("int8"),
            "delivery_distance_km": np.where(labels == 0, 4.0, 8.0),
            "rider_id": np.where(labels == 0, "R1", "R2"),
        }
    )
    restaurants = pd.DataFrame(
        {
            "restaurant_latitude": [12.95, 13.10],
            "restaurant_longitude": [77.57, 77.69],
            "orders_at_location": [20, 20],
        }
    )
    return GeographicDataset(
        orders=orders,
        restaurants=restaurants,
        audit={"input_orders": count},
    )


class DemandZoneTests(unittest.TestCase):
    def test_customer_coordinate_validation_excludes_only_invalid_customers(self) -> None:
        raw = pd.DataFrame(
            {
                "Delivery_location_latitude": [13, 13, 0, 13, 39, None, 13.2, 13.1],
                "Delivery_location_longitude": [77.6, 77.7, 0, 77.5, 77.1, 77.4, 77.7, 77.8],
                "Restaurant_latitude": [13, 13, 13, 13, 13, 13, 0, 13],
                "Restaurant_longitude": [77, 77, 77, 77, 77, 77, 0, 77],
                "Delivery_person_ID": ["R1"] * 8,
            }
        )
        delay = pd.DataFrame(
            {
                "Time_taken(min)": [20, 30, 31, 40, 25, 20, np.nan, 60],
                "analysis_distance_km": [4.0] * 8,
            }
        )
        with (
            patch("src.analytics.demand_zones.load_orders", return_value=raw),
            patch("src.analytics.demand_zones.load_delay_data", return_value=delay),
        ):
            dataset = load_geographic_dataset(Path("unused.csv"))
        self.assertEqual(len(dataset.orders), 5)
        self.assertEqual(dataset.audit["orders_excluded_by_reason"]["zero_coordinate_pair"], 1)
        self.assertEqual(
            dataset.audit["orders_excluded_by_reason"]["outside_configured_india_envelope"],
            1,
        )
        self.assertEqual(
            dataset.audit["orders_excluded_by_reason"]["missing_or_unparseable"], 1
        )
        self.assertEqual(dataset.audit["missing_delivery_targets_among_clustered_orders"], 1)
        self.assertTrue(dataset.orders["is_late"].isna().any())
        self.assertEqual(len(dataset.restaurants), 1)

    def test_local_projection_is_reproducible_and_in_kilometres(self) -> None:
        latitude = np.array([12.0, 12.0])
        longitude = np.array([77.0, 77.01])
        projected = _project_coordinates(
            latitude,
            longitude,
            origin_latitude=12.0,
            origin_longitude=77.0,
            reference_latitude=12.0,
        )
        self.assertAlmostEqual(projected[0, 0], 0.0, places=9)
        self.assertGreater(projected[1, 0], 1.0)
        self.assertAlmostEqual(projected[1, 1], 0.0, places=9)

    def test_weighted_kmeans_is_reproducible_and_assigns_each_order(self) -> None:
        coordinates = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 10.0], [10.0, 10.0]])
        first_model, first_labels = _fit_weighted_kmeans(coordinates, 2)
        second_model, second_labels = _fit_weighted_kmeans(coordinates, 2)
        np.testing.assert_array_equal(first_labels, second_labels)
        np.testing.assert_allclose(first_model.cluster_centers_, second_model.cluster_centers_)
        self.assertEqual(len(first_labels), len(coordinates))
        self.assertEqual(first_labels[0], first_labels[1])

    def test_cluster_metrics_use_order_outcomes_and_centroids(self) -> None:
        dataset = small_geographic_dataset(30)
        projected = _project_coordinates(
            dataset.orders["customer_latitude"],
            dataset.orders["customer_longitude"],
            origin_latitude=float(dataset.orders["customer_latitude"].mean()),
            origin_longitude=float(dataset.orders["customer_longitude"].mean()),
            reference_latitude=float(dataset.orders["customer_latitude"].mean()),
        )
        from src.analytics.demand_zones import _inverse_project_coordinates

        model, labels = _fit_weighted_kmeans(projected, 2)
        centers = _inverse_project_coordinates(
            model.cluster_centers_,
            origin_latitude=float(dataset.orders["customer_latitude"].mean()),
            origin_longitude=float(dataset.orders["customer_longitude"].mean()),
            reference_latitude=float(dataset.orders["customer_latitude"].mean()),
        )
        orders = dataset.orders.assign(cluster_id=labels)
        summary = _cluster_summary(orders, centers, dataset.restaurants)
        self.assertEqual(int(summary["order_count"].sum()), len(orders))
        self.assertEqual(int(summary["observed_delivery_outcomes"].sum()), len(orders))
        self.assertTrue(summary["late_rate"].between(0, 1).all())
        self.assertTrue(summary["nearest_restaurant_km"].ge(0).all())
        self.assertTrue(summary["mean_distance_from_centroid_km"].ge(0).all())
        network_mean = orders["delivery_minutes"].mean()
        for row in summary.itertuples(index=False):
            cluster_orders = orders.loc[orders["cluster_id"].eq(row.cluster_id)]
            self.assertAlmostEqual(
                row.late_rate,
                cluster_orders["is_late"].mean(),
            )
            self.assertAlmostEqual(
                row.delivery_time_difference_vs_network_minutes,
                row.mean_delivery_minutes - network_mean,
            )
            self.assertAlmostEqual(
                row.centroid_latitude,
                centers[int(row.cluster_id), 0],
            )
            self.assertAlmostEqual(
                row.centroid_longitude,
                centers[int(row.cluster_id), 1],
            )

    def test_fit_persistence_and_coordinate_assignment(self) -> None:
        dataset = small_geographic_dataset()
        report, assignments, _, bundle = fit_demand_zones(
            dataset,
            cluster_counts=(2, 3, 4),
            silhouette_sample_size=40,
        )
        self.assertEqual(len(assignments), len(dataset.orders))
        self.assertEqual(report["metadata"]["selected_cluster_count"], 2)
        self.assertEqual(report["metadata"]["random_seed"], 42)
        self.assertEqual(len(report["clusters"]), 2)
        self.assertAlmostEqual(
            sum(row["demand_share"] for row in report["clusters"]),
            1.0,
        )
        self.assertEqual(
            sorted(row["cluster_count"] for row in report["cluster_count_evaluation"]),
            [2, 3, 4],
        )
        self.assertTrue(report["candidate_zone_ranking"])
        self.assertEqual(
            bundle["metadata"]["feature_names"],
            ["customer_latitude", "customer_longitude"],
        )
        self.assertEqual(len(bundle["centroids_latitude_longitude"]), 2)
        self.assertEqual(len(bundle["cluster_count_evaluation"]), 3)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            processed = root / "processed.csv"
            processed.write_text("test data", encoding="utf-8")
            model_path = root / "model.joblib"
            report_path = root / "report.json"
            assignments_path = root / "assignments.csv"
            restaurants_path = root / "restaurants.csv"
            with patch(
                "src.analytics.demand_zones.load_geographic_dataset",
                return_value=dataset,
            ):
                saved_report = run_demand_zone_analysis(
                    processed,
                    model_path=model_path,
                    report_path=report_path,
                    assignments_path=assignments_path,
                    restaurants_path=restaurants_path,
                    cluster_counts=(2, 3, 4),
                    silhouette_sample_size=40,
                )
            loaded = load_demand_zone_model(model_path)
            self.assertEqual(len(loaded["centroids_latitude_longitude"]), 2)
            self.assertIn("origin_latitude", loaded["preprocessing"])
            self.assertTrue(report_path.is_file())
            self.assertTrue(assignments_path.is_file())
            self.assertTrue(restaurants_path.is_file())
            self.assertEqual(saved_report["metadata"]["selected_cluster_count"], 2)
            self.assertEqual(
                assign_demand_zone(12.96, 77.58, model_path),
                int(bundle["model"].predict(
                    _project_coordinates(
                        [12.96],
                        [77.58],
                        origin_latitude=loaded["projection"]["origin_latitude"],
                        origin_longitude=loaded["projection"]["origin_longitude"],
                        reference_latitude=loaded["projection"]["reference_latitude"],
                    )
                )[0]),
            )

    def test_dashboard_filter_and_location_aggregation(self) -> None:
        assignments = pd.DataFrame(
            {
                "source_row_number": [2, 3, 4],
                "customer_latitude": [13.0, 13.0, 13.1],
                "customer_longitude": [77.0, 77.0, 77.1],
                "cluster_id": [0, 0, 1],
                "is_late": [0, 1, 0],
                "delivery_minutes": [20, 40, 25],
            }
        )
        selected = filter_zone_assignments(assignments, 0)
        self.assertEqual(len(selected), 2)
        self.assertEqual(len(filter_zone_assignments(assignments, None)), 3)
        locations = aggregate_customer_locations(selected)
        self.assertEqual(len(locations), 1)
        self.assertEqual(int(locations.iloc[0]["order_count"]), 2)
        self.assertAlmostEqual(float(locations.iloc[0]["late_rate"]), 0.5)
        with self.assertRaises(ValueError):
            filter_zone_assignments(assignments, 8)


if __name__ == "__main__":
    unittest.main()
