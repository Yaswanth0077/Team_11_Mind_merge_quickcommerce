"""Reproducible customer-demand clustering and dark-store candidate evidence."""

from __future__ import annotations

import argparse
import importlib.metadata
import json
import math
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from config import (
    DARK_STORE_ASSIGNMENTS_PATH,
    DARK_STORE_MODEL_PATH,
    DARK_STORE_RESTAURANTS_PATH,
    DARK_STORE_REPORT_PATH,
    GEO_CLUSTER_COUNTS,
    GEO_CLUSTER_RANDOM_SEED,
    GEO_CLUSTER_SILHOUETTE_SAMPLE_SIZE,
    INDIA_BOUNDS,
    PROCESSED_DATA_PATH,
    SLA_THRESHOLD_MINUTES,
    TARGET_COLUMN,
)
from src.data.feature_engineering import haversine_km
from src.data.feature_engineering import label_late_orders
from src.data.loader import load_orders, sha256_file
from src.analytics.delay_analysis import load_delay_data

EARTH_RADIUS_KM = 6371.0088
ZONE_SCORE_WEIGHTS = {
    "order_demand": 0.35,
    "late_order_rate": 0.25,
    "average_delivery_time": 0.20,
    "geographic_concentration": 0.10,
    "existing_restaurant_gap": 0.10,
}


@dataclass(frozen=True)
class GeographicDataset:
    """Validated customer order locations and separate restaurant locations."""

    orders: pd.DataFrame
    restaurants: pd.DataFrame
    audit: dict[str, Any]


def _as_numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values.astype("string").str.strip(), errors="coerce")


def _coordinate_status(
    latitude: pd.Series, longitude: pd.Series
) -> tuple[pd.Series, dict[str, int]]:
    lat = _as_numeric(latitude)
    lon = _as_numeric(longitude)
    missing = lat.isna() | lon.isna()
    zero = ~missing & lat.eq(0) & lon.eq(0)
    globally_invalid = ~missing & (
        ~lat.between(-90, 90) | ~lon.between(-180, 180)
    )
    outside_region = ~missing & ~zero & ~globally_invalid & (
        ~lat.between(*INDIA_BOUNDS["latitude"])
        | ~lon.between(*INDIA_BOUNDS["longitude"])
    )
    valid = ~missing & ~zero & ~globally_invalid & ~outside_region
    counts = {
        "valid": int(valid.sum()),
        "missing_or_unparseable": int(missing.sum()),
        "zero_coordinate_pair": int(zero.sum()),
        "outside_global_latitude_longitude_ranges": int(globally_invalid.sum()),
        "outside_configured_india_envelope": int(outside_region.sum()),
    }
    return valid, counts


def load_geographic_dataset(path: Path = PROCESSED_DATA_PATH) -> GeographicDataset:
    """Use validated PS1/PS3 loading and retain valid customer and restaurant layers."""
    raw = load_orders(path)
    delay = load_delay_data(path)
    if len(raw) != len(delay):
        raise ValueError("Raw and validated analysis rows are not aligned.")

    customer_valid, customer_counts = _coordinate_status(
        raw["Delivery_location_latitude"],
        raw["Delivery_location_longitude"],
    )
    store_valid, store_counts = _coordinate_status(
        raw["Restaurant_latitude"],
        raw["Restaurant_longitude"],
    )
    target = pd.to_numeric(delay[TARGET_COLUMN], errors="coerce")
    latitude = _as_numeric(raw["Delivery_location_latitude"])
    longitude = _as_numeric(raw["Delivery_location_longitude"])
    distance = pd.to_numeric(delay["analysis_distance_km"], errors="coerce")
    rider = raw["Delivery_person_ID"].astype("string").str.strip()
    rider = rider.mask(rider.str.casefold().isin({"", "nan", "none", "null"}), pd.NA)

    orders = pd.DataFrame(
        {
            "source_row_number": np.arange(len(raw), dtype="int64") + 2,
            "customer_latitude": latitude,
            "customer_longitude": longitude,
            "delivery_minutes": target,
            "is_late": label_late_orders(target).astype("Int8"),
            "delivery_distance_km": distance,
            "rider_id": rider,
        },
        index=raw.index,
    ).loc[customer_valid]

    restaurant_locations = pd.DataFrame(
        {
            "restaurant_latitude": _as_numeric(raw["Restaurant_latitude"]),
            "restaurant_longitude": _as_numeric(raw["Restaurant_longitude"]),
        },
        index=raw.index,
    ).loc[store_valid]
    restaurants = (
        restaurant_locations.groupby(
            ["restaurant_latitude", "restaurant_longitude"], as_index=False
        )
        .size()
        .rename(columns={"size": "orders_at_location"})
    )
    audit = {
        "input_orders": int(len(raw)),
        "customer_coordinate_validation": customer_counts,
        "restaurant_coordinate_validation": store_counts,
        "orders_excluded_from_customer_clustering": int((~customer_valid).sum()),
        "orders_excluded_by_reason": {
            key: value
            for key, value in customer_counts.items()
            if key != "valid"
        },
        "customer_cluster_observations": int(len(orders)),
        "valid_delivery_targets_among_clustered_orders": int(
            orders["delivery_minutes"].notna().sum()
        ),
        "missing_delivery_targets_among_clustered_orders": int(
            orders["delivery_minutes"].isna().sum()
        ),
        "restaurant_orders_with_valid_coordinates": int(store_valid.sum()),
        "unique_valid_restaurant_locations": int(len(restaurants)),
        "rider_identifier_available": int(orders["rider_id"].notna().sum()),
        "limitations": [
            "Each record represents an order; no customer identifier is available, so order volume is not unique-customer count.",
            "Restaurant locations are a service-coverage proxy, not verified dark-store sites.",
            "The configured India coordinate envelope is a screening assumption.",
        ],
    }
    if orders.empty:
        raise ValueError("No customer locations pass the configured geographic validation.")
    return GeographicDataset(
        orders=orders.reset_index(drop=True),
        restaurants=restaurants.reset_index(drop=True),
        audit=audit,
    )


def _project_coordinates(
    latitude: pd.Series | np.ndarray,
    longitude: pd.Series | np.ndarray,
    *,
    origin_latitude: float,
    origin_longitude: float,
    reference_latitude: float,
) -> np.ndarray:
    """Project a city-scale region to local east/north kilometres for K-Means."""
    lat = np.asarray(latitude, dtype="float64")
    lon = np.asarray(longitude, dtype="float64")
    x = (
        EARTH_RADIUS_KM
        * np.radians(lon - origin_longitude)
        * math.cos(math.radians(reference_latitude))
    )
    y = EARTH_RADIUS_KM * np.radians(lat - origin_latitude)
    return np.column_stack((x, y))


def _inverse_project_coordinates(
    points: np.ndarray,
    *,
    origin_latitude: float,
    origin_longitude: float,
    reference_latitude: float,
) -> np.ndarray:
    x, y = points[:, 0], points[:, 1]
    latitude = origin_latitude + np.degrees(y / EARTH_RADIUS_KM)
    longitude = origin_longitude + np.degrees(
        x / (EARTH_RADIUS_KM * math.cos(math.radians(reference_latitude)))
    )
    return np.column_stack((latitude, longitude))


def _fit_weighted_kmeans(
    coordinates_km: np.ndarray,
    cluster_count: int,
    *,
    random_seed: int = GEO_CLUSTER_RANDOM_SEED,
) -> tuple[KMeans, np.ndarray]:
    """Fit on unique coordinate points, weighting each by its observed order count."""
    unique_points, _, counts = np.unique(
        coordinates_km, axis=0, return_inverse=True, return_counts=True
    )
    model = KMeans(
        n_clusters=cluster_count,
        random_state=random_seed,
        n_init=10,
        algorithm="lloyd",
    )
    model.fit(unique_points, sample_weight=counts)
    return model, model.predict(coordinates_km)


def evaluate_cluster_counts(
    coordinates_km: np.ndarray,
    *,
    cluster_counts: tuple[int, ...] = GEO_CLUSTER_COUNTS,
    sample_size: int = GEO_CLUSTER_SILHOUETTE_SAMPLE_SIZE,
    random_seed: int = GEO_CLUSTER_RANDOM_SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Compute weighted inertia and reproducible sampled silhouette by k."""
    if coordinates_km.ndim != 2 or coordinates_km.shape[1] != 2:
        raise ValueError("Projected coordinates must have two columns.")
    if len(coordinates_km) < 3 or sample_size < 2:
        raise ValueError("At least three locations and a silhouette sample of two are required.")
    unique_location_count = len(np.unique(coordinates_km, axis=0))
    if not cluster_counts or any(k < 2 or k >= len(coordinates_km) for k in cluster_counts):
        raise ValueError("Cluster counts must satisfy 2 <= k < observation count.")
    if any(k > unique_location_count for k in cluster_counts):
        raise ValueError("Each cluster count must not exceed the number of unique locations.")
    rng = np.random.default_rng(random_seed)
    sample_count = min(sample_size, len(coordinates_km))
    sample_positions = np.sort(
        rng.choice(len(coordinates_km), size=sample_count, replace=False)
    )
    sample = coordinates_km[sample_positions]
    results: list[dict[str, Any]] = []
    for cluster_count in cluster_counts:
        model, labels = _fit_weighted_kmeans(
            coordinates_km,
            cluster_count,
            random_seed=random_seed,
        )
        sample_labels = labels[sample_positions]
        silhouette = (
            float(silhouette_score(sample, sample_labels, metric="euclidean"))
            if len(np.unique(sample_labels)) > 1
            and len(np.unique(sample_labels)) < len(sample_labels)
            else None
        )
        results.append(
            {
                "cluster_count": int(cluster_count),
                "inertia_km_squared": float(model.inertia_),
                "silhouette_score": silhouette,
                "smallest_cluster_orders": int(np.bincount(labels).min()),
                "largest_cluster_orders": int(np.bincount(labels).max()),
            }
        )
    return pd.DataFrame(results), {
        "sample_size": int(sample_count),
        "sample_strategy": "Uniform without-replacement sample of customer orders; fixed seed.",
        "random_seed": random_seed,
    }


def _nearest_restaurant_distance(
    centroids: np.ndarray, restaurants: pd.DataFrame
) -> np.ndarray:
    if restaurants.empty:
        return np.full(len(centroids), np.nan, dtype="float64")
    nearest = np.full(len(centroids), np.inf, dtype="float64")
    restaurant_latitudes = restaurants["restaurant_latitude"].to_numpy()
    restaurant_longitudes = restaurants["restaurant_longitude"].to_numpy()
    for start in range(0, len(restaurants), 1000):
        end = min(start + 1000, len(restaurants))
        for index, centroid in enumerate(centroids):
            distances = haversine_km(
                np.full(end - start, centroid[0]),
                np.full(end - start, centroid[1]),
                restaurant_latitudes[start:end],
                restaurant_longitudes[start:end],
            )
            nearest[index] = min(nearest[index], float(np.nanmin(distances)))
    return nearest


def _rank_score(values: pd.Series, *, higher_is_better: bool = True) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    ranked = numeric.rank(method="average", pct=True)
    if not higher_is_better:
        ranked = 1.0 - ranked + (1.0 / max(len(ranked), 1))
    return ranked.fillna(0.0).clip(0, 1)


def score_candidate_zones(
    clusters: pd.DataFrame,
    *,
    weights: dict[str, float] = ZONE_SCORE_WEIGHTS,
) -> pd.DataFrame:
    """Rank observed clusters with transparent relative components, not ROI claims."""
    if set(weights) != set(ZONE_SCORE_WEIGHTS):
        raise ValueError("Candidate score weights must define every documented component.")
    if any(not math.isfinite(value) or value < 0 for value in weights.values()):
        raise ValueError("Candidate score weights must be finite and non-negative.")
    weight_total = sum(weights.values())
    if weight_total <= 0:
        raise ValueError("At least one candidate-zone score weight must be positive.")
    result = clusters.copy()
    components = {
        "order_demand": _rank_score(result["order_count"]),
        "late_order_rate": _rank_score(result["late_rate"]),
        "average_delivery_time": _rank_score(result["mean_delivery_minutes"]),
        "geographic_concentration": _rank_score(
            result["mean_distance_from_centroid_km"],
            higher_is_better=False,
        ),
        "existing_restaurant_gap": _rank_score(
            result["nearest_restaurant_km"]
        ),
    }
    for name, scores in components.items():
        result[f"score_{name}"] = scores
    result["candidate_priority_score"] = sum(
        weights[name] * components[name] for name in weights
    ) / weight_total * 100.0
    result["score_method"] = (
        "Relative weighted percentile score; prioritization aid, not site suitability, "
        "financial return, or causal impact."
    )
    return result.sort_values(
        ["candidate_priority_score", "order_count"],
        ascending=False,
        ignore_index=True,
    )


def _cluster_summary(
    orders: pd.DataFrame,
    centroids: np.ndarray,
    restaurants: pd.DataFrame,
) -> pd.DataFrame:
    frame = orders.copy()
    grouped = (
        frame.groupby("cluster_id", observed=True)
        .agg(
            order_count=("source_row_number", "size"),
            observed_delivery_outcomes=("delivery_minutes", "count"),
            mean_delivery_minutes=("delivery_minutes", "mean"),
            median_delivery_minutes=("delivery_minutes", "median"),
            late_orders=("is_late", "sum"),
            late_rate=("is_late", "mean"),
            average_delivery_distance_km=("delivery_distance_km", "mean"),
            unique_riders=("rider_id", "nunique"),
        )
        .reset_index()
    )
    exact_coordinate_counts = (
        frame.drop_duplicates(
            ["cluster_id", "customer_latitude", "customer_longitude"]
        )
        .groupby("cluster_id", observed=True)
        .size()
        .rename("unique_customer_locations")
    )
    grouped = grouped.join(exact_coordinate_counts, on="cluster_id")
    centers = centroids[grouped["cluster_id"].to_numpy(dtype="int64")]
    grouped["centroid_latitude"] = centers[:, 0]
    grouped["centroid_longitude"] = centers[:, 1]
    order_centers = centroids[frame["cluster_id"].to_numpy(dtype="int64")]
    distances = haversine_km(
        frame["customer_latitude"].to_numpy(),
        frame["customer_longitude"].to_numpy(),
        order_centers[:, 0],
        order_centers[:, 1],
    )
    frame["distance_to_centroid_km"] = distances
    compactness = frame.groupby("cluster_id", observed=True)[
        "distance_to_centroid_km"
    ].mean()
    grouped["mean_distance_from_centroid_km"] = grouped["cluster_id"].map(compactness)
    grouped["demand_share"] = grouped["order_count"] / len(frame)
    grouped["nearest_restaurant_km"] = _nearest_restaurant_distance(
        centers, restaurants
    )
    network_delivery_mean = float(frame["delivery_minutes"].mean())
    network_late_rate = float(frame["is_late"].mean())
    network_distance_mean = float(frame["delivery_distance_km"].mean())
    grouped["delivery_time_difference_vs_network_minutes"] = (
        grouped["mean_delivery_minutes"] - network_delivery_mean
    )
    grouped["late_rate_difference_vs_network_percentage_points"] = (
        grouped["late_rate"] - network_late_rate
    ) * 100.0
    grouped["distance_difference_vs_network_km"] = (
        grouped["average_delivery_distance_km"] - network_distance_mean
    )
    return grouped


def _elbow_suggestion(evaluations: pd.DataFrame) -> int:
    """Find the largest normalized second-difference bend in the inertia curve."""
    x = evaluations["cluster_count"].to_numpy(dtype="float64")
    y = evaluations["inertia_km_squared"].to_numpy(dtype="float64")
    if len(x) < 3:
        return int(x[np.argmax(evaluations["silhouette_score"].fillna(-np.inf))])
    normalized_x = (x - x.min()) / max(x.max() - x.min(), 1.0)
    normalized_y = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    curvature = np.diff(normalized_y, n=2)
    return int(x[1:-1][np.argmax(curvature)])


def fit_demand_zones(
    dataset: GeographicDataset,
    *,
    cluster_counts: tuple[int, ...] = GEO_CLUSTER_COUNTS,
    silhouette_sample_size: int = GEO_CLUSTER_SILHOUETTE_SAMPLE_SIZE,
    random_seed: int = GEO_CLUSTER_RANDOM_SEED,
    selected_cluster_count: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Evaluate k, select from silhouette/elbow evidence, and summarize zones."""
    orders = dataset.orders.copy().reset_index(drop=True)
    origin_latitude = float(orders["customer_latitude"].mean())
    origin_longitude = float(orders["customer_longitude"].mean())
    reference_latitude = origin_latitude
    coordinates_km = _project_coordinates(
        orders["customer_latitude"],
        orders["customer_longitude"],
        origin_latitude=origin_latitude,
        origin_longitude=origin_longitude,
        reference_latitude=reference_latitude,
    )
    evaluation, sample_metadata = evaluate_cluster_counts(
        coordinates_km,
        cluster_counts=cluster_counts,
        sample_size=silhouette_sample_size,
        random_seed=random_seed,
    )
    elbow_k = _elbow_suggestion(evaluation)
    silhouette_row = evaluation.loc[evaluation["silhouette_score"].notna()].sort_values(
        ["silhouette_score", "cluster_count"], ascending=[False, True]
    )
    if silhouette_row.empty:
        raise ValueError("No valid silhouette scores could be calculated.")
    silhouette_k = int(silhouette_row.iloc[0]["cluster_count"])
    if selected_cluster_count is None:
        selected_cluster_count = silhouette_k
        selection_method = (
            "Selected the candidate with the highest reproducible sampled silhouette score. "
            "The inertia elbow is an independent, coarser diagnostic; compare the saved "
            "centroids, sizes, and compactness before treating this exploratory k as useful."
        )
    else:
        if selected_cluster_count not in cluster_counts:
            raise ValueError("selected_cluster_count must be one of the evaluated candidates.")
        selection_method = (
            "Explicit analyst override from the evaluated candidates; review the saved "
            "silhouette, elbow, cluster sizes, and map before use."
        )

    model, labels = _fit_weighted_kmeans(
        coordinates_km,
        selected_cluster_count,
        random_seed=random_seed,
    )
    orders["cluster_id"] = labels.astype("int64")
    centroid_coordinates = _inverse_project_coordinates(
        model.cluster_centers_,
        origin_latitude=origin_latitude,
        origin_longitude=origin_longitude,
        reference_latitude=reference_latitude,
    )
    clusters = _cluster_summary(orders, centroid_coordinates, dataset.restaurants)
    candidate_ranking = score_candidate_zones(clusters)
    chosen_silhouette = evaluation.loc[
        evaluation["cluster_count"].eq(selected_cluster_count),
        "silhouette_score",
    ].iloc[0]
    report = {
        "metadata": {
            "model_version": "1.0.0",
            "algorithm": "KMeans",
            "selected_cluster_count": int(selected_cluster_count),
            "random_seed": int(random_seed),
            "coordinate_projection": (
                "Local equirectangular east/north kilometres centered at the mean customer "
                "coordinate. Appropriate for this geographically compact dataset; not a "
                "general global projection."
            ),
            "demand_weighting": (
                "Exact duplicate customer coordinate points are grouped and K-Means "
                "sample_weight is set to the observed order count. No rows are duplicated."
            ),
            "target_leakage_policy": (
                "Only customer latitude/longitude and order-frequency weights fit K-Means. "
                "Delivery time and SLA labels are joined after cluster assignment for "
                "descriptive cluster performance."
            ),
            "selection_method": selection_method,
            "silhouette_sample": sample_metadata,
            "selected_silhouette_score": (
                float(chosen_silhouette) if pd.notna(chosen_silhouette) else None
            ),
            "inertia_elbow_suggestion": elbow_k,
            "silhouette_selection_at_candidate_boundary": (
                selected_cluster_count
                == int(evaluation["cluster_count"].max())
                and selected_cluster_count == silhouette_k
            ),
            "customer_coordinate_envelope": INDIA_BOUNDS,
            "feature_names": ["customer_latitude", "customer_longitude"],
            "sla_threshold_minutes": SLA_THRESHOLD_MINUTES,
            "sla_status": "provisional",
            "score_components_and_weights": ZONE_SCORE_WEIGHTS,
            "score_interpretation": (
                "Relative cluster ranking calculated from component percentile ranks. "
                "Not a probability, causal benefit, economic estimate, or site recommendation."
            ),
        },
        "data_audit": dataset.audit,
        "cluster_count_evaluation": _records(evaluation),
        "network_benchmarks": {
            "clustered_orders": int(len(orders)),
            "mean_delivery_minutes": float(orders["delivery_minutes"].mean()),
            "median_delivery_minutes": float(orders["delivery_minutes"].median()),
            "late_rate": float(orders["is_late"].mean()),
            "average_delivery_distance_km": float(
                orders["delivery_distance_km"].mean()
            ),
            "unique_riders": int(orders["rider_id"].nunique()),
        },
        "clusters": _records(clusters),
        "candidate_zone_ranking": _records(candidate_ranking),
        "limitations": [
            "The source has delivery orders, not distinct customer IDs; order count is demand proxy, not unique demanding customers.",
            "Clusters describe customer order density, not an optimal dark-store location.",
            "Restaurants are an imperfect proxy for existing operational coverage; dark-store footprints are not supplied.",
            "Candidate ranking omits property availability, rent, inventory, road access, staffing, capacity, and operational feasibility.",
            "Straight-line geographic density and delivery outcomes do not establish causal benefit from opening a site.",
        ],
    }
    bundle = {
        "model": model,
        "metadata": report["metadata"],
        "cluster_count_evaluation": report["cluster_count_evaluation"],
        "centroids_latitude_longitude": centroid_coordinates.tolist(),
        "preprocessing": {
            "method": "local equirectangular east/north kilometer projection",
            "origin_latitude": origin_latitude,
            "origin_longitude": origin_longitude,
            "reference_latitude": reference_latitude,
        },
        "projection": {
            "origin_latitude": origin_latitude,
            "origin_longitude": origin_longitude,
            "reference_latitude": reference_latitude,
        },
    }
    return report, orders, dataset.restaurants, bundle


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        records.append(
            {
                key: (
                    None
                    if isinstance(value, (np.generic, float))
                    and not math.isfinite(float(value))
                    else value.item()
                    if isinstance(value, np.generic)
                    else value.isoformat()
                    if isinstance(value, (pd.Timestamp, datetime))
                    else None
                    if pd.isna(value)
                    else value
                )
                for key, value in row.items()
            }
        )
    return records


def _save_report(report: dict[str, Any], path: Path) -> None:
    def safe(value: Any) -> Any:
        if isinstance(value, dict):
            return {str(key): safe(item) for key, item in value.items()}
        if isinstance(value, (list, tuple)):
            return [safe(item) for item in value]
        if isinstance(value, np.generic):
            return safe(value.item())
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(safe(report), indent=2, allow_nan=False),
        encoding="utf-8",
    )


def run_demand_zone_analysis(
    path: Path = PROCESSED_DATA_PATH,
    *,
    model_path: Path = DARK_STORE_MODEL_PATH,
    report_path: Path = DARK_STORE_REPORT_PATH,
    assignments_path: Path = DARK_STORE_ASSIGNMENTS_PATH,
    restaurants_path: Path = DARK_STORE_RESTAURANTS_PATH,
    cluster_counts: tuple[int, ...] = GEO_CLUSTER_COUNTS,
    silhouette_sample_size: int = GEO_CLUSTER_SILHOUETTE_SAMPLE_SIZE,
    selected_cluster_count: int | None = None,
) -> dict[str, Any]:
    """Load, fit, save, and return the reproducible demand-zone experiment."""
    dataset = load_geographic_dataset(path)
    report, assignments, restaurants, bundle = fit_demand_zones(
        dataset,
        cluster_counts=cluster_counts,
        silhouette_sample_size=silhouette_sample_size,
        selected_cluster_count=selected_cluster_count,
    )
    report["metadata"]["processed_dataset_path"] = str(path)
    report["metadata"]["processed_dataset_sha256"] = sha256_file(path)
    report["metadata"]["package_versions"] = {
        package: importlib.metadata.version(package)
        for package in ("scikit-learn", "joblib", "pandas", "numpy")
    }
    report["metadata"]["trained_at_utc"] = datetime.now(timezone.utc).isoformat()
    report["artifacts"] = {
        "model": str(model_path),
        "report": str(report_path),
        "assignments": str(assignments_path),
        "restaurants": str(restaurants_path),
    }
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(assignments_path, index=False)
    restaurants_path.parent.mkdir(parents=True, exist_ok=True)
    restaurants.to_csv(restaurants_path, index=False)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    _save_report(report, report_path)
    return report


def load_demand_zone_model(path: Path = DARK_STORE_MODEL_PATH) -> dict[str, Any]:
    """Load saved K-Means model and local projection metadata."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Demand-zone model not found at {path}. Run `python -m src.analytics.demand_zones`."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not {"model", "metadata", "projection"}.issubset(bundle):
        raise ValueError(f"Demand-zone model at {path} has an unsupported structure.")
    return bundle


def assign_demand_zone(
    latitude: float,
    longitude: float,
    model_path: Path = DARK_STORE_MODEL_PATH,
) -> int:
    """Assign one valid customer coordinate to a saved exploratory zone."""
    if (
        not math.isfinite(latitude)
        or not math.isfinite(longitude)
        or not INDIA_BOUNDS["latitude"][0] <= latitude <= INDIA_BOUNDS["latitude"][1]
        or not INDIA_BOUNDS["longitude"][0] <= longitude <= INDIA_BOUNDS["longitude"][1]
        or (latitude == 0 and longitude == 0)
    ):
        raise ValueError("Provide a valid non-zero coordinate inside the configured India envelope.")
    bundle = load_demand_zone_model(model_path)
    projection = bundle["projection"]
    point = _project_coordinates(
        [latitude],
        [longitude],
        origin_latitude=projection["origin_latitude"],
        origin_longitude=projection["origin_longitude"],
        reference_latitude=projection["reference_latitude"],
    )
    return int(bundle["model"].predict(point)[0])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate K-Means customer-demand clusters and candidate zones."
    )
    parser.add_argument("--input", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--clusters", type=int, default=None)
    args = parser.parse_args()
    report = run_demand_zone_analysis(args.input, selected_cluster_count=args.clusters)
    print(
        f"Valid customer orders: "
        f"{report['data_audit']['customer_coordinate_validation']['valid']:,}/"
        f"{report['data_audit']['input_orders']:,}"
    )
    print(
        f"Selected k={report['metadata']['selected_cluster_count']} "
        f"(elbow suggestion k={report['metadata']['inertia_elbow_suggestion']})"
    )
    for zone in report["candidate_zone_ranking"]:
        print(
            f"Zone {zone['cluster_id']}: orders={zone['order_count']:,}, "
            f"late rate={zone['late_rate']:.1%}, "
            f"priority={zone['candidate_priority_score']:.1f}"
        )
    print(f"Report: {DARK_STORE_REPORT_PATH}")
    print(f"Model: {DARK_STORE_MODEL_PATH}")


if __name__ == "__main__":
    main()
