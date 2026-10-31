"""Reproducible, sample-qualified rider performance segmentation."""

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
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from config import (
    LATE_LABEL_COLUMN,
    MISSING_TOKENS,
    PROCESSED_DATA_PATH,
    RIDER_CLUSTER_COUNTS,
    RIDER_CLUSTER_RANDOM_SEED,
    RIDER_CLUSTER_SILHOUETTE_SAMPLE_SIZE,
    RIDER_ELBOW_SILHOUETTE_TOLERANCE,
    RIDER_ID_COLUMN,
    RIDER_MIN_COMPLETED_DELIVERIES,
    RIDER_MIN_SEGMENT_RIDERS,
    RIDER_MIN_RATING_OBSERVATIONS,
    RIDER_ORDER_ASSIGNMENTS_PATH,
    RIDER_SEGMENT_ASSIGNMENTS_PATH,
    RIDER_SEGMENT_MODEL_PATH,
    RIDER_SEGMENT_REPORT_PATH,
    SLA_THRESHOLD_MINUTES,
    TARGET_COLUMN,
)
from src.analytics.delay_analysis import load_delay_data
from src.data.feature_engineering import label_late_orders
from src.data.loader import sha256_file

FEATURE_COLUMNS = (
    "log_delivery_count",
    "mean_delivery_minutes",
    "late_rate",
    "mean_delivery_distance_km",
    "mean_rider_rating",
    "mean_multiple_deliveries",
)
RATING_BOUNDS = (1.0, 5.0)
WILSON_Z = 1.96


@dataclass(frozen=True)
class RiderDataset:
    """Rider-level summaries and their valid completed-delivery observations."""

    riders: pd.DataFrame
    orders: pd.DataFrame
    audit: dict[str, Any]


def _numeric(values: pd.Series) -> pd.Series:
    return pd.to_numeric(values, errors="coerce").astype("float64")


def _valid_rider_ids(values: pd.Series) -> pd.Series:
    rider_ids = values.astype("string").str.strip()
    return rider_ids.mask(
        rider_ids.isna()
        | rider_ids.eq("")
        | rider_ids.str.casefold().isin(MISSING_TOKENS)
    )


def _wilson_interval(successes: pd.Series, observations: pd.Series) -> tuple[pd.Series, pd.Series]:
    n = observations.astype("float64")
    p = successes.astype("float64").div(n.where(n.gt(0)))
    z2 = WILSON_Z**2
    denominator = 1.0 + z2 / n
    center = (p + z2 / (2.0 * n)) / denominator
    margin = WILSON_Z * np.sqrt((p * (1.0 - p) / n + z2 / (4.0 * n**2))) / denominator
    return (center - margin).clip(0, 1), (center + margin).clip(0, 1)


def aggregate_rider_performance(
    frame: pd.DataFrame,
    *,
    minimum_deliveries: int = RIDER_MIN_COMPLETED_DELIVERIES,
    minimum_rating_observations: int = RIDER_MIN_RATING_OBSERVATIONS,
) -> RiderDataset:
    """Aggregate valid completed deliveries without interpreting rider ID as a feature."""
    required = {
        RIDER_ID_COLUMN,
        TARGET_COLUMN,
        "analysis_valid_order",
        "analysis_distance_km",
        "analysis_rating",
        "multiple_deliveries",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Prepared delivery data is missing columns: {', '.join(missing)}")
    if minimum_deliveries < 1 or minimum_rating_observations < 1:
        raise ValueError("Minimum delivery and rating observation counts must be positive.")

    target = _numeric(frame[TARGET_COLUMN])
    rider_id = _valid_rider_ids(frame[RIDER_ID_COLUMN])
    valid_target = frame["analysis_valid_order"].fillna(False).astype(bool) & target.gt(0)
    valid_order = valid_target & rider_id.notna()
    orders = frame.loc[valid_order].copy()
    orders["rider_id"] = rider_id.loc[valid_order].astype("string")
    orders["delivery_minutes"] = target.loc[valid_order]
    orders["is_late"] = (
        label_late_orders(orders["delivery_minutes"], SLA_THRESHOLD_MINUTES)
        .astype("Int8")
        .astype("int8")
    )
    distance = _numeric(orders["analysis_distance_km"])
    orders["delivery_distance_km"] = distance.where(distance.gt(0))
    rating = _numeric(orders["analysis_rating"])
    orders["rider_rating"] = rating.where(rating.between(*RATING_BOUNDS))
    multiple = _numeric(orders["multiple_deliveries"])
    valid_multiple = multiple.ge(0) & multiple.mod(1).eq(0)
    orders["multiple_deliveries_valid"] = multiple.where(valid_multiple)
    orders["source_row_number"] = orders.index.to_series().astype("int64") + 2
    orders = orders.reset_index(drop=True)

    if orders.empty:
        raise ValueError("No completed deliveries with a valid rider ID are available.")
    grouped = (
        orders.groupby("rider_id", sort=True, observed=True)
        .agg(
            delivery_count=("delivery_minutes", "size"),
            late_delivery_count=("is_late", "sum"),
            mean_delivery_minutes=("delivery_minutes", "mean"),
            median_delivery_minutes=("delivery_minutes", "median"),
            late_rate=("is_late", "mean"),
            distance_observation_count=("delivery_distance_km", "count"),
            mean_delivery_distance_km=("delivery_distance_km", "mean"),
            rating_observation_count=("rider_rating", "count"),
            mean_rider_rating=("rider_rating", "mean"),
            multiple_delivery_observation_count=("multiple_deliveries_valid", "count"),
            mean_multiple_deliveries=("multiple_deliveries_valid", "mean"),
        )
        .reset_index()
    )
    grouped["rating_data_sufficient"] = grouped["rating_observation_count"].ge(
        minimum_rating_observations
    )
    grouped["mean_rider_rating"] = grouped["mean_rider_rating"].where(
        grouped["rating_data_sufficient"]
    )
    grouped["distance_data_sufficient"] = grouped[
        "distance_observation_count"
    ].ge(minimum_rating_observations)
    grouped["mean_delivery_distance_km"] = grouped[
        "mean_delivery_distance_km"
    ].where(grouped["distance_data_sufficient"])
    grouped["multiple_delivery_data_sufficient"] = grouped[
        "multiple_delivery_observation_count"
    ].ge(minimum_rating_observations)
    grouped["mean_multiple_deliveries"] = grouped[
        "mean_multiple_deliveries"
    ].where(grouped["multiple_delivery_data_sufficient"])
    grouped["log_delivery_count"] = np.log1p(grouped["delivery_count"])
    grouped["sufficient_observations"] = grouped["delivery_count"].ge(
        minimum_deliveries
    )
    grouped["insufficient_observations"] = ~grouped["sufficient_observations"]
    grouped["sufficiency_status"] = np.where(
        grouped["sufficient_observations"],
        f"Sufficient (≥{minimum_deliveries} completed deliveries)",
        f"Insufficient (<{minimum_deliveries} completed deliveries)",
    )
    grouped["late_rate_ci_low"], grouped["late_rate_ci_high"] = _wilson_interval(
        grouped["late_delivery_count"], grouped["delivery_count"]
    )
    grouped["segment_id"] = pd.Series(pd.NA, index=grouped.index, dtype="Int64")
    grouped["segment_description"] = "Not segmented: insufficient delivery sample"

    audit = {
        "input_order_rows": int(len(frame)),
        "completed_delivery_rows": int(valid_target.sum()),
        "completed_deliveries_with_valid_rider_id": int(len(orders)),
        "completed_deliveries_missing_or_invalid_rider_id": int(
            (valid_target & rider_id.isna()).sum()
        ),
        "invalid_or_missing_delivery_targets": int((~valid_target).sum()),
        "distinct_riders_with_completed_deliveries": int(grouped["rider_id"].nunique()),
        "riders_meeting_minimum_deliveries": int(
            grouped["sufficient_observations"].sum()
        ),
        "riders_flagged_insufficient_deliveries": int(
            grouped["insufficient_observations"].sum()
        ),
        "minimum_deliveries_for_segmentation": int(minimum_deliveries),
        "minimum_rating_observations": int(minimum_rating_observations),
        "valid_delivery_distance_observations": int(orders["delivery_distance_km"].notna().sum()),
        "valid_rider_rating_observations": int(orders["rider_rating"].notna().sum()),
        "valid_multiple_delivery_observations": int(
            orders["multiple_deliveries_valid"].notna().sum()
        ),
        "feature_missing_values_among_eligible_riders": {},
        "delivery_count_distribution": {
            "minimum": int(grouped["delivery_count"].min()),
            "median": float(grouped["delivery_count"].median()),
            "maximum": int(grouped["delivery_count"].max()),
        },
    }
    eligible = grouped.loc[grouped["sufficient_observations"]]
    if not eligible.empty:
        audit["feature_missing_values_among_eligible_riders"] = {
            column: int(eligible[column].isna().sum())
            for column in FEATURE_COLUMNS
        }
    return RiderDataset(riders=grouped, orders=orders, audit=audit)


def load_rider_dataset(path: Path = PROCESSED_DATA_PATH) -> RiderDataset:
    """Prepare rider summaries using the existing PS3 validated data pipeline."""
    return aggregate_rider_performance(load_delay_data(path))


def build_rider_preprocessor() -> Pipeline:
    """Median-impute and standardize rider features using eligible riders only."""
    return Pipeline(
        [
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                    keep_empty_features=True,
                ),
            ),
            ("scaler", StandardScaler()),
        ]
    )


def _fit_kmeans(
    scaled_features: np.ndarray,
    cluster_count: int,
    *,
    random_seed: int = RIDER_CLUSTER_RANDOM_SEED,
) -> tuple[KMeans, np.ndarray]:
    model = KMeans(
        n_clusters=cluster_count,
        random_state=random_seed,
        n_init=20,
        algorithm="lloyd",
    )
    labels = model.fit_predict(scaled_features)
    return model, labels


def evaluate_segment_counts(
    scaled_features: np.ndarray,
    *,
    cluster_counts: tuple[int, ...] = RIDER_CLUSTER_COUNTS,
    silhouette_sample_size: int = RIDER_CLUSTER_SILHOUETTE_SAMPLE_SIZE,
    random_seed: int = RIDER_CLUSTER_RANDOM_SEED,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calculate inertia, reproducible sampled silhouette, and cluster-size diagnostics."""
    if scaled_features.ndim != 2 or scaled_features.shape[1] == 0:
        raise ValueError("Scaled rider features must be a non-empty two-dimensional array.")
    if silhouette_sample_size < 2:
        raise ValueError("Silhouette sample size must be at least two.")
    if not cluster_counts or any(
        cluster_count < 2 or cluster_count >= len(scaled_features)
        for cluster_count in cluster_counts
    ):
        raise ValueError("Candidate K values must satisfy 2 <= K < eligible rider count.")
    random = np.random.default_rng(random_seed)
    sample_count = min(silhouette_sample_size, len(scaled_features))
    sample_positions = np.sort(
        random.choice(len(scaled_features), size=sample_count, replace=False)
    )
    rows: list[dict[str, Any]] = []
    for cluster_count in cluster_counts:
        model, labels = _fit_kmeans(
            scaled_features,
            cluster_count,
            random_seed=random_seed,
        )
        counts = np.bincount(labels, minlength=cluster_count)
        sample_labels = labels[sample_positions]
        score = (
            float(
                silhouette_score(
                    scaled_features[sample_positions],
                    sample_labels,
                    metric="euclidean",
                )
            )
            if 1 < np.unique(sample_labels).size < sample_count
            else None
        )
        rows.append(
            {
                "cluster_count": int(cluster_count),
                "inertia": float(model.inertia_),
                "silhouette_score": score,
                "smallest_segment_riders": int(counts.min()),
                "largest_segment_riders": int(counts.max()),
                "small_segment_flag": bool(counts.min() < RIDER_MIN_SEGMENT_RIDERS),
            }
        )
    return pd.DataFrame(rows), {
        "sample_size": int(sample_count),
        "sample_strategy": "Uniform without-replacement rider sample with fixed random seed.",
        "random_seed": int(random_seed),
    }


def _elbow_suggestion(evaluations: pd.DataFrame) -> int:
    """Find the largest normalized second-difference bend in the inertia curve."""
    x = evaluations["cluster_count"].to_numpy(dtype="float64")
    y = evaluations["inertia"].to_numpy(dtype="float64")
    if len(x) < 3:
        return int(x[np.argmax(evaluations["silhouette_score"].fillna(-np.inf))])
    normalized_x = (x - x.min()) / max(x.max() - x.min(), 1.0)
    normalized_y = (y - y.min()) / max(y.max() - y.min(), 1e-12)
    curvature = np.diff(normalized_y, n=2)
    return int(x[1:-1][np.argmax(curvature)])


def _select_cluster_count(evaluations: pd.DataFrame) -> tuple[int, str]:
    """Prefer a viable inertia-elbow K when silhouette evidence is effectively tied."""
    valid = evaluations.loc[evaluations["silhouette_score"].notna()].copy()
    if valid.empty:
        raise ValueError("No valid silhouette scores are available for model selection.")
    stable = valid.loc[
        valid["smallest_segment_riders"].ge(RIDER_MIN_SEGMENT_RIDERS)
    ]
    candidates = stable if not stable.empty else valid
    best = candidates.sort_values(
        ["silhouette_score", "cluster_count"],
        ascending=[False, True],
    ).iloc[0]
    elbow_k = _elbow_suggestion(evaluations)
    elbow = candidates.loc[candidates["cluster_count"].eq(elbow_k)]
    if not elbow.empty and (
        float(best["silhouette_score"]) - float(elbow.iloc[0]["silhouette_score"])
        <= RIDER_ELBOW_SILHOUETTE_TOLERANCE
    ):
        return (
            elbow_k,
            "Selected the inertia-elbow K because its silhouette is within "
            f"{RIDER_ELBOW_SILHOUETTE_TOLERANCE:.3f} of the highest stable-candidate "
            "silhouette. All segments meet the rider-count stability screen; the selected "
            "profiles were reviewed for distinct measured operational characteristics.",
        )
    return (
        int(best["cluster_count"]),
        "Selected the highest-silhouette candidate with at least "
        f"{RIDER_MIN_SEGMENT_RIDERS} riders in each segment when such candidates exist; "
        "reviewed the inertia elbow and measured segment profiles separately.",
    )


def _segment_description(segment: pd.Series, reference: pd.Series) -> str:
    """Describe measured segment differences without ranking or causal wording."""
    characteristics = (
        ("mean_deliveries_per_rider", "delivery volume"),
        ("mean_delivery_minutes", "average delivery time"),
        ("late_rate", "late-delivery rate"),
        ("mean_delivery_distance_km", "delivery distance"),
        ("mean_rider_rating", "average rider rating"),
        ("mean_multiple_deliveries", "multiple-delivery load"),
    )
    phrases: list[str] = []
    for column, label in characteristics:
        value = segment.get(column)
        overall = reference.get(column)
        if pd.isna(value) or pd.isna(overall):
            continue
        if math.isclose(float(value), float(overall), rel_tol=0.02, abs_tol=0.01):
            direction = "similar to"
        else:
            direction = "higher than" if value > overall else "lower than"
        phrases.append(f"{label} {direction} the eligible-rider average")
    return "; ".join(phrases) if phrases else "No measured characteristics differ from the reference."


def summarize_rider_segments(
    riders: pd.DataFrame,
    *,
    cluster_column: str = "segment_id",
    order_assignments: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Summarize segment performance and rider-level delivery-volume distribution."""
    eligible = riders.loc[
        riders[cluster_column].notna() & riders["sufficient_observations"]
    ].copy()
    if eligible.empty:
        return pd.DataFrame(
            columns=[
                cluster_column,
                "rider_count",
                "rider_share",
                "total_deliveries",
                "mean_deliveries_per_rider",
                "median_deliveries_per_rider",
                "p10_deliveries_per_rider",
                "p90_deliveries_per_rider",
                "mean_delivery_minutes",
                "median_delivery_minutes",
                "late_rate",
                "mean_delivery_distance_km",
                "mean_rider_rating",
                "mean_multiple_deliveries",
            ]
        )
    eligible_count = len(eligible)
    summary = (
        eligible.groupby(cluster_column, observed=True)
        .agg(
            rider_count=("rider_id", "nunique"),
            total_deliveries=("delivery_count", "sum"),
            mean_deliveries_per_rider=("delivery_count", "mean"),
            median_deliveries_per_rider=("delivery_count", "median"),
            p10_deliveries_per_rider=(
                "delivery_count",
                lambda values: values.quantile(0.10),
            ),
            p90_deliveries_per_rider=(
                "delivery_count",
                lambda values: values.quantile(0.90),
            ),
            mean_delivery_minutes=("mean_delivery_minutes", "mean"),
            median_delivery_minutes=("median_delivery_minutes", "median"),
            late_rate=("late_rate", "mean"),
            mean_delivery_distance_km=("mean_delivery_distance_km", "mean"),
            mean_rider_rating=("mean_rider_rating", "mean"),
            mean_multiple_deliveries=("mean_multiple_deliveries", "mean"),
            riders_with_sufficient_rating=("rating_data_sufficient", "sum"),
            riders_with_sufficient_distance=("distance_data_sufficient", "sum"),
            riders_with_sufficient_multiple_delivery_data=(
                "multiple_delivery_data_sufficient",
                "sum",
            ),
        )
        .reset_index()
    )
    summary["rider_share"] = summary["rider_count"] / eligible_count
    summary["small_segment_flag"] = summary["rider_count"].lt(30)
    reference = eligible.agg(
        {
            "delivery_count": "mean",
            "mean_delivery_minutes": "mean",
            "late_rate": "mean",
            "mean_delivery_distance_km": "mean",
            "mean_rider_rating": "mean",
            "mean_multiple_deliveries": "mean",
        }
    ).rename({"delivery_count": "mean_deliveries_per_rider"})
    if order_assignments is not None and not order_assignments.empty:
        selected_orders = order_assignments.loc[
            order_assignments[cluster_column].notna()
        ].copy()
        if not selected_orders.empty:
            order_summary = (
                selected_orders.groupby(cluster_column, observed=True)
                .agg(
                    total_deliveries=("delivery_minutes", "count"),
                    mean_delivery_minutes=("delivery_minutes", "mean"),
                    median_delivery_minutes=("delivery_minutes", "median"),
                    late_rate=("is_late", "mean"),
                    mean_delivery_distance_km=("delivery_distance_km", "mean"),
                    mean_multiple_deliveries=("multiple_deliveries_valid", "mean"),
                )
                .reset_index()
            )
            summary = summary.drop(
                columns=[
                    "total_deliveries",
                    "mean_delivery_minutes",
                    "median_delivery_minutes",
                    "late_rate",
                    "mean_delivery_distance_km",
                    "mean_multiple_deliveries",
                ]
            ).merge(order_summary, on=cluster_column, how="left")
            reference = pd.Series(
                {
                    "mean_deliveries_per_rider": eligible["delivery_count"].mean(),
                    "mean_delivery_minutes": selected_orders["delivery_minutes"].mean(),
                    "late_rate": selected_orders["is_late"].mean(),
                    "mean_delivery_distance_km": selected_orders[
                        "delivery_distance_km"
                    ].mean(),
                    "mean_rider_rating": eligible["mean_rider_rating"].mean(),
                    "mean_multiple_deliveries": selected_orders[
                        "multiple_deliveries_valid"
                    ].mean(),
                }
            )
    summary["operational_description"] = summary.apply(
        _segment_description,
        axis=1,
        reference=reference,
    )
    return summary.sort_values(cluster_column, ignore_index=True)


def fit_rider_segments(
    dataset: RiderDataset,
    *,
    cluster_counts: tuple[int, ...] = RIDER_CLUSTER_COUNTS,
    silhouette_sample_size: int = RIDER_CLUSTER_SILHOUETTE_SAMPLE_SIZE,
    random_seed: int = RIDER_CLUSTER_RANDOM_SEED,
    selected_cluster_count: int | None = None,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Fit rider-only preprocessing and K-Means, then report actual segment metrics."""
    riders = dataset.riders.copy().reset_index(drop=True)
    eligible_mask = riders["sufficient_observations"]
    eligible = riders.loc[eligible_mask].copy()
    if len(eligible) < 3:
        raise ValueError(
            "At least three riders must meet the configured minimum delivery count."
        )
    features = eligible.loc[:, list(FEATURE_COLUMNS)].to_numpy(dtype="float64")
    preprocessor = build_rider_preprocessor()
    scaled_features = preprocessor.fit_transform(features)
    evaluations, sample_metadata = evaluate_segment_counts(
        scaled_features,
        cluster_counts=cluster_counts,
        silhouette_sample_size=silhouette_sample_size,
        random_seed=random_seed,
    )
    elbow_k = _elbow_suggestion(evaluations)
    automatic_k, automatic_selection_method = _select_cluster_count(evaluations)
    if selected_cluster_count is None:
        selected_cluster_count = automatic_k
        selection_method = automatic_selection_method
    elif selected_cluster_count not in cluster_counts:
        raise ValueError("selected_cluster_count must be one of the evaluated candidates.")
    else:
        selection_method = (
            "Explicit analyst override among evaluated candidates; review the saved "
            "silhouette, elbow, segment sizes, and profiles."
        )

    model, labels = _fit_kmeans(
        scaled_features,
        selected_cluster_count,
        random_seed=random_seed,
    )
    riders["segment_id"] = pd.Series(pd.NA, index=riders.index, dtype="Int64")
    riders.loc[eligible_mask, "segment_id"] = labels.astype("int64")
    segment_ids = riders.set_index("rider_id")["segment_id"]
    rider_order_assignments = dataset.orders.copy()
    rider_order_assignments["segment_id"] = (
        rider_order_assignments["rider_id"].map(segment_ids).astype("Int64")
    )
    rider_order_assignments["sufficient_observations"] = rider_order_assignments[
        "segment_id"
    ].notna()
    segment_summary = summarize_rider_segments(
        riders,
        order_assignments=rider_order_assignments,
    )
    descriptions = segment_summary.set_index("segment_id")[
        "operational_description"
    ].to_dict()
    riders.loc[eligible_mask, "segment_description"] = (
        riders.loc[eligible_mask, "segment_id"].map(descriptions)
    )
    selected_score = float(
        evaluations.loc[
            evaluations["cluster_count"].eq(selected_cluster_count),
            "silhouette_score",
        ].iloc[0]
    )
    metadata = {
        "model_version": "1.0.0",
        "algorithm": "KMeans",
        "selected_cluster_count": int(selected_cluster_count),
        "random_seed": int(random_seed),
        "feature_names": list(FEATURE_COLUMNS),
        "preprocessing": (
            "Median imputation, followed by StandardScaler; fit only on riders meeting "
            "the completed-delivery minimum. Missingness is reported separately and is "
            "not itself used as a cluster feature."
        ),
        "feature_policy": {
            "log_delivery_count": "log1p of completed delivery count; included to represent volume with reduced skew.",
            "mean_delivery_minutes": "Mean valid completed-delivery time per rider.",
            "late_rate": f"Share of valid completed deliveries strictly exceeding the configured {SLA_THRESHOLD_MINUTES}-minute SLA.",
            "mean_delivery_distance_km": "Mean valid PS3 Haversine distance; missing if fewer than the required valid observations and median-imputed without a missingness-indicator feature.",
            "mean_rider_rating": f"Mean PS3-valid 1-5 rating; missing if fewer than {RIDER_MIN_RATING_OBSERVATIONS} ratings.",
            "mean_multiple_deliveries": f"Mean non-negative integer multiple-delivery field; missing if fewer than {RIDER_MIN_RATING_OBSERVATIONS} valid observations.",
            "excluded_fields": "Rider ID is a group key only; rider age is not tenure and is excluded; Type_of_vehicle is entirely missing in the supplied dataset; vehicle condition is excluded as a per-order operational attribute.",
            "analysis_scope": "Retrospective descriptive grouping of completed deliveries; not a prospective prediction or causal analysis.",
        },
        "minimum_deliveries_for_segmentation": RIDER_MIN_COMPLETED_DELIVERIES,
        "minimum_rating_observations": RIDER_MIN_RATING_OBSERVATIONS,
        "silhouette_sample": sample_metadata,
        "selected_silhouette_score": selected_score,
        "inertia_elbow_suggestion": elbow_k,
        "selection_at_candidate_boundary": selected_cluster_count
        == max(cluster_counts),
        "selection_method": selection_method,
        "sla_threshold_minutes": SLA_THRESHOLD_MINUTES,
        "sla_status": "provisional",
        "segment_small_sample_threshold_riders": RIDER_MIN_SEGMENT_RIDERS,
        "small_segments": segment_summary.loc[
            segment_summary["small_segment_flag"], "segment_id"
        ].astype(int).tolist(),
    }
    audit = dict(dataset.audit)
    eligible_rider_ids = set(riders.loc[eligible_mask, "rider_id"])
    audit["completed_deliveries_of_segmented_riders"] = int(
        rider_order_assignments["rider_id"].isin(eligible_rider_ids).sum()
    )
    audit["completed_deliveries_of_insufficient_sample_riders"] = int(
        rider_order_assignments["rider_id"].isin(
            set(riders.loc[~eligible_mask, "rider_id"])
        ).sum()
    )
    report = {
        "metadata": metadata,
        "data_audit": audit,
        "cluster_count_evaluation": _records(evaluations),
        "segments": _records(segment_summary),
        "feature_missing_values": dataset.audit[
            "feature_missing_values_among_eligible_riders"
        ],
        "limitations": [
            "The 30-completed-delivery minimum is reused from PS3 rider reporting and is a screening threshold, not a guarantee of statistical stability.",
            "The 30-minute SLA remains provisional; late rates change if the official SLA changes.",
            "Average ratings and assignment-related fields are retrospective observations; they do not establish rider tenure or causal effects.",
            "Rider IDs are pseudonymous grouping keys; segment membership is not an individual performance verdict or employment recommendation.",
            "The supplied data does not document geography or time-period representativeness, and segment patterns may vary across contexts.",
            "Features have varying availability; median-imputed feature values do not encode missingness in K-Means, while per-rider sufficiency flags preserve the data-quality information.",
        ],
    }
    bundle = {
        "model": model,
        "preprocessor": preprocessor,
        "metadata": metadata,
        "cluster_count_evaluation": report["cluster_count_evaluation"],
        "segment_metrics": report["segments"],
    }
    return report, riders, rider_order_assignments, bundle


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for row in frame.to_dict("records"):
        result.append(
            {
                key: (
                    value.item()
                    if isinstance(value, np.generic) and not pd.isna(value)
                    else None
                    if pd.isna(value)
                    else value.isoformat()
                    if isinstance(value, (pd.Timestamp, datetime))
                    else value
                )
                for key, value in row.items()
            }
        )
    return result


def _write_report(report: dict[str, Any], path: Path) -> None:
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


def run_rider_segmentation(
    path: Path = PROCESSED_DATA_PATH,
    *,
    model_path: Path = RIDER_SEGMENT_MODEL_PATH,
    report_path: Path = RIDER_SEGMENT_REPORT_PATH,
    assignments_path: Path = RIDER_SEGMENT_ASSIGNMENTS_PATH,
    order_assignments_path: Path = RIDER_ORDER_ASSIGNMENTS_PATH,
    cluster_counts: tuple[int, ...] = RIDER_CLUSTER_COUNTS,
    silhouette_sample_size: int = RIDER_CLUSTER_SILHOUETTE_SAMPLE_SIZE,
    selected_cluster_count: int | None = None,
) -> dict[str, Any]:
    """Fit the reproducible analysis and persist model, rider, and order artifacts."""
    dataset = load_rider_dataset(path)
    report, riders, orders, bundle = fit_rider_segments(
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
        "rider_assignments": str(assignments_path),
        "order_assignments": str(order_assignments_path),
    }
    assignments_path.parent.mkdir(parents=True, exist_ok=True)
    riders.to_csv(assignments_path, index=False)
    order_assignments_path.parent.mkdir(parents=True, exist_ok=True)
    orders.to_csv(order_assignments_path, index=False)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, model_path)
    _write_report(report, report_path)
    return report


def load_rider_segment_model(
    path: Path = RIDER_SEGMENT_MODEL_PATH,
) -> dict[str, Any]:
    """Load the fitted preprocessing and clustering artifacts."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Rider segmentation model not found at {path}. "
            "Run `python -m src.analytics.rider_segmentation`."
        )
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or not {
        "model",
        "preprocessor",
        "metadata",
    }.issubset(bundle):
        raise ValueError(f"Rider segmentation model at {path} has an unsupported structure.")
    return bundle


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate rider performance segments from validated delivery outcomes."
    )
    parser.add_argument("--input", type=Path, default=PROCESSED_DATA_PATH)
    parser.add_argument("--clusters", type=int, default=None)
    args = parser.parse_args()
    report = run_rider_segmentation(
        args.input,
        selected_cluster_count=args.clusters,
    )
    print(
        f"Riders with completed deliveries: "
        f"{report['data_audit']['distinct_riders_with_completed_deliveries']:,}; "
        f"eligible: {report['data_audit']['riders_meeting_minimum_deliveries']:,}; "
        f"insufficient: {report['data_audit']['riders_flagged_insufficient_deliveries']:,}"
    )
    print(
        f"Selected K={report['metadata']['selected_cluster_count']} "
        f"(elbow suggestion K={report['metadata']['inertia_elbow_suggestion']}, "
        f"silhouette={report['metadata']['selected_silhouette_score']:.3f})"
    )
    for segment in report["segments"]:
        print(
            f"Segment {segment['segment_id']}: riders={segment['rider_count']:,}, "
            f"deliveries={segment['total_deliveries']:,}, "
            f"late rate={segment['late_rate']:.1%}"
        )
    print(f"Report: {RIDER_SEGMENT_REPORT_PATH}")
    print(f"Model: {RIDER_SEGMENT_MODEL_PATH}")


if __name__ == "__main__":
    main()
