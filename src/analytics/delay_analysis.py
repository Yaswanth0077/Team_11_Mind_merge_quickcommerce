"""Descriptive analysis of delivery delays using the Phase 1 processed dataset."""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    COORDINATE_COLUMNS,
    DISTANCE_BAND_EDGES_KM,
    DISTANCE_BAND_LABELS,
    DISTANCE_COLUMN,
    LATE_LABEL_COLUMN,
    MISSING_TOKENS,
    PROCESSED_DATA_PATH,
    REPORT_DIR,
    RIDER_ID_COLUMN,
    SLA_THRESHOLD_MINUTES,
    TARGET_COLUMN,
)
from src.data.cleaning import extract_delivery_time_minutes
from src.data.feature_engineering import (
    coordinate_validity,
    label_late_orders,
    normalize_traffic,
    normalize_weather,
    parse_timestamps,
)
from src.data.loader import load_orders

MISSING_CATEGORY = "(Missing)"
DAY_NAMES = ("Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday")
DEFAULT_RIDER_MIN_ORDERS = 30
DEFAULT_GROUP_MIN_ORDERS = 100


@dataclass(frozen=True)
class DelayFilters:
    """Optional dashboard filters; None means unrestricted, while an empty choice matches none."""

    start_date: date | None = None
    end_date: date | None = None
    first_hour: int = 0
    last_hour: int = 23
    weather: tuple[str, ...] | None = None
    traffic: tuple[str, ...] | None = None
    order_type: tuple[str, ...] | None = None
    city: tuple[str, ...] | None = None
    distance_km: tuple[float, float] | None = None


def _boolean_series(values: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(values.dtype):
        return values.fillna(False)
    return values.astype("string").str.casefold().eq("true")


def _category_values(values: pd.Series) -> pd.Series:
    cleaned = values.astype("string").str.strip()
    missing = (
        cleaned.isna()
        | cleaned.eq("")
        | cleaned.str.casefold().isin(MISSING_TOKENS)
    )
    return cleaned.mask(missing, MISSING_CATEGORY)


def prepare_delay_data(
    frame: pd.DataFrame,
    sla_threshold_minutes: int = SLA_THRESHOLD_MINUTES,
) -> pd.DataFrame:
    """Validate and derive analysis fields without dropping any input rows."""
    required = {
        TARGET_COLUMN,
        "Order_Date",
        "Time_Orderd",
        "Weatherconditions",
        "Road_traffic_density",
        "Type_of_order",
        "City",
        "Delivery_person_Ratings",
        RIDER_ID_COLUMN,
        "Restaurant_latitude",
        "Restaurant_longitude",
        "Delivery_location_latitude",
        "Delivery_location_longitude",
    }
    missing_columns = sorted(required.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")
    if sla_threshold_minutes <= 0:
        raise ValueError("SLA threshold must be a positive number of minutes.")

    result = frame.copy()
    raw_target = result[TARGET_COLUMN]
    target = pd.to_numeric(raw_target, errors="coerce")
    text_targets = raw_target.astype("string").str.contains(
        r"\(min\)", case=False, na=False
    )
    if text_targets.any():
        target.loc[text_targets] = extract_delivery_time_minutes(raw_target.loc[text_targets])
    target = target.where(target.gt(0))
    result[TARGET_COLUMN] = target
    result[LATE_LABEL_COLUMN] = label_late_orders(target, sla_threshold_minutes)
    result["analysis_valid_order"] = target.notna()

    timestamps = parse_timestamps(result)
    result["analysis_date"] = timestamps["Order_Date"].dt.date
    result["analysis_hour"] = timestamps["Time_Orderd"].dt.hour.astype("Int64")
    result["analysis_day_of_week"] = timestamps["Order_Date"].dt.dayofweek.astype("Int64")
    result["analysis_day_name"] = timestamps["Order_Date"].dt.dayofweek.map(
        dict(enumerate(DAY_NAMES))
    ).astype("string")
    result["analysis_is_weekend"] = timestamps["Order_Date"].dt.dayofweek.isin(
        (5, 6)
    ).astype("boolean").where(timestamps["Order_Date"].notna(), pd.NA)

    _, coordinates_valid, _ = coordinate_validity(result)
    if DISTANCE_COLUMN in result:
        distance = pd.to_numeric(result[DISTANCE_COLUMN], errors="coerce")
    else:
        raise ValueError(
            f"Processed dataset must contain `{DISTANCE_COLUMN}`; rerun the Phase 1 pipeline."
        )
    coordinate_values = pd.DataFrame(
        {
            name: pd.to_numeric(result[column], errors="coerce")
            for name, column in COORDINATE_COLUMNS.items()
        },
        index=result.index,
    )
    coordinates_missing = coordinate_values.isna().any(axis=1)
    result["analysis_coordinates_valid"] = coordinates_valid.astype("boolean")
    result["analysis_coordinates_missing"] = coordinates_missing
    result["analysis_coordinates_invalid"] = ~coordinates_valid & ~coordinates_missing
    result["analysis_distance_km"] = distance.where(coordinates_valid & distance.gt(0))

    weather_column = (
        "weather_normalized" if "weather_normalized" in result else "Weatherconditions"
    )
    traffic_column = (
        "traffic_normalized"
        if "traffic_normalized" in result
        else "Road_traffic_density"
    )
    result["analysis_weather"] = _category_values(
        normalize_weather(result[weather_column])
    )
    result["analysis_traffic"] = _category_values(
        normalize_traffic(result[traffic_column])
    )
    for source_column, analysis_column in (
        ("Type_of_order", "analysis_order_type"),
        ("City", "analysis_city"),
    ):
        result[analysis_column] = _category_values(result[source_column])

    rating_text = result["Delivery_person_Ratings"].astype("string").str.strip()
    rating_missing = (
        rating_text.isna()
        | rating_text.eq("")
        | rating_text.str.casefold().isin(MISSING_TOKENS)
    )
    ratings = pd.to_numeric(rating_text, errors="coerce")
    result["analysis_rating_missing"] = rating_missing
    result["analysis_rating_invalid"] = (
        ~rating_missing & ~ratings.between(1, 5).fillna(False)
    )
    result["analysis_rating"] = ratings.where(ratings.between(1, 5))
    return result


def load_delay_data(
    path: Path = PROCESSED_DATA_PATH,
    sla_threshold_minutes: int = SLA_THRESHOLD_MINUTES,
) -> pd.DataFrame:
    """Load the Phase 1 processed CSV and prepare it for descriptive analysis."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Processed dataset not found at {path}. Run `python -m src.data.pipeline` first."
        )
    return prepare_delay_data(load_orders(path), sla_threshold_minutes)


def apply_delay_filters(
    frame: pd.DataFrame, filters: DelayFilters | None = None
) -> pd.DataFrame:
    """Apply selected filters consistently, retaining only orders with valid targets."""
    filters = filters or DelayFilters()
    if not 0 <= filters.first_hour <= filters.last_hour <= 23:
        raise ValueError("Hour range must satisfy 0 <= first_hour <= last_hour <= 23.")
    selected = frame.loc[frame["analysis_valid_order"]].copy()

    dated_orders = selected["analysis_date"].dropna()
    full_date_span_selected = (
        not dated_orders.empty
        and (filters.start_date is None or filters.start_date <= dated_orders.min())
        and (filters.end_date is None or filters.end_date >= dated_orders.max())
    )
    if filters.start_date is not None and not full_date_span_selected:
        selected = selected.loc[selected["analysis_date"].notna()]
        selected = selected.loc[selected["analysis_date"] >= filters.start_date]
    if filters.end_date is not None and not full_date_span_selected:
        selected = selected.loc[selected["analysis_date"].notna()]
        selected = selected.loc[selected["analysis_date"] <= filters.end_date]

    if filters.first_hour != 0 or filters.last_hour != 23:
        selected = selected.loc[
            selected["analysis_hour"].between(filters.first_hour, filters.last_hour)
        ]
    for column, choices in (
        ("analysis_weather", filters.weather),
        ("analysis_traffic", filters.traffic),
        ("analysis_order_type", filters.order_type),
        ("analysis_city", filters.city),
    ):
        if choices is not None:
            selected = selected.loc[selected[column].isin(choices)]
    if filters.distance_km is not None:
        minimum, maximum = filters.distance_km
        if minimum > maximum:
            raise ValueError("Minimum distance cannot exceed maximum distance.")
        distance = selected["analysis_distance_km"]
        selected = selected.loc[distance.between(minimum, maximum)]
    return selected


def delivery_time_summary(frame: pd.DataFrame) -> dict[str, Any]:
    """Summarize target distribution and late rate for valid delivery outcomes."""
    valid = frame.loc[frame["analysis_valid_order"]]
    delivery = pd.to_numeric(valid[TARGET_COLUMN], errors="coerce").dropna()
    late = _late_mask(valid)
    if delivery.empty:
        return {
            "order_count": 0,
            "mean_minutes": None,
            "median_minutes": None,
            "standard_deviation_minutes": None,
            "percentiles_minutes": {},
            "late_orders": 0,
            "late_rate": None,
        }
    percentiles = delivery.quantile([0.1, 0.25, 0.5, 0.75, 0.9, 0.95])
    return {
        "order_count": int(delivery.size),
        "mean_minutes": float(delivery.mean()),
        "median_minutes": float(delivery.median()),
        "standard_deviation_minutes": float(delivery.std(ddof=1))
        if delivery.size > 1
        else 0.0,
        "percentiles_minutes": {
            f"p{int(percentile * 100):02d}": float(value)
            for percentile, value in percentiles.items()
        },
        "late_orders": int(late.sum()),
        "late_rate": float(late.mean()),
    }


def _late_mask(frame: pd.DataFrame) -> pd.Series:
    """Return a non-null boolean late label for valid target rows."""
    return _boolean_series(frame[LATE_LABEL_COLUMN])


def _group_summary(
    frame: pd.DataFrame,
    group_column: str,
    order_by: str | None = None,
) -> pd.DataFrame:
    valid = frame.loc[frame["analysis_valid_order"]].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                group_column,
                "order_count",
                "mean_delivery_minutes",
                "median_delivery_minutes",
                "late_orders",
                "late_rate",
                "average_distance_km",
            ]
        )
    valid[LATE_LABEL_COLUMN] = _late_mask(valid).astype("int8")
    grouped = (
        valid.groupby(group_column, dropna=False, observed=True)
        .agg(
            order_count=(TARGET_COLUMN, "count"),
            mean_delivery_minutes=(TARGET_COLUMN, "mean"),
            median_delivery_minutes=(TARGET_COLUMN, "median"),
            late_orders=(LATE_LABEL_COLUMN, "sum"),
            late_rate=(LATE_LABEL_COLUMN, "mean"),
            average_distance_km=("analysis_distance_km", "mean"),
        )
        .reset_index()
    )
    if order_by is not None:
        grouped = grouped.sort_values(order_by)
    return grouped.reset_index(drop=True)


def analyze_by_hour(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize delivery time, late rate, volume, and distance by order hour."""
    timed = frame.loc[frame["analysis_hour"].notna()]
    grouped = _group_summary(timed, "analysis_hour", "analysis_hour")
    return grouped.rename(columns={"analysis_hour": "hour"})


def analyze_by_day(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize order outcomes by weekday."""
    dated = frame.loc[frame["analysis_day_of_week"].notna()]
    grouped = _group_summary(dated, "analysis_day_of_week", "analysis_day_of_week")
    grouped["day_name"] = grouped["analysis_day_of_week"].map(
        dict(enumerate(DAY_NAMES))
    )
    return grouped.rename(columns={"analysis_day_of_week": "day_of_week"})


def analyze_by_category(frame: pd.DataFrame, category: str) -> pd.DataFrame:
    """Summarize outcomes for a selected weather, traffic, order, or city category."""
    allowed = {
        "weather": "analysis_weather",
        "traffic": "analysis_traffic",
        "order_type": "analysis_order_type",
        "city": "analysis_city",
    }
    if category not in allowed:
        raise ValueError(f"Unsupported category: {category}")
    grouped = _group_summary(frame, allowed[category], allowed[category])
    if category == "traffic" and not grouped.empty:
        traffic_order = {
            label: position
            for position, label in enumerate(
                ("Low", "Medium", "High", "Jam", MISSING_CATEGORY)
            )
        }
        grouped["_category_order"] = (
            grouped[allowed[category]].map(traffic_order).fillna(5)
        )
        grouped = (
            grouped.sort_values(["_category_order", allowed[category]])
            .drop(columns="_category_order")
            .reset_index(drop=True)
        )
    return grouped


def analyze_weather_traffic(frame: pd.DataFrame) -> pd.DataFrame:
    """Compare late rate and average delivery time for each weather/traffic pair."""
    valid = frame.loc[frame["analysis_valid_order"]].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                "analysis_weather",
                "analysis_traffic",
                "order_count",
                "mean_delivery_minutes",
                "late_rate",
            ]
        )
    valid[LATE_LABEL_COLUMN] = _late_mask(valid).astype("int8")
    return (
        valid.groupby(
            ["analysis_weather", "analysis_traffic"], dropna=False, observed=True
        )
        .agg(
            order_count=(TARGET_COLUMN, "count"),
            mean_delivery_minutes=(TARGET_COLUMN, "mean"),
            late_rate=(LATE_LABEL_COLUMN, "mean"),
        )
        .reset_index()
    )


def analyze_distance_bands(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize valid-coordinate orders using the Phase 1 configured bands."""
    valid = frame.loc[
        frame["analysis_valid_order"] & frame["analysis_distance_km"].notna()
    ].copy()
    valid["analysis_distance_band"] = pd.cut(
        valid["analysis_distance_km"],
        bins=DISTANCE_BAND_EDGES_KM,
        labels=DISTANCE_BAND_LABELS,
        right=False,
        include_lowest=True,
    ).astype("string")
    grouped = _group_summary(valid, "analysis_distance_band")
    if not grouped.empty:
        grouped["analysis_distance_band"] = pd.Categorical(
            grouped["analysis_distance_band"],
            categories=DISTANCE_BAND_LABELS,
            ordered=True,
        )
        grouped = grouped.sort_values("analysis_distance_band").reset_index(drop=True)
    return grouped


def analyze_ratings(frame: pd.DataFrame) -> pd.DataFrame:
    """Summarize delivery outcomes by valid 1–5 rider rating."""
    valid = frame.loc[
        frame["analysis_valid_order"] & frame["analysis_rating"].notna()
    ].copy()
    valid["analysis_rating_group"] = valid["analysis_rating"].astype(float)
    grouped = _group_summary(valid, "analysis_rating_group", "analysis_rating_group")
    return grouped.rename(columns={"analysis_rating_group": "rating"})


def rider_performance(
    frame: pd.DataFrame,
    minimum_orders: int = DEFAULT_RIDER_MIN_ORDERS,
) -> pd.DataFrame:
    """Calculate rider delivery summaries; no rider-tenure or causal inference."""
    if minimum_orders < 1:
        raise ValueError("minimum_orders must be at least one.")
    valid = frame.loc[
        frame["analysis_valid_order"] & frame[RIDER_ID_COLUMN].notna()
    ].copy()
    if valid.empty:
        return pd.DataFrame(
            columns=[
                RIDER_ID_COLUMN,
                "order_count",
                "mean_delivery_minutes",
                "late_rate",
                "average_rating",
                "meets_minimum_sample",
            ]
        )
    valid[LATE_LABEL_COLUMN] = _late_mask(valid).astype("int8")
    riders = (
        valid.groupby(RIDER_ID_COLUMN, dropna=True)
        .agg(
            order_count=(TARGET_COLUMN, "count"),
            mean_delivery_minutes=(TARGET_COLUMN, "mean"),
            late_orders=(LATE_LABEL_COLUMN, "sum"),
            late_rate=(LATE_LABEL_COLUMN, "mean"),
            average_rating=("analysis_rating", "mean"),
        )
        .reset_index()
    )
    riders["meets_minimum_sample"] = riders["order_count"].ge(minimum_orders)
    return riders.sort_values(
        ["late_rate", "order_count"], ascending=[False, False]
    ).reset_index(drop=True)


def network_benchmark_metrics(frame: pd.DataFrame) -> dict[str, float | int | None]:
    """Calculate fixed full-network values once for reuse across dashboard interactions."""
    summary = delivery_time_summary(frame)
    distances = pd.to_numeric(
        frame.loc[frame["analysis_valid_order"], "analysis_distance_km"],
        errors="coerce",
    ).dropna()
    return {
        "order_count": summary["order_count"],
        "mean_minutes": summary["mean_minutes"],
        "median_minutes": summary["median_minutes"],
        "late_rate": summary["late_rate"],
        "average_distance_km": (
            float(distances.mean()) if not distances.empty else None
        ),
    }


def benchmark_comparison_to_network(
    selected: pd.DataFrame,
    network: dict[str, float | int | None],
) -> pd.DataFrame:
    """Compare the selected segment with precomputed full-network values."""
    selected_metrics = delivery_time_summary(selected)
    distances = pd.to_numeric(
        selected.loc[selected["analysis_valid_order"], "analysis_distance_km"],
        errors="coerce",
    ).dropna()
    metrics = [
        ("Orders", "order_count", "order_count", "count"),
        ("Average delivery time (min)", "mean_minutes", "mean_minutes", "minutes"),
        ("Median delivery time (min)", "median_minutes", "median_minutes", "minutes"),
        ("Late-order rate (%)", "late_rate", "late_rate", "rate"),
    ]
    rows = []
    for label, selected_key, network_key, kind in metrics:
        selected_value = selected_metrics[selected_key]
        overall_value = network[network_key]
        if kind == "rate":
            selected_value = selected_value * 100 if selected_value is not None else None
            overall_value = overall_value * 100 if overall_value is not None else None
        rows.append(_benchmark_row(label, selected_value, overall_value, kind))
    rows.append(
        _benchmark_row(
            "Average delivery distance (km)",
            float(distances.mean()) if not distances.empty else None,
            network["average_distance_km"],
            "distance",
        )
    )
    return pd.DataFrame(rows)


def benchmark_comparison(selected: pd.DataFrame, overall: pd.DataFrame) -> pd.DataFrame:
    """Compare a filtered segment with a separately calculated fixed network baseline."""
    return benchmark_comparison_to_network(
        selected,
        network_benchmark_metrics(overall),
    )


def _benchmark_row(
    label: str, selected: float | int | None, overall: float | int | None, kind: str
) -> dict[str, Any]:
    difference = (
        float(selected) - float(overall)
        if selected is not None and overall is not None
        else None
    )
    percent_difference = (
        difference / float(overall) * 100
        if difference is not None and float(overall) != 0
        else None
    )
    return {
        "metric": label,
        "selected": selected,
        "overall_network": overall,
        "difference": difference,
        "percent_difference": percent_difference,
        "difference_unit": "percentage points" if kind == "rate" else kind,
    }


def generate_insights(
    selected: pd.DataFrame,
    minimum_group_orders: int = DEFAULT_GROUP_MIN_ORDERS,
    minimum_rider_orders: int = DEFAULT_RIDER_MIN_ORDERS,
    maximum_insights: int = 5,
) -> list[dict[str, Any]]:
    """Find high-late-rate groups with adequate samples; describe association only."""
    if minimum_group_orders < 1 or minimum_rider_orders < 1:
        raise ValueError("Minimum sample sizes must be positive.")
    baseline = delivery_time_summary(selected)
    if baseline["order_count"] == 0 or baseline["late_rate"] is None:
        return []
    candidates: list[dict[str, Any]] = []
    dimensions = (
        ("hour", analyze_by_hour(selected), "hour"),
        ("weekday", analyze_by_day(selected), "day_name"),
        ("weather", analyze_by_category(selected, "weather"), "analysis_weather"),
        ("traffic", analyze_by_category(selected, "traffic"), "analysis_traffic"),
        (
            "distance band",
            analyze_distance_bands(selected),
            "analysis_distance_band",
        ),
        ("order type", analyze_by_category(selected, "order_type"), "analysis_order_type"),
    )
    for dimension, grouped, label_column in dimensions:
        for row in grouped.to_dict("records"):
            count = int(row["order_count"])
            rate = float(row["late_rate"])
            if count < minimum_group_orders:
                continue
            difference = rate - float(baseline["late_rate"])
            if difference <= 0:
                continue
            label = row.get(label_column)
            if dimension == "hour":
                label = f"{int(label):02d}:00–{int(label):02d}:59"
            candidates.append(
                {
                    "dimension": dimension,
                    "segment": str(label),
                    "order_count": count,
                    "late_rate": rate,
                    "baseline_late_rate": float(baseline["late_rate"]),
                    "difference_percentage_points": difference * 100,
                    "mean_delivery_minutes": float(row["mean_delivery_minutes"]),
                    "finding": (
                        f"{count:,} {dimension} orders in “{label}” had a "
                        f"{rate:.1%} late rate, {difference:.1%} above the "
                        f"{baseline['late_rate']:.1%} rate in the current selection."
                    ),
                    "operational_relevance": (
                        "This observed segment may merit closer monitoring or staffing review; "
                        "the comparison does not establish a cause."
                    ),
                }
            )
    candidates.sort(
        key=lambda item: (
            item["difference_percentage_points"],
            item["order_count"],
        ),
        reverse=True,
    )
    insights = candidates[:maximum_insights]

    qualified_riders = rider_performance(selected, minimum_rider_orders)
    qualified_riders = qualified_riders.loc[qualified_riders["meets_minimum_sample"]]
    if not qualified_riders.empty:
        rider = qualified_riders.iloc[0]
        difference = float(rider["late_rate"]) - float(baseline["late_rate"])
        if difference > 0:
            insights.append(
                {
                    "dimension": "rider",
                    "segment": str(rider[RIDER_ID_COLUMN]),
                    "order_count": int(rider["order_count"]),
                    "late_rate": float(rider["late_rate"]),
                    "baseline_late_rate": float(baseline["late_rate"]),
                    "difference_percentage_points": difference * 100,
                    "mean_delivery_minutes": float(rider["mean_delivery_minutes"]),
                    "finding": (
                        f"Rider {rider[RIDER_ID_COLUMN]} had a {rider['late_rate']:.1%} "
                        f"late rate across {int(rider['order_count']):,} orders, "
                        f"{difference:.1%} above the current selection."
                    ),
                    "operational_relevance": (
                        "This descriptive comparison is sample-qualified and does not infer "
                        "rider experience or explain why outcomes differ."
                    ),
                }
            )
    insights.sort(
        key=lambda item: (
            item["difference_percentage_points"],
            item["order_count"],
        ),
        reverse=True,
    )
    return insights[:maximum_insights]


def analysis_quality_summary(frame: pd.DataFrame) -> dict[str, int]:
    """Count eligible and unavailable values relevant to PS3 comparisons."""
    valid_target = frame["analysis_valid_order"]
    missing_coordinates = valid_target & frame["analysis_coordinates_missing"]
    invalid_coordinates = valid_target & frame["analysis_coordinates_invalid"]
    missing_ratings = valid_target & frame["analysis_rating_missing"]
    invalid_ratings = valid_target & frame["analysis_rating_invalid"]
    return {
        "input_records": int(len(frame)),
        "records_retained_by_preparation": int(len(frame)),
        "records_excluded_by_preparation": 0,
        "valid_delivery_time_records": int(valid_target.sum()),
        "missing_or_invalid_delivery_time_records": int((~valid_target).sum()),
        "missing_order_date_records": int(frame["analysis_date"].isna().sum()),
        "missing_order_time_records": int(frame["analysis_hour"].isna().sum()),
        "valid_coordinate_distance_records": int(
            (valid_target & frame["analysis_distance_km"].notna()).sum()
        ),
        "invalid_coordinate_records": int(invalid_coordinates.sum()),
        "missing_coordinate_records": int(missing_coordinates.sum()),
        "invalid_or_missing_coordinate_records": int(
            invalid_coordinates.sum() + missing_coordinates.sum()
        ),
        "unavailable_distance_records": int(
            (valid_target & frame["analysis_distance_km"].isna()).sum()
        ),
        "missing_rating_records": int(missing_ratings.sum()),
        "invalid_rating_records": int(invalid_ratings.sum()),
        "invalid_or_missing_rating_records": int(
            missing_ratings.sum() + invalid_ratings.sum()
        ),
        "missing_weather_records": int(
            (valid_target & frame["analysis_weather"].eq(MISSING_CATEGORY)).sum()
        ),
        "missing_traffic_records": int(
            (valid_target & frame["analysis_traffic"].eq(MISSING_CATEGORY)).sum()
        ),
        "missing_order_type_records": int(
            (valid_target & frame["analysis_order_type"].eq(MISSING_CATEGORY)).sum()
        ),
        "missing_city_records": int(
            (valid_target & frame["analysis_city"].eq(MISSING_CATEGORY)).sum()
        ),
    }


def build_analysis_report(
    frame: pd.DataFrame,
    source_path: Path = PROCESSED_DATA_PATH,
    sla_threshold_minutes: int = SLA_THRESHOLD_MINUTES,
) -> dict[str, Any]:
    """Build a reproducible full-dataset PS3 summary and ranked observations."""
    return {
        "source": str(source_path),
        "sla_threshold_minutes": sla_threshold_minutes,
        "sla_status": "provisional",
        "quality": analysis_quality_summary(frame),
        "delivery_time": delivery_time_summary(frame),
        "hour": _records(analyze_by_hour(frame)),
        "day": _records(analyze_by_day(frame)),
        "weather": _records(analyze_by_category(frame, "weather")),
        "traffic": _records(analyze_by_category(frame, "traffic")),
        "weather_traffic": _records(analyze_weather_traffic(frame)),
        "distance_bands": _records(analyze_distance_bands(frame)),
        "order_type": _records(analyze_by_category(frame, "order_type")),
        "rating": _records(analyze_ratings(frame)),
        "rider_minimum_orders": DEFAULT_RIDER_MIN_ORDERS,
        "riders_meeting_minimum": _records(
            rider_performance(frame).query("meets_minimum_sample")
        ),
        "insights": generate_insights(frame),
        "interpretation": (
            "All group differences are descriptive associations. They do not establish causation; "
            "rider tenure is not present in the dataset."
        ),
    }


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert table values to JSON-safe Python scalars."""
    result = []
    for row in frame.to_dict("records"):
        result.append(
            {
                key: (
                    None
                    if value is None or (isinstance(value, float) and math.isnan(value))
                    else value.item()
                    if isinstance(value, np.generic)
                    else value
                )
                for key, value in row.items()
            }
        )
    return result


def _write_markdown(report: dict[str, Any], path: Path) -> None:
    metrics = report["delivery_time"]
    lines = [
        "# Delivery Delay Intelligence — PS3",
        "",
        f"Descriptive analysis of {report['quality']['valid_delivery_time_records']:,} "
        f"valid delivery outcomes. SLA: {report['sla_threshold_minutes']} minutes "
        "(provisional).",
        "",
        "## Overall delivery performance",
        "",
        f"- Mean: {metrics['mean_minutes']:.2f} min; median: {metrics['median_minutes']:.2f} min.",
        f"- Standard deviation: {metrics['standard_deviation_minutes']:.2f} min.",
        f"- Late: {metrics['late_orders']:,} orders ({metrics['late_rate']:.2%}).",
        "",
        "## Data coverage",
        "",
    ]
    for label, key in (
        ("Rows retained by preparation", "records_retained_by_preparation"),
        ("Rows excluded by preparation", "records_excluded_by_preparation"),
        ("Valid delivery times", "valid_delivery_time_records"),
        ("Missing/invalid delivery times", "missing_or_invalid_delivery_time_records"),
        ("Valid-coordinate distances", "valid_coordinate_distance_records"),
        ("Invalid coordinate pairs", "invalid_coordinate_records"),
        ("Missing coordinate pairs", "missing_coordinate_records"),
        ("Invalid/missing-coordinate distances", "invalid_or_missing_coordinate_records"),
        ("Unavailable distances", "unavailable_distance_records"),
        ("Missing rider ratings", "missing_rating_records"),
        ("Invalid rider ratings", "invalid_rating_records"),
        ("Missing weather", "missing_weather_records"),
        ("Missing traffic", "missing_traffic_records"),
        ("Missing order type", "missing_order_type_records"),
        ("Missing city", "missing_city_records"),
    ):
        lines.append(f"- {label}: {report['quality'][key]:,}.")
    lines += ["", "## Highest observed late-rate segments", ""]
    if report["insights"]:
        for insight in report["insights"]:
            lines.append(
                f"- **{insight['dimension']} — {insight['segment']}:** "
                f"{insight['finding']} {insight['operational_relevance']}"
            )
    else:
        lines.append("No segment met the sample-size and higher-than-baseline criteria.")
    lines += [
        "",
        "## Interpretation",
        "",
        report["interpretation"],
        "",
        "Rider summaries include only riders with at least "
        f"{report['rider_minimum_orders']} orders. The dataset does not contain rider tenure.",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def write_analysis_report(report: dict[str, Any], report_dir: Path = REPORT_DIR) -> None:
    """Save machine-readable and concise human-readable PS3 findings."""
    report_dir.mkdir(parents=True, exist_ok=True)
    (report_dir / "delay_analysis.json").write_text(
        json.dumps(report, indent=2, allow_nan=False), encoding="utf-8"
    )
    _write_markdown(report, report_dir / "delay_analysis.md")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate full-dataset PS3 findings.")
    parser.add_argument("--input", type=Path, default=PROCESSED_DATA_PATH)
    args = parser.parse_args()
    frame = load_delay_data(args.input)
    report = build_analysis_report(frame, args.input)
    write_analysis_report(report)
    print(f"Analyzed {report['quality']['valid_delivery_time_records']:,} valid orders.")
    print(f"Findings: {REPORT_DIR / 'delay_analysis.md'}")


if __name__ == "__main__":
    main()
