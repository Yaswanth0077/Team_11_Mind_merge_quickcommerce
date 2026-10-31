"""Reusable, leakage-aware feature transformations."""

from __future__ import annotations

import numpy as np
import pandas as pd

from config import (
    COORDINATE_COLUMNS,
    DATE_FORMAT,
    DISTANCE_BAND_EDGES_KM,
    DISTANCE_BAND_LABELS,
    DISTANCE_COLUMN,
    INDIA_BOUNDS,
    LATE_LABEL_COLUMN,
    PEAK_HOUR_WINDOWS,
    PRE_DELIVERY_FEATURE_COLUMNS,
    MIN_RIDER_AGE_YEARS,
    SLA_THRESHOLD_MINUTES,
    SUSPICIOUS_DISTANCE_KM,
    TARGET_COLUMN,
    TIME_FORMAT,
)


def _numeric_series(values: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(values, errors="coerce")
    return pd.Series(
        numeric.to_numpy(dtype="float64", na_value=np.nan),
        index=values.index,
        dtype="float64",
    )


def coordinate_validity(frame: pd.DataFrame) -> tuple[pd.Series, pd.Series, pd.Series]:
    """Return global-range, dataset-region, and zero-pair coordinate validity flags."""
    latitude_store = _numeric_series(frame[COORDINATE_COLUMNS["store_latitude"]])
    longitude_store = _numeric_series(frame[COORDINATE_COLUMNS["store_longitude"]])
    latitude_customer = _numeric_series(frame[COORDINATE_COLUMNS["customer_latitude"]])
    longitude_customer = _numeric_series(frame[COORDINATE_COLUMNS["customer_longitude"]])

    range_valid = (
        latitude_store.between(-90, 90)
        & longitude_store.between(-180, 180)
        & latitude_customer.between(-90, 90)
        & longitude_customer.between(-180, 180)
    )
    zero_pair = (
        (latitude_store.eq(0) & longitude_store.eq(0))
        | (latitude_customer.eq(0) & longitude_customer.eq(0))
    )
    region_valid = (
        range_valid
        & ~zero_pair
        & latitude_store.between(*INDIA_BOUNDS["latitude"])
        & latitude_customer.between(*INDIA_BOUNDS["latitude"])
        & longitude_store.between(*INDIA_BOUNDS["longitude"])
        & longitude_customer.between(*INDIA_BOUNDS["longitude"])
    )
    return range_valid, region_valid, zero_pair


def haversine_km(
    latitude_1: pd.Series | np.ndarray | float,
    longitude_1: pd.Series | np.ndarray | float,
    latitude_2: pd.Series | np.ndarray | float,
    longitude_2: pd.Series | np.ndarray | float,
) -> pd.Series | float:
    """Calculate great-circle distance in kilometres; out-of-range coordinates yield NaN."""
    inputs = (latitude_1, longitude_1, latitude_2, longitude_2)
    scalar_input = all(np.isscalar(value) for value in inputs)
    if scalar_input:
        try:
            arrays = [np.asarray(value, dtype="float64") for value in inputs]
        except (TypeError, ValueError):
            return float("nan")
        index = None
    else:
        series = [pd.Series(value) for value in inputs]
        index = series[0].index
        if any(not item.index.equals(index) for item in series[1:]):
            raise ValueError("Coordinate series must have matching indexes.")
        arrays = [
            _numeric_series(item).to_numpy(dtype="float64", na_value=np.nan)
            for item in series
        ]

    lat1, lon1, lat2, lon2 = arrays
    valid = (
        np.isfinite(lat1)
        & np.isfinite(lat2)
        & np.isfinite(lon1)
        & np.isfinite(lon2)
        & (lat1 >= -90)
        & (lat1 <= 90)
        & (lat2 >= -90)
        & (lat2 <= 90)
        & (lon1 >= -180)
        & (lon1 <= 180)
        & (lon2 >= -180)
        & (lon2 <= 180)
    )
    lat1_r, lon1_r, lat2_r, lon2_r = np.radians(arrays)
    a = (
        np.sin((lat2_r - lat1_r) / 2) ** 2
        + np.cos(lat1_r)
        * np.cos(lat2_r)
        * np.sin((lon2_r - lon1_r) / 2) ** 2
    )
    distance = 6371.0 * 2 * np.arctan2(np.sqrt(np.clip(a, 0, 1)), np.sqrt(np.clip(1 - a, 0, 1)))
    distance = np.where(valid, distance, np.nan)
    if scalar_input:
        return float(distance)
    return pd.Series(distance, index=index, dtype="float64")


def parse_timestamps(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Parse date and time columns with declared formats; malformed values become NaT."""
    date_values = frame["Order_Date"].astype("string").str.strip()
    order_values = frame["Time_Orderd"].astype("string").str.strip()
    pickup_values = frame["Time_Order_picked"].astype("string").str.strip()
    return {
        "Order_Date": pd.to_datetime(date_values, format=DATE_FORMAT, errors="coerce"),
        "Time_Orderd": pd.to_datetime(order_values, format=TIME_FORMAT, errors="coerce"),
        "Time_Order_picked": pd.to_datetime(pickup_values, format=TIME_FORMAT, errors="coerce"),
    }


def normalize_weather(values: pd.Series) -> pd.Series:
    """Strip source prefixes/spacing and canonicalize recognized weather labels."""
    cleaned = values.astype("string").str.strip().str.replace(
        r"^conditions\s+", "", regex=True, case=False
    )
    labels = {
        "sunny": "Sunny",
        "cloudy": "Cloudy",
        "fog": "Fog",
        "stormy": "Stormy",
        "windy": "Windy",
        "sandstorms": "Sandstorms",
    }
    return cleaned.str.casefold().map(labels).fillna(cleaned)


def normalize_traffic(values: pd.Series) -> pd.Series:
    """Trim and standardize known traffic-density categories."""
    cleaned = values.astype("string").str.strip()
    labels = {label.casefold(): label for label in ("Low", "Medium", "High", "Jam")}
    return cleaned.str.casefold().map(labels).fillna(cleaned)


def label_late_orders(
    delivery_minutes: pd.Series, sla_threshold_minutes: int = SLA_THRESHOLD_MINUTES
) -> pd.Series:
    """Label delivery outcomes strictly greater than the configured provisional SLA."""
    if sla_threshold_minutes <= 0:
        raise ValueError("SLA threshold must be a positive number of minutes.")
    numeric = pd.to_numeric(delivery_minutes, errors="coerce")
    return numeric.gt(sla_threshold_minutes).astype("boolean").where(numeric.notna(), pd.NA)


def engineer_features(
    frame: pd.DataFrame, sla_threshold_minutes: int = SLA_THRESHOLD_MINUTES
) -> pd.DataFrame:
    """Add reusable date, time, distance, category, and outcome-label features."""
    result = frame.copy()
    timestamps = parse_timestamps(result)
    order_time = timestamps["Time_Orderd"]
    order_date = timestamps["Order_Date"]

    range_valid, region_valid, zero_pair = coordinate_validity(result)
    result["coordinate_range_valid"] = range_valid.astype("boolean")
    result["coordinates_valid"] = region_valid.astype("boolean")
    result["zero_coordinate_pair"] = zero_pair.astype("boolean")
    result["geographically_implausible_pair"] = (~region_valid).astype("boolean")
    distance = haversine_km(
        result[COORDINATE_COLUMNS["store_latitude"]],
        result[COORDINATE_COLUMNS["store_longitude"]],
        result[COORDINATE_COLUMNS["customer_latitude"]],
        result[COORDINATE_COLUMNS["customer_longitude"]],
    )
    result["raw_haversine_distance_km"] = distance
    result[DISTANCE_COLUMN] = distance.where(region_valid)
    result["distance_over_review_threshold"] = (
        distance.gt(SUSPICIOUS_DISTANCE_KM).astype("boolean").where(distance.notna(), pd.NA)
    )

    result["order_hour"] = order_time.dt.hour.astype("Int64")
    result["order_day"] = order_date.dt.day.astype("Int64")
    result["day_of_week"] = order_date.dt.dayofweek.astype("Int64")
    result["is_weekend"] = order_date.dt.dayofweek.isin((5, 6)).astype("boolean").where(
        order_date.notna(), pd.NA
    )
    peak_hours = {
        hour
        for start, end in PEAK_HOUR_WINDOWS
        for hour in range(start, end)
    }
    result["is_peak_hour"] = order_time.dt.hour.isin(peak_hours).astype("boolean").where(
        order_time.notna(), pd.NA
    )

    result["distance_band"] = pd.cut(
        result[DISTANCE_COLUMN],
        bins=DISTANCE_BAND_EDGES_KM,
        labels=DISTANCE_BAND_LABELS,
        right=False,
        include_lowest=True,
    ).astype("string")
    result["weather_normalized"] = normalize_weather(result["Weatherconditions"])
    result["traffic_normalized"] = normalize_traffic(result["Road_traffic_density"])

    age = pd.to_numeric(result["Delivery_person_Age"], errors="coerce")
    age_valid = age.ge(MIN_RIDER_AGE_YEARS)
    result["rider_age_band"] = pd.cut(
        age.where(age_valid),
        bins=(MIN_RIDER_AGE_YEARS - 1, 25, 35, 45, float("inf")),
        labels=("18-25", "26-35", "36-45", "46+"),
        right=True,
    ).astype("string")

    if TARGET_COLUMN in result:
        target = pd.to_numeric(result[TARGET_COLUMN], errors="coerce")
        result[LATE_LABEL_COLUMN] = label_late_orders(target, sla_threshold_minutes)
    return result


def get_pre_delivery_features(frame: pd.DataFrame) -> pd.DataFrame:
    """Return only the explicitly approved order-placement feature set."""
    missing = [column for column in PRE_DELIVERY_FEATURE_COLUMNS if column not in frame.columns]
    if missing:
        raise ValueError(f"Required engineered features are missing: {', '.join(missing)}")
    return frame.loc[:, list(PRE_DELIVERY_FEATURE_COLUMNS)].copy()


def outlier_flags(
    values: pd.Series,
) -> tuple[pd.Series, pd.Series, dict[str, float | int | None]]:
    """Return IQR and percentile flags plus their thresholds; retain all observations."""
    parsed = pd.to_numeric(values, errors="coerce")
    numeric = pd.Series(
        parsed.to_numpy(dtype="float64", na_value=np.nan),
        index=parsed.index,
        dtype="float64",
    )
    valid = numeric.dropna()
    if valid.empty:
        no_values = pd.Series(pd.NA, index=numeric.index, dtype="boolean")
        return no_values, no_values.copy(), {
            "valid_count": 0,
            "q1": None,
            "q3": None,
            "iqr_lower_bound": None,
            "iqr_upper_bound": None,
            "p01": None,
            "p99": None,
            "iqr_outlier_count": 0,
            "percentile_outlier_count": 0,
        }
    q1, q3 = valid.quantile([0.25, 0.75])
    iqr = q3 - q1
    p01, p99 = valid.quantile([0.01, 0.99])
    iqr_flag = numeric.lt(q1 - 1.5 * iqr) | numeric.gt(q3 + 1.5 * iqr)
    percentile_flag = numeric.lt(p01) | numeric.gt(p99)
    iqr_flag = iqr_flag.astype("boolean").where(numeric.notna(), pd.NA)
    percentile_flag = percentile_flag.astype("boolean").where(numeric.notna(), pd.NA)
    return iqr_flag, percentile_flag, {
        "valid_count": int(valid.size),
        "q1": float(q1),
        "q3": float(q3),
        "iqr_lower_bound": float(q1 - 1.5 * iqr),
        "iqr_upper_bound": float(q3 + 1.5 * iqr),
        "p01": float(p01),
        "p99": float(p99),
        "iqr_outlier_count": int(iqr_flag.sum()),
        "percentile_outlier_count": int(percentile_flag.sum()),
    }
