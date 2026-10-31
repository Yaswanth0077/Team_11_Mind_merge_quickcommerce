"""Validated order-placement input and paired ETA/late-risk inference."""

from __future__ import annotations

import math
from datetime import datetime
from typing import Any, Mapping

from config import INDIA_BOUNDS, SLA_THRESHOLD_MINUTES
from src.models.classification import predict_late_order
from src.models.regression import PREDICTION_INPUT_COLUMNS, predict_delivery_time

COORDINATE_INPUTS = (
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
)
REQUIRED_INPUTS = ("Order_Date", "Time_Orderd", "City", "Type_of_order")
_MISSING_VALUES = {"", "nan", "na", "n/a", "null", "none"}


def validate_order_placement_inputs(order: Mapping[str, Any]) -> dict[str, Any]:
    """Validate order-placement inputs and normalize them for both saved pipelines."""
    unsupported = set(order).difference(PREDICTION_INPUT_COLUMNS)
    if unsupported:
        raise ValueError(
            "Unsupported or post-placement input field(s): "
            + ", ".join(sorted(unsupported))
        )

    normalized = dict(order)
    for field in REQUIRED_INPUTS:
        value = normalized.get(field)
        if value is None or str(value).strip().casefold() in _MISSING_VALUES:
            raise ValueError(f"{field} is required for this prediction demo.")
        normalized[field] = str(value).strip()

    try:
        datetime.strptime(normalized["Order_Date"], "%d-%m-%Y")
    except ValueError as exc:
        raise ValueError("Order_Date must use the format DD-MM-YYYY.") from exc
    try:
        datetime.strptime(normalized["Time_Orderd"], "%H:%M:%S")
    except ValueError as exc:
        raise ValueError("Time_Orderd must use the format HH:MM:SS.") from exc

    coordinate_values = {
        field: normalized.get(field) for field in COORDINATE_INPUTS
    }
    provided = {
        field: value
        for field, value in coordinate_values.items()
        if value is not None and str(value).strip().casefold() not in _MISSING_VALUES
    }
    if provided and len(provided) != len(COORDINATE_INPUTS):
        raise ValueError(
            "Provide all four store/customer coordinates, or leave all four blank."
        )
    if provided:
        bounds = {
            "Restaurant_latitude": INDIA_BOUNDS["latitude"],
            "Delivery_location_latitude": INDIA_BOUNDS["latitude"],
            "Restaurant_longitude": INDIA_BOUNDS["longitude"],
            "Delivery_location_longitude": INDIA_BOUNDS["longitude"],
        }
        for field, raw_value in provided.items():
            try:
                value = float(raw_value)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{field} must be a valid number.") from exc
            lower, upper = bounds[field]
            if not math.isfinite(value) or not lower <= value <= upper:
                raise ValueError(
                    f"{field} must be within the configured India screening bounds "
                    f"[{lower:g}, {upper:g}]."
                )
            normalized[field] = value
    else:
        for field in COORDINATE_INPUTS:
            normalized.pop(field, None)

    return normalized


def predict_order_outcomes(order: Mapping[str, Any]) -> dict[str, Any]:
    """Return saved-model ETA and late-risk estimates without changing model policy."""
    validated = validate_order_placement_inputs(order)
    eta = predict_delivery_time(validated)
    late_risk = predict_late_order(validated)
    if late_risk["sla_threshold_minutes"] != SLA_THRESHOLD_MINUTES:
        raise ValueError("The classifier artifact uses an unexpected SLA threshold.")
    if not math.isclose(late_risk["classification_threshold"], 0.15):
        raise ValueError("The saved classifier threshold is not the selected 0.15 value.")
    return {
        "predicted_delivery_time_minutes": eta["predicted_delivery_time_minutes"],
        "late_order_risk": late_risk["risk_score"],
        "classification_threshold": late_risk["classification_threshold"],
        "predicted_status": late_risk["predicted_status"],
        "sla_threshold_minutes": late_risk["sla_threshold_minutes"],
        "regression_model": eta["model"],
        "classification_model": late_risk["model"],
        "model_version": eta["model_version"],
        "warnings": list(dict.fromkeys([*eta["warnings"], *late_risk["warnings"]])),
    }
