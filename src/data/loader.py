"""Validated loading helpers for the supplied delivery dataset."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pandas as pd

from config import ORDER_ID_COLUMN, RIDER_ID_COLUMN, TARGET_COLUMN

REQUIRED_COLUMNS = {
    ORDER_ID_COLUMN,
    RIDER_ID_COLUMN,
    "Delivery_person_Age",
    "Delivery_person_Ratings",
    "Restaurant_latitude",
    "Restaurant_longitude",
    "Delivery_location_latitude",
    "Delivery_location_longitude",
    "Order_Date",
    "Time_Orderd",
    "Time_Order_picked",
    "Weatherconditions",
    "Road_traffic_density",
    "Vehicle_condition",
    "Type_of_order",
    "Type_of_vehicle",
    "multiple_deliveries",
    "Festival",
    "City",
    TARGET_COLUMN,
}


def load_orders(path: Path) -> pd.DataFrame:
    """Load a CSV without silently converting textual missing tokens to nulls."""
    if not path.is_file():
        raise FileNotFoundError(f"Delivery dataset was not found: {path}")
    frame = pd.read_csv(path, dtype="string", keep_default_na=False)
    missing_columns = sorted(REQUIRED_COLUMNS.difference(frame.columns))
    if missing_columns:
        raise ValueError(f"Dataset is missing required columns: {', '.join(missing_columns)}")
    return frame


def sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for provenance checks."""
    digest = hashlib.sha256()
    with path.open("rb") as file_handle:
        for chunk in iter(lambda: file_handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
