"""Auditable, non-destructive normalization and type conversion."""

from __future__ import annotations

import re
from typing import Any

import pandas as pd

from config import (
    MIN_RIDER_AGE_YEARS,
    MISSING_TOKENS,
    RATING_BOUNDS,
    TARGET_COLUMN,
)

INTEGER_COLUMNS = {
    "Delivery_person_Age",
    "Vehicle_condition",
    "multiple_deliveries",
    TARGET_COLUMN,
}
NUMERIC_COLUMN_DTYPES = {
    "Delivery_person_Age": "Int64",
    "Delivery_person_Ratings": "Float64",
    "Restaurant_latitude": "Float64",
    "Restaurant_longitude": "Float64",
    "Delivery_location_latitude": "Float64",
    "Delivery_location_longitude": "Float64",
    "Vehicle_condition": "Int64",
    "multiple_deliveries": "Int64",
    TARGET_COLUMN: "Int64",
}


def extract_delivery_time_minutes(values: pd.Series) -> pd.Series:
    """Extract integer minutes from values formatted like ``(min) 24``."""
    extracted = values.astype("string").str.extract(
        r"^\s*\(min\)\s*([+-]?\d+)\s*$", expand=False, flags=re.IGNORECASE
    )
    return pd.to_numeric(extracted, errors="coerce").astype("Int64")


def normalize_missing_values(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Convert actual nulls, blanks, known tokens, and weather placeholders to ``pd.NA``."""
    cleaned = frame.copy()
    original_null_cells = int(cleaned.isna().sum().sum())
    changed_by_column: dict[str, int] = {}
    affected_rows = pd.Series(False, index=cleaned.index)

    for column in cleaned.columns:
        values = cleaned[column].astype("string")
        stripped = values.str.strip()
        token_mask = stripped.str.casefold().isin(MISSING_TOKENS)
        placeholder_mask = pd.Series(False, index=cleaned.index)
        if column == "Weatherconditions":
            placeholder_mask = stripped.str.fullmatch(
                r"conditions\s+nan", case=False, na=False
            )
        missing_mask = values.isna() | stripped.eq("") | token_mask | placeholder_mask
        changed_mask = missing_mask & values.notna()
        count = int(changed_mask.sum())
        if count:
            cleaned.loc[missing_mask, column] = pd.NA
            changed_by_column[column] = count
            affected_rows |= changed_mask

    return cleaned, {
        "operation": "Normalize missing values",
        "reason": "Represent actual nulls, blank strings, known string tokens, and the weather 'conditions NaN' placeholder consistently.",
        "affected_values": int(sum(changed_by_column.values())),
        "affected_records": int(affected_rows.sum()),
        "affected_by_column": changed_by_column,
        "preexisting_null_values": original_null_cells,
        "records_retained": len(frame),
        "records_excluded": 0,
    }


def normalize_categories(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Trim string categories and remove the source's redundant weather prefix."""
    cleaned = frame.copy()
    changed_by_column: dict[str, int] = {}
    affected_rows = pd.Series(False, index=cleaned.index)
    for column in cleaned.columns:
        if not isinstance(cleaned[column].dtype, pd.StringDtype):
            continue
        before = cleaned[column].copy()
        after = before.str.strip()
        if column == "Weatherconditions":
            after = after.str.replace(r"^conditions\s+", "", regex=True, case=False)
        changed = before.notna() & before.ne(after)
        count = int(changed.sum())
        if count:
            cleaned[column] = after
            changed_by_column[column] = count
            affected_rows |= changed
    return cleaned, {
        "operation": "Trim categorical text and normalize weather labels",
        "reason": "Remove inconsistent surrounding whitespace and the repeated 'conditions ' prefix without changing category meaning.",
        "affected_values": int(sum(changed_by_column.values())),
        "affected_records": int(affected_rows.sum()),
        "affected_by_column": changed_by_column,
        "records_retained": len(frame),
        "records_excluded": 0,
    }


def convert_numeric_columns(frame: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Parse numeric fields; retain unparseable source values in the audit counts."""
    cleaned = frame.copy()
    affected_rows = pd.Series(False, index=cleaned.index)
    conversion_errors: dict[str, int] = {}
    converted_counts: dict[str, int] = {}
    for column, dtype in NUMERIC_COLUMN_DTYPES.items():
        if column not in cleaned.columns:
            continue
        original = cleaned[column].astype("string")
        if column == TARGET_COLUMN:
            parsed = extract_delivery_time_minutes(original)
        else:
            parsed = pd.to_numeric(original, errors="coerce")
            if column in INTEGER_COLUMNS:
                parsed = parsed.where(parsed.isna() | parsed.mod(1).eq(0))
        invalid_nonmissing = original.notna() & parsed.isna()
        conversion_errors[column] = int(invalid_nonmissing.sum())
        converted_counts[column] = int(parsed.notna().sum())
        affected_rows |= original.notna()
        cleaned[column] = parsed.astype(dtype)
    return cleaned, {
        "operation": "Convert numeric fields",
        "reason": "Parse rider, coordinate, vehicle, delivery-count, and delivery-time fields as nullable numeric types; record unparseable nonmissing values.",
        "affected_values": int(sum(converted_counts.values())),
        "affected_records": int(affected_rows.sum()),
        "converted_values_by_column": converted_counts,
        "unparseable_nonmissing_values_by_column": conversion_errors,
        "records_retained": len(frame),
        "records_excluded": 0,
    }


def clean_orders(frame: pd.DataFrame) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    """Apply conservative cleaning and return the data plus an operation log."""
    cleaned, missing_operation = normalize_missing_values(frame)
    cleaned, category_operation = normalize_categories(cleaned)
    cleaned, numeric_operation = convert_numeric_columns(cleaned)
    cleaning_log = [missing_operation, category_operation, numeric_operation]
    cleaning_log.append(
        {
            "operation": "Imputation and row exclusion",
            "reason": "No imputation or row deletion is justified in this foundation phase; preserve missing and suspicious records for review and model-specific handling.",
            "affected_values": 0,
            "affected_records": 0,
            "records_retained": len(frame),
            "records_excluded": 0,
        }
    )
    return cleaned, cleaning_log


def numeric_range_flags(frame: pd.DataFrame) -> dict[str, pd.Series]:
    """Create review flags without overwriting values or removing observations."""
    age = pd.to_numeric(frame["Delivery_person_Age"], errors="coerce")
    rating = pd.to_numeric(frame["Delivery_person_Ratings"], errors="coerce")
    multiple = pd.to_numeric(frame["multiple_deliveries"], errors="coerce")
    vehicle = pd.to_numeric(frame["Vehicle_condition"], errors="coerce")
    delivery = pd.to_numeric(frame[TARGET_COLUMN], errors="coerce")
    return {
        "rider_age_suspicious": age.notna()
        & age.lt(MIN_RIDER_AGE_YEARS),
        "rating_out_of_range": rating.notna()
        & ~rating.between(RATING_BOUNDS[0], RATING_BOUNDS[1]),
        "multiple_deliveries_invalid": multiple.notna()
        & ((multiple < 0) | (multiple % 1 != 0)),
        "vehicle_condition_invalid": vehicle.notna()
        & ((vehicle < 0) | (vehicle % 1 != 0)),
        "delivery_time_invalid": delivery.notna()
        & ((delivery <= 0) | (delivery % 1 != 0)),
    }
