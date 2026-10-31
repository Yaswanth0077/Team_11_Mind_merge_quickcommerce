"""Dataset-level quality checks and human-readable audit report generation."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from config import (
    COORDINATE_COLUMNS,
    DATE_FORMAT,
    FEATURE_AVAILABILITY_POLICY,
    INDIA_BOUNDS,
    LATE_LABEL_COLUMN,
    MIN_RIDER_AGE_YEARS,
    MISSING_TOKENS,
    ORDER_ID_COLUMN,
    PRE_DELIVERY_FEATURE_COLUMNS,
    RATING_BOUNDS,
    RIDER_ID_COLUMN,
    SLA_THRESHOLD_MINUTES,
    SUSPICIOUS_DISTANCE_KM,
    TARGET_COLUMN,
    TIME_FORMAT,
)
from src.data.cleaning import numeric_range_flags
from src.data.feature_engineering import (
    coordinate_validity,
    label_late_orders,
    outlier_flags,
    parse_timestamps,
)
from src.data.loader import sha256_file

PERCENTILES = (0.01, 0.25, 0.5, 0.75, 0.95, 0.99)


def _count_true(values: pd.Series) -> int:
    return int(values.fillna(False).sum())


def _numeric_distribution(values: pd.Series) -> dict[str, Any]:
    numeric = pd.to_numeric(values, errors="coerce").dropna()
    if numeric.empty:
        return {"count": 0, "minimum": None, "maximum": None, "percentiles": {}}
    quantiles = numeric.quantile(PERCENTILES)
    return {
        "count": int(numeric.size),
        "minimum": float(numeric.min()),
        "maximum": float(numeric.max()),
        "percentiles": {
            f"p{int(percentile * 100):02d}": float(value)
            for percentile, value in quantiles.items()
        },
    }


def _outlier_summary(values: pd.Series) -> dict[str, float | int | None]:
    _, _, statistics = outlier_flags(values)
    return statistics


def _missing_value_table(raw: pd.DataFrame, cleaned: pd.DataFrame) -> list[dict[str, Any]]:
    rows = []
    for column in raw.columns:
        values = raw[column].astype("string")
        stripped = values.str.strip()
        actual_nulls = int(values.isna().sum())
        empty_strings = int(values.eq("").fillna(False).sum())
        whitespace_only = int((values.notna() & values.ne("") & stripped.eq("")).sum())
        tokens = int(stripped.str.casefold().isin(MISSING_TOKENS).sum())
        placeholders = 0
        if column == "Weatherconditions":
            placeholders = int(
                stripped.str.fullmatch(r"conditions\s+nan", case=False, na=False).sum()
            )
        total = int(cleaned[column].isna().sum())
        rows.append(
            {
                "column": column,
                "actual_null_values": actual_nulls,
                "empty_strings": empty_strings,
                "whitespace_only_values": whitespace_only,
                "string_missing_tokens": tokens,
                "placeholder_missing_tokens": placeholders,
                "normalized_missing_values": total,
                "missing_percentage": round(100.0 * total / len(cleaned), 4)
                if len(cleaned)
                else 0.0,
            }
        )
    return rows


def _timestamp_audit(cleaned: pd.DataFrame) -> dict[str, dict[str, Any]]:
    parsed = parse_timestamps(cleaned)
    formats = {
        "Order_Date": DATE_FORMAT,
        "Time_Orderd": TIME_FORMAT,
        "Time_Order_picked": TIME_FORMAT,
    }
    results: dict[str, dict[str, Any]] = {}
    for column, parsed_values in parsed.items():
        source = cleaned[column].astype("string").str.strip()
        missing = source.isna() | source.eq("")
        malformed = (~missing) & parsed_values.isna()
        valid = parsed_values.dropna()
        if column == "Order_Date":
            parsed_minimum = str(valid.min()) if not valid.empty else None
            parsed_maximum = str(valid.max()) if not valid.empty else None
        else:
            parsed_minimum = valid.dt.strftime("%H:%M:%S").min() if not valid.empty else None
            parsed_maximum = valid.dt.strftime("%H:%M:%S").max() if not valid.empty else None
        results[column] = {
            "expected_format": formats[column],
            "missing_count": int(missing.sum()),
            "malformed_nonmissing_count": int(malformed.sum()),
            "valid_count": int(valid.size),
            "parsed_minimum": parsed_minimum,
            "parsed_maximum": parsed_maximum,
        }
    return results


def _numeric_validation(cleaned: pd.DataFrame) -> dict[str, Any]:
    flags = numeric_range_flags(cleaned)
    age = pd.to_numeric(cleaned["Delivery_person_Age"], errors="coerce")
    rating = pd.to_numeric(cleaned["Delivery_person_Ratings"], errors="coerce")
    multiple = pd.to_numeric(cleaned["multiple_deliveries"], errors="coerce")
    vehicle = pd.to_numeric(cleaned["Vehicle_condition"], errors="coerce")
    delivery = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce")
    return {
        "rider_age": {
            **_numeric_distribution(age),
            "review_minimum_inclusive": MIN_RIDER_AGE_YEARS,
            "suspicious_count": _count_true(flags["rider_age_suspicious"]),
        },
        "rider_rating": {
            **_numeric_distribution(rating),
            "valid_scale_inclusive": list(RATING_BOUNDS),
            "out_of_range_count": _count_true(flags["rating_out_of_range"]),
        },
        "multiple_deliveries": {
            **_numeric_distribution(multiple),
            "observed_values": sorted(int(value) for value in multiple.dropna().unique()),
            "invalid_negative_or_noninteger_count": _count_true(
                flags["multiple_deliveries_invalid"]
            ),
        },
        "vehicle_condition": {
            **_numeric_distribution(vehicle),
            "observed_values": sorted(int(value) for value in vehicle.dropna().unique()),
            "invalid_negative_or_noninteger_count": _count_true(
                flags["vehicle_condition_invalid"]
            ),
            "note": "The dataset does not define the meaning or valid scale; values are reported without imposing one.",
        },
        "delivery_time_minutes": {
            **_numeric_distribution(delivery),
            "invalid_nonpositive_or_noninteger_count": _count_true(
                flags["delivery_time_invalid"]
            ),
        },
    }


def _coordinate_audit(enriched: pd.DataFrame) -> dict[str, Any]:
    range_valid, region_valid, zero_pair = coordinate_validity(enriched)
    coordinates = {}
    negative_cells = 0
    store_zero_pair = (
        pd.to_numeric(enriched[COORDINATE_COLUMNS["store_latitude"]], errors="coerce").eq(0)
        & pd.to_numeric(enriched[COORDINATE_COLUMNS["store_longitude"]], errors="coerce").eq(0)
    )
    customer_zero_pair = (
        pd.to_numeric(
            enriched[COORDINATE_COLUMNS["customer_latitude"]], errors="coerce"
        ).eq(0)
        & pd.to_numeric(
            enriched[COORDINATE_COLUMNS["customer_longitude"]], errors="coerce"
        ).eq(0)
    )
    for label, column in COORDINATE_COLUMNS.items():
        values = pd.to_numeric(enriched[column], errors="coerce")
        coordinates[label] = {
            "zero_count": int(values.eq(0).sum()),
            "negative_count": int(values.lt(0).sum()),
            "outside_global_range_count": int(
                ((values.abs() > (90 if "latitude" in label else 180))).sum()
            ),
            "minimum": float(values.min()) if values.notna().any() else None,
            "maximum": float(values.max()) if values.notna().any() else None,
        }
        negative_cells += int(values.lt(0).sum())

    distances = pd.to_numeric(enriched["delivery_distance_km"], errors="coerce")
    raw_distances = pd.to_numeric(
        enriched["raw_haversine_distance_km"], errors="coerce"
    )
    iqr_flag, percentile_flag, outlier_statistics = outlier_flags(distances)
    return {
        "global_coordinate_range": {
            "latitude": [-90, 90],
            "longitude": [-180, 180],
            "invalid_pair_count": int((~range_valid).sum()),
        },
        "expected_region_envelope": {
            "latitude": list(INDIA_BOUNDS["latitude"]),
            "longitude": list(INDIA_BOUNDS["longitude"]),
            "region_valid_pair_count": int(region_valid.sum()),
            "geographically_implausible_pair_count": int((~region_valid).sum()),
            "outside_region_nonzero_pair_count": int((~region_valid & ~zero_pair).sum()),
        },
        "coordinates_by_column": coordinates,
        "rows_with_any_negative_coordinate": int(
            pd.concat(
                [
                    pd.to_numeric(enriched[column], errors="coerce").lt(0)
                    for column in COORDINATE_COLUMNS.values()
                ],
                axis=1,
            ).any(axis=1).sum()
        ),
        "negative_coordinate_cells": negative_cells,
        "zero_coordinate_pair_count": _count_true(zero_pair),
        "zero_store_coordinate_pair_count": _count_true(store_zero_pair),
        "zero_customer_coordinate_pair_count": _count_true(customer_zero_pair),
        "distance_km": {
            **_numeric_distribution(distances),
            "model_eligible_pair_count": int(distances.notna().sum()),
            "not_model_eligible_pair_count": int(distances.isna().sum()),
            "haversine_computed_pairs": int(raw_distances.notna().sum()),
            "over_100_km_count": int(
                (raw_distances > SUSPICIOUS_DISTANCE_KM).sum()
            ),
            "iqr_outliers": _count_true(iqr_flag),
            "percentile_outliers": _count_true(percentile_flag),
            "outlier_statistics": outlier_statistics,
            "review_distance_threshold_km": SUSPICIOUS_DISTANCE_KM,
            "distance_band_definition": "0-3, 3-7, 7-12, and 12+ km; bands are descriptive.",
        },
    }


def add_quality_flags(enriched: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Attach review-only numeric and outlier flags; never remove records."""
    result = enriched.copy()
    decisions: list[dict[str, Any]] = []
    numeric_flags = numeric_range_flags(result)
    for name, flag in numeric_flags.items():
        result[name] = flag.astype("boolean")
        decisions.append(
            {
                "operation": f"Flag {name}",
                "reason": "Preserve the source value while making records requiring validation explicit.",
                "affected_values": _count_true(flag),
                "affected_records": _count_true(flag),
                "records_retained": len(result),
                "records_excluded": 0,
            }
        )

    distance_flag = result["distance_over_review_threshold"].astype("boolean")
    decisions.append(
        {
            "operation": f"Flag distances over {SUSPICIOUS_DISTANCE_KM:g} km",
            "reason": "Review unusually long coordinate pairs; the threshold is a screening heuristic and no records are removed.",
            "affected_values": _count_true(distance_flag),
            "affected_records": _count_true(distance_flag),
            "records_retained": len(result),
            "records_excluded": 0,
        }
    )

    for field, column in (
        ("delivery_time", TARGET_COLUMN),
        ("distance", "delivery_distance_km"),
    ):
        iqr_flag, percentile_flag, statistics = outlier_flags(result[column])
        result[f"{field}_iqr_outlier"] = iqr_flag
        result[f"{field}_percentile_outlier"] = percentile_flag
        for method, flag_key in (
            ("IQR", f"{field}_iqr_outlier"),
            ("1st/99th percentile", f"{field}_percentile_outlier"),
        ):
            flag = result[flag_key]
            decisions.append(
                {
                    "operation": f"Flag {method} {field} outliers",
                    "reason": "Identify statistically unusual observations for review; do not automatically discard them.",
                    "affected_values": _count_true(flag),
                    "affected_records": _count_true(flag),
                    "records_retained": len(result),
                    "records_excluded": 0,
                    "bounds_and_counts": {
                        key: value
                        for key, value in statistics.items()
                        if key != "percentile_flag"
                    },
                }
            )
    return result, {"decisions": decisions}


def build_data_quality_report(
    raw: pd.DataFrame,
    cleaned: pd.DataFrame,
    enriched: pd.DataFrame,
    cleaning_log: list[dict[str, Any]],
    flag_log: dict[str, Any],
    input_path: Path,
    sla_threshold_minutes: int = SLA_THRESHOLD_MINUTES,
) -> dict[str, Any]:
    """Create a structured audit report with observed values and explicit limitations."""
    duplicate_order_ids = raw[ORDER_ID_COLUMN].duplicated(keep=False)
    duplicate_rows = raw.duplicated(keep=False)
    inferred_source = pd.read_csv(input_path, low_memory=False)
    delivery_minutes = pd.to_numeric(cleaned[TARGET_COLUMN], errors="coerce")
    late_labels = label_late_orders(delivery_minutes, sla_threshold_minutes)
    column_profiles = [
        {
            "column": column,
            "raw_preservation_dtype": str(raw[column].dtype),
            "inferred_csv_dtype": str(inferred_source[column].dtype),
            "cleaned_dtype": str(cleaned[column].dtype),
            "unique_values_including_missing": int(raw[column].nunique(dropna=False)),
            "memory_usage_bytes": int(raw[column].memory_usage(index=False, deep=True)),
        }
        for column in raw.columns
    ]

    return {
        "dataset_overview": {
            "source_path": str(input_path),
            "source_sha256": sha256_file(input_path),
            "rows": int(len(raw)),
            "columns": int(len(raw.columns)),
            "column_names": list(raw.columns),
            "loaded_dataframe_memory_bytes": int(
                raw.memory_usage(index=True, deep=True).sum()
            ),
            "column_profiles": column_profiles,
            "raw_preservation_dtype_summary": raw.dtypes.astype(str).to_dict(),
            "inferred_csv_dtype_summary": inferred_source.dtypes.astype(str).to_dict(),
            "cleaned_dtype_summary": cleaned.dtypes.astype(str).to_dict(),
        },
        "missing_values": _missing_value_table(raw, cleaned),
        "duplicate_analysis": {
            "duplicate_full_rows": int(raw.duplicated().sum()),
            "rows_involved_in_duplicate_full_rows": int(duplicate_rows.sum()),
            "duplicate_order_id_rows_after_first": int(
                raw[ORDER_ID_COLUMN].duplicated().sum()
            ),
            "rows_involved_in_duplicate_order_ids": int(duplicate_order_ids.sum()),
            "duplicate_order_id_values": int(
                raw.loc[duplicate_order_ids, ORDER_ID_COLUMN].nunique()
            ),
        },
        "numeric_validation": _numeric_validation(cleaned),
        "coordinate_validation": _coordinate_audit(enriched),
        "timestamp_validation": _timestamp_audit(cleaned),
        "outlier_analysis": {
            "delivery_time_minutes": {
                **_numeric_distribution(cleaned[TARGET_COLUMN]),
                "iqr_and_percentile_flags": _outlier_summary(
                    cleaned[TARGET_COLUMN]
                ),
            },
            "distance_km": {
                **_numeric_distribution(enriched["delivery_distance_km"]),
                "iqr_and_percentile_flags": _outlier_summary(
                    enriched["delivery_distance_km"]
                ),
                "haversine_computed_pairs": int(
                    enriched["raw_haversine_distance_km"].notna().sum()
                ),
                "pairs_over_review_threshold": int(
                    enriched["distance_over_review_threshold"].fillna(False).sum()
                ),
                "review_threshold_km": SUSPICIOUS_DISTANCE_KM,
            },
            "flag_policy": "IQR outside Q1-1.5*IQR/Q3+1.5*IQR and values strictly below p01 or above p99 are flagged; all records are retained.",
        },
        "cleaning_decisions": cleaning_log + flag_log["decisions"],
        "records": {
            "input_records": int(len(raw)),
            "retained_records": int(len(enriched)),
            "excluded_records": int(len(raw) - len(enriched)),
            "imputed_values": 0,
            "row_deduplication_applied": False,
        },
        "delay_lens": {
            "target_column": TARGET_COLUMN,
            "target_unit": "minutes",
            "sla_threshold_minutes": int(sla_threshold_minutes),
            "sla_status": "provisional; challenge brief does not confirm an official threshold",
            "late_definition": "delivery_time_minutes > SLA threshold; equality is not late",
            "prediction_feature_columns": list(PRE_DELIVERY_FEATURE_COLUMNS),
            "excluded_columns": [
                ORDER_ID_COLUMN,
                RIDER_ID_COLUMN,
                "Delivery_person_Age",
                "Delivery_person_Ratings",
                "Vehicle_condition",
                "Type_of_vehicle",
                "multiple_deliveries",
                "Time_Order_picked",
                TARGET_COLUMN,
                LATE_LABEL_COLUMN,
            ],
            "conditional_availability": [
                "Weather and traffic must be available at placement time.",
                "Festival must be known at placement time.",
                "Coordinates must be known at placement time and pass geographic validation.",
            ],
            "observed_outcomes": {
                "late_orders": _count_true(late_labels),
                "not_late_orders": _count_true(~late_labels),
                "missing_delivery_times": int(delivery_minutes.isna().sum()),
            },
            "feature_availability_policy": FEATURE_AVAILABILITY_POLICY,
        },
        "cleaning_policy": {
            "raw_preserved": True,
            "missing_values_imputed": False,
            "suspicious_values_deleted": False,
            "outlier_values_deleted": False,
            "invalid_coordinate_distances": "raw Haversine distances are retained for audit; model distance is missing unless both points fall within the broad India envelope",
            "reason": "Defer imputation and row exclusion until business semantics and training-only preprocessing are established.",
        },
        "limitations": [
            "The supplied file is a food-delivery dataset used as a quick-commerce proxy, not verified grocery dark-store operations data.",
            "The 30-minute SLA is provisional and requires organizer/business confirmation.",
            "Rider ages below 18 are flagged for review; 18 is a screening assumption and no maximum age is imposed.",
            "Rider tenure/experience is not present and is not fabricated.",
            "Rider/vehicle assignment timing is unknown; rider-specific attributes and multiple-delivery load are excluded from order-placement features.",
            "Weather, traffic, and festival fields are eligible only if known when a prediction is made.",
            "The broad India coordinate envelope is a review heuristic; flagged records are retained and require source confirmation.",
            "No imputation or duplicate removal has been performed.",
        ],
    }


def write_report(report: dict[str, Any], report_dir: Path) -> tuple[Path, Path]:
    """Write JSON and Markdown representations of the audit."""
    report_dir.mkdir(parents=True, exist_ok=True)
    json_path = report_dir / "data_quality_report.json"
    markdown_path = report_dir / "data_quality_report.md"
    import json

    json_path.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")
    markdown_path.write_text(render_markdown_report(report), encoding="utf-8")
    return json_path, markdown_path


def render_markdown_report(report: dict[str, Any]) -> str:
    """Render the structured report as a readable, self-contained Markdown document."""
    overview = report["dataset_overview"]
    lines = [
        "# Quick-Commerce Delivery Intelligence — Data Quality Report",
        "",
        "## 1. Dataset overview",
        f"- Source: `{overview['source_path']}`",
        f"- SHA-256: `{overview['source_sha256']}`",
        f"- Shape: {overview['rows']:,} rows × {overview['columns']} columns",
        f"- Loaded DataFrame memory use: {overview['loaded_dataframe_memory_bytes']:,} bytes",
        "",
        "| Column | Inferred CSV dtype | Cleaned dtype | Unique values |",
        "|---|---|---|---:|",
    ]
    for profile in overview["column_profiles"]:
        lines.append(
            f"| `{profile['column']}` | {profile['inferred_csv_dtype']} | "
            f"{profile['cleaned_dtype']} | {profile['unique_values_including_missing']:,} |"
        )
    lines += ["", "## 2. Missing-value investigation", ""]
    lines += [
        "| Column | Null | Empty | Whitespace | Missing tokens | Placeholders | Normalized missing | % |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in report["missing_values"]:
        lines.append(
            f"| `{item['column']}` | {item['actual_null_values']} | {item['empty_strings']} | "
            f"{item['whitespace_only_values']} | {item['string_missing_tokens']} | "
            f"{item['placeholder_missing_tokens']} | {item['normalized_missing_values']} | "
            f"{item['missing_percentage']:.2f} |"
        )
    lines += [
        "",
        "## 3. Duplicate analysis",
        "",
        f"- Exact duplicate rows: {report['duplicate_analysis']['duplicate_full_rows']:,}",
        f"- Duplicate order-ID rows after first occurrence: {report['duplicate_analysis']['duplicate_order_id_rows_after_first']:,}",
        f"- Rows involved in duplicate order IDs: {report['duplicate_analysis']['rows_involved_in_duplicate_order_ids']:,}",
        "",
        "## 4. Numeric validation",
        "",
        "| Field | Valid values | Minimum | Maximum | Values flagged |",
        "|---|---:|---:|---:|---:|",
    ]
    numeric = report["numeric_validation"]
    for field, summary, count_key in (
        ("Rider age", numeric["rider_age"], "suspicious_count"),
        ("Rider rating", numeric["rider_rating"], "out_of_range_count"),
        (
            "Multiple deliveries",
            numeric["multiple_deliveries"],
            "invalid_negative_or_noninteger_count",
        ),
        (
            "Vehicle condition",
            numeric["vehicle_condition"],
            "invalid_negative_or_noninteger_count",
        ),
        (
            "Delivery time (minutes)",
            numeric["delivery_time_minutes"],
            "invalid_nonpositive_or_noninteger_count",
        ),
    ):
        lines.append(
            f"| {field} | {summary['count']:,} | {summary['minimum']} | "
            f"{summary['maximum']} | {summary[count_key]:,} |"
        )
    coordinates = report["coordinate_validation"]
    region = coordinates["expected_region_envelope"]
    distance = coordinates["distance_km"]
    lines += [
        "",
        "## 5. Geographic validation",
        "",
        f"- Zero-coordinate pairs: {coordinates['zero_coordinate_pair_count']:,}.",
        f"- Zero store pairs: {coordinates['zero_store_coordinate_pair_count']:,}; "
        f"zero customer pairs: {coordinates['zero_customer_coordinate_pair_count']:,}.",
        f"- Pairs within global latitude/longitude ranges: "
        f"{overview['rows'] - coordinates['global_coordinate_range']['invalid_pair_count']:,} "
        f"of {overview['rows']:,}.",
        f"- Pairs within the broad India review envelope: {region['region_valid_pair_count']:,}; "
        f"flagged outside it: {region['geographically_implausible_pair_count']:,}.",
        "",
        "| Coordinate field | Zero values | Negative values | Outside global range |",
        "|---|---:|---:|---:|",
    ]
    for field, counts in coordinates["coordinates_by_column"].items():
        lines.append(
            f"| `{field}` | {counts['zero_count']:,} | {counts['negative_count']:,} | "
            f"{counts['outside_global_range_count']:,} |"
        )
    lines += [
        "",
        f"- Haversine distance computed for {distance['haversine_computed_pairs']:,} "
        f"globally in-range pairs. {distance['over_100_km_count']:,} exceeded the "
        f"{distance['review_distance_threshold_km']:g} km review threshold.",
        f"- Model-eligible distance range: {distance['minimum']}–{distance['maximum']} km; "
        f"median {distance['percentiles'].get('p50')} km. Distances for pairs outside "
        "the regional envelope remain available separately for audit, not model input.",
        "",
        "## 6. Timestamp validation",
        "",
        "| Field | Expected format | Missing | Malformed nonmissing | Valid | Parsed range |",
        "|---|---|---:|---:|---:|---|",
    ]
    for column, details in report["timestamp_validation"].items():
        lines.append(
            f"| `{column}` | `{details['expected_format']}` | {details['missing_count']:,} | "
            f"{details['malformed_nonmissing_count']:,} | {details['valid_count']:,} | "
            f"{details['parsed_minimum']} to {details['parsed_maximum']} |"
        )
    lines += [
        "",
        "## 7. Outlier analysis",
        "",
        "| Measure | IQR bounds | IQR flags | 1st/99th percentiles | Percentile flags |",
        "|---|---|---:|---|---:|",
    ]
    for label, key in (
        ("Delivery time (minutes)", "delivery_time_minutes"),
        ("Model-eligible distance (km)", "distance_km"),
    ):
        summary = report["outlier_analysis"][key]
        flags = summary["iqr_and_percentile_flags"]
        bounds = (
            f"{flags['iqr_lower_bound']:.2f} to {flags['iqr_upper_bound']:.2f}"
            if flags["iqr_lower_bound"] is not None
            else "not available"
        )
        percentiles = (
            f"{flags['p01']:.2f}, {flags['p99']:.2f}"
            if flags["p01"] is not None
            else "not available"
        )
        lines.append(
            f"| {label} | {bounds} | {flags['iqr_outlier_count']:,} | "
            f"{percentiles} | {flags['percentile_outlier_count']:,} |"
        )
    lines += [
        "",
        f"Percentile columns show p01 and p99. {report['outlier_analysis']['flag_policy']}",
        "All flagged observations are retained.",
        "",
        "## 8. Cleaning decisions",
        "",
        "| Operation | Reason | Affected records | Retained | Excluded |",
        "|---|---|---:|---:|---:|",
    ]
    for decision in report["cleaning_decisions"]:
        reason = decision["reason"].replace("|", "\\|")
        lines.append(
            f"| {decision['operation']} | {reason} | {decision['affected_records']:,} | "
            f"{decision['records_retained']:,} | {decision['records_excluded']:,} |"
        )
    record_summary = report["records"]
    lines += [
        "",
        "## 9. Records retained and excluded",
        "",
        f"- Input: {record_summary['input_records']:,}",
        f"- Retained: {record_summary['retained_records']:,}",
        f"- Excluded: {record_summary['excluded_records']:,}",
        f"- Imputed values: {record_summary['imputed_values']:,}",
        "",
        "## Delay lens and leakage policy",
        "",
        f"- Target: `{report['delay_lens']['target_column']}` in minutes.",
        f"- SLA: **{report['delay_lens']['sla_threshold_minutes']} minutes (provisional)**.",
        f"- Late means delivery time **strictly greater** than the SLA.",
        f"- In this dataset, {report['delay_lens']['observed_outcomes']['late_orders']:,} "
        f"observed deliveries exceed this provisional threshold; this is a label count, "
        "not a model result.",
        "- `Time_Order_picked`, the delivery outcome, identifiers, and assignment-dependent rider/vehicle fields are excluded from order-placement model inputs.",
        "- Weather, traffic, festival, and coordinates require verified availability at prediction time.",
        "- Distance, date, and time source columns are transformed; quality flags are audit-only.",
        "",
        "| Field | Availability policy |",
        "|---|---|",
    ]
    for column, policy in report["delay_lens"]["feature_availability_policy"].items():
        safe_policy = policy.replace("|", "\\|")
        lines.append(f"| `{column}` | {safe_policy} |")
    lines += [
        "",
        "## 10. Remaining limitations",
        "",
    ]
    lines.extend(f"- {limitation}" for limitation in report["limitations"])
    return "\n".join(lines) + "\n"
