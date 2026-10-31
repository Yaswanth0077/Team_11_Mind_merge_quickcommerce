"""Project-wide paths, data schema, and prediction policy."""

from pathlib import Path
from typing import Final

PROJECT_ROOT: Final[Path] = Path(__file__).resolve().parent
RAW_DATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "raw" / "train.csv"
PROCESSED_DATA_PATH: Final[Path] = PROJECT_ROOT / "data" / "processed" / "cleaned_orders.csv"
REPORT_DIR: Final[Path] = PROJECT_ROOT / "reports"
MANAGEMENT_INSIGHTS_REPORT_PATH: Final[Path] = REPORT_DIR / "management_insights.json"
MANAGEMENT_REPORT_MARKDOWN_PATH: Final[Path] = REPORT_DIR / "management_report.md"
ARTIFACTS_DIR: Final[Path] = PROJECT_ROOT / "artifacts"
MODEL_DIR: Final[Path] = ARTIFACTS_DIR / "models"
METRICS_DIR: Final[Path] = ARTIFACTS_DIR / "metrics"
REGRESSION_MODEL_PATH: Final[Path] = MODEL_DIR / "delivery_time_regressor.joblib"
REGRESSION_REPORT_PATH: Final[Path] = METRICS_DIR / "regression_results.json"
REGRESSION_PREDICTIONS_PATH: Final[Path] = METRICS_DIR / "regression_test_predictions.csv"
REGRESSION_IMPORTANCE_PATH: Final[Path] = METRICS_DIR / "regression_feature_importance.csv"
CLASSIFICATION_MODEL_PATH: Final[Path] = MODEL_DIR / "late_order_classifier.joblib"
CLASSIFICATION_REPORT_PATH: Final[Path] = METRICS_DIR / "classification_results.json"
CLASSIFICATION_TEST_PREDICTIONS_PATH: Final[Path] = (
    METRICS_DIR / "classification_test_predictions.csv"
)
CLASSIFICATION_VALIDATION_PREDICTIONS_PATH: Final[Path] = (
    METRICS_DIR / "classification_validation_predictions.csv"
)
CLASSIFICATION_IMPORTANCE_PATH: Final[Path] = (
    METRICS_DIR / "classification_feature_importance.csv"
)
DARK_STORE_MODEL_PATH: Final[Path] = MODEL_DIR / "demand_zone_kmeans.joblib"
DARK_STORE_REPORT_PATH: Final[Path] = METRICS_DIR / "demand_zone_results.json"
DARK_STORE_ASSIGNMENTS_PATH: Final[Path] = METRICS_DIR / "demand_zone_assignments.csv"
DARK_STORE_RESTAURANTS_PATH: Final[Path] = METRICS_DIR / "demand_zone_restaurants.csv"
RIDER_SEGMENT_MODEL_PATH: Final[Path] = MODEL_DIR / "rider_performance_kmeans.joblib"
RIDER_SEGMENT_REPORT_PATH: Final[Path] = METRICS_DIR / "rider_segmentation_results.json"
RIDER_SEGMENT_ASSIGNMENTS_PATH: Final[Path] = METRICS_DIR / "rider_segment_assignments.csv"
RIDER_ORDER_ASSIGNMENTS_PATH: Final[Path] = METRICS_DIR / "rider_order_assignments.csv"

TARGET_COLUMN: Final[str] = "Time_taken(min)"
ORDER_ID_COLUMN: Final[str] = "ID"
RIDER_ID_COLUMN: Final[str] = "Delivery_person_ID"
SLA_THRESHOLD_MINUTES: Final[int] = 30  # Provisional; the challenge gives no official SLA.
SUSPICIOUS_DISTANCE_KM: Final[float] = 100.0  # Review threshold, not a deletion rule.
LATE_LABEL_COLUMN: Final[str] = "is_late_order"
DISTANCE_COLUMN: Final[str] = "delivery_distance_km"

COORDINATE_COLUMNS: Final[dict[str, str]] = {
    "store_latitude": "Restaurant_latitude",
    "store_longitude": "Restaurant_longitude",
    "customer_latitude": "Delivery_location_latitude",
    "customer_longitude": "Delivery_location_longitude",
}
MISSING_TOKENS: Final[frozenset[str]] = frozenset({"nan", "na", "n/a", "null", "none"})
DATE_FORMAT: Final[str] = "%d-%m-%Y"
TIME_FORMAT: Final[str] = "%H:%M:%S"

# Broad India envelope: used to flag implausible coordinates in this India-focused dataset.
INDIA_BOUNDS: Final[dict[str, tuple[float, float]]] = {
    "latitude": (6.0, 38.0),
    "longitude": (68.0, 98.0),
}
MIN_RIDER_AGE_YEARS: Final[int] = 18
RATING_BOUNDS: Final[tuple[float, float]] = (1.0, 5.0)
PEAK_HOUR_WINDOWS: Final[tuple[tuple[int, int], ...]] = ((11, 14), (17, 22))
DISTANCE_BAND_EDGES_KM: Final[tuple[float, ...]] = (0.0, 3.0, 7.0, 12.0, float("inf"))
DISTANCE_BAND_LABELS: Final[tuple[str, ...]] = ("0-3 km", "3-7 km", "7-12 km", "12+ km")

# The feature selector uses this allowlist; unlisted columns are never model inputs.
PRE_DELIVERY_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    DISTANCE_COLUMN,
    "order_hour",
    "order_day",
    "day_of_week",
    "is_weekend",
    "is_peak_hour",
    "distance_band",
    "weather_normalized",
    "traffic_normalized",
    "Festival",
    "City",
    "Type_of_order",
)
REGRESSION_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    DISTANCE_COLUMN,
    "order_hour",
    "order_day",
    "day_of_week",
    "is_weekend",
    "is_peak_hour",
    "distance_band",
    "City",
    "Type_of_order",
)
REGRESSION_NUMERIC_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    DISTANCE_COLUMN,
    "order_hour",
    "order_day",
    "day_of_week",
    "is_weekend",
    "is_peak_hour",
)
REGRESSION_CATEGORICAL_FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "distance_band",
    "City",
    "Type_of_order",
)
REGRESSION_FEATURE_POLICY: Final[dict[str, str]] = {
    DISTANCE_COLUMN: (
        "Included only when store and customer coordinates are available at order placement; "
        "invalid regional coordinate pairs are missing."
    ),
    "order_hour": "Included when the order time is available at order placement.",
    "order_day": "Included: derived from the order date.",
    "day_of_week": "Included: derived from the order date.",
    "is_weekend": "Included: derived from the order date.",
    "is_peak_hour": "Included: configured time-window heuristic derived from order time.",
    "distance_band": "Included: derived from validated store-to-customer distance.",
    "City": "Included: destination city is part of order information.",
    "Type_of_order": "Included: order category is known from the submitted order.",
    "weather_normalized": (
        "Excluded: dataset does not establish that weather is available at placement."
    ),
    "traffic_normalized": (
        "Excluded: dataset does not establish that traffic is available at placement."
    ),
    "Festival": "Excluded: dataset does not establish schedule availability at placement.",
    "Delivery_person_ID": "Excluded: rider assignment timing is unknown.",
    "Delivery_person_Age": "Excluded: rider assignment timing is unknown.",
    "Delivery_person_Ratings": "Excluded: rating availability/assignment timing is unknown.",
    "Vehicle_condition": "Excluded: vehicle assignment timing is unknown.",
    "Type_of_vehicle": "Excluded: vehicle assignment timing is unknown.",
    "multiple_deliveries": "Excluded: assignment/load timing is unknown.",
    ORDER_ID_COLUMN: "Excluded: identifier, not a predictor.",
    "Time_Order_picked": "Excluded: observed after order placement.",
    TARGET_COLUMN: "Excluded from predictors: regression target.",
    LATE_LABEL_COLUMN: "Excluded: derived from the delivery-time target.",
}
REGRESSION_RANDOM_SEED: Final[int] = 42
CLASSIFICATION_TARGET_COLUMN: Final[str] = "is_late"
CLASSIFICATION_RANDOM_SEED: Final[int] = REGRESSION_RANDOM_SEED
CLASSIFICATION_THRESHOLD_BETA: Final[float] = 2.0
CLASSIFICATION_FALSE_NEGATIVE_COST: Final[float | None] = None
CLASSIFICATION_FALSE_POSITIVE_COST: Final[float | None] = None
GEO_CLUSTER_RANDOM_SEED: Final[int] = 42
GEO_CLUSTER_COUNTS: Final[tuple[int, ...]] = tuple(range(2, 11))
GEO_CLUSTER_SILHOUETTE_SAMPLE_SIZE: Final[int] = 5000
RIDER_MIN_COMPLETED_DELIVERIES: Final[int] = 30
RIDER_MIN_RATING_OBSERVATIONS: Final[int] = 5
RIDER_CLUSTER_RANDOM_SEED: Final[int] = 42
RIDER_CLUSTER_COUNTS: Final[tuple[int, ...]] = tuple(range(2, 9))
RIDER_CLUSTER_SILHOUETTE_SAMPLE_SIZE: Final[int] = 2000
RIDER_MIN_SEGMENT_RIDERS: Final[int] = 30
RIDER_ELBOW_SILHOUETTE_TOLERANCE: Final[float] = 0.01
FEATURE_AVAILABILITY_POLICY: Final[dict[str, str]] = {
    "delivery_distance_km": "Eligible if store/customer coordinates are known at placement and pass geographic validation.",
    "order_hour": "Eligible if the order time is known at placement; missing/malformed values remain missing.",
    "order_day": "Eligible: derived from the order date.",
    "day_of_week": "Eligible: derived from the order date; Monday=0 through Sunday=6.",
    "is_weekend": "Eligible: derived from the order date.",
    "is_peak_hour": "Heuristic windows 11:00-13:59 and 17:00-21:59; confirm with operations.",
    "distance_band": "Eligible: derived from validated store-to-customer distance.",
    "weather_normalized": "Conditionally eligible: use only if weather is available at order placement.",
    "traffic_normalized": "Conditionally eligible: use only if traffic is available at order placement.",
    "Festival": "Conditionally eligible: use only if the schedule is known at order placement.",
    "City": "Eligible if known from the order destination at placement.",
    "Type_of_order": "Eligible if known from the submitted order at placement.",
    ORDER_ID_COLUMN: "Excluded: order identifier, not a predictive input.",
    "Delivery_person_ID": "Excluded: rider identifier; rider assignment timing is unknown.",
    "Delivery_person_Age": "Excluded: rider assignment timing is unknown.",
    "Delivery_person_Ratings": "Excluded: rider assignment timing is unknown.",
    "Vehicle_condition": "Excluded: vehicle assignment timing is unknown.",
    "Type_of_vehicle": "Excluded: vehicle assignment timing is unknown.",
    "multiple_deliveries": "Excluded: assignment/load timing is unknown.",
    "Time_Order_picked": "Excluded: known only after order placement.",
    TARGET_COLUMN: "Excluded from inputs: delivery outcome target.",
    LATE_LABEL_COLUMN: "Excluded from inputs: label derived from the delivery outcome.",
}
