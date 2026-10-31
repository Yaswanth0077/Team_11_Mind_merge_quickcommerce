"""Evidence checks and final management synthesis for PS8."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    brier_score_loss,
    confusion_matrix,
    f1_score,
    fbeta_score,
    mean_absolute_error,
    mean_squared_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)

from config import (
    CLASSIFICATION_IMPORTANCE_PATH,
    CLASSIFICATION_MODEL_PATH,
    CLASSIFICATION_REPORT_PATH,
    CLASSIFICATION_TEST_PREDICTIONS_PATH,
    DARK_STORE_REPORT_PATH,
    MANAGEMENT_INSIGHTS_REPORT_PATH,
    MANAGEMENT_REPORT_MARKDOWN_PATH,
    PROCESSED_DATA_PATH,
    PROJECT_ROOT,
    REGRESSION_IMPORTANCE_PATH,
    REGRESSION_MODEL_PATH,
    REGRESSION_PREDICTIONS_PATH,
    REGRESSION_REPORT_PATH,
    RIDER_SEGMENT_REPORT_PATH,
    SLA_THRESHOLD_MINUTES,
)
from src.data.loader import sha256_file
from src.models.regression import load_regression_dataset

METRIC_TOLERANCE = 1e-8


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise FileNotFoundError(f"Required PS8 evidence report is missing: {path}")
    with path.open(encoding="utf-8") as file_handle:
        return json.load(file_handle)


def _assert_close(name: str, actual: float, reported: float) -> None:
    if not math.isclose(actual, reported, rel_tol=METRIC_TOLERANCE, abs_tol=METRIC_TOLERANCE):
        raise ValueError(
            f"Saved artifact mismatch for {name}: recomputed {actual:.12g}, "
            f"reported {reported:.12g}."
        )


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records", double_precision=15))


def _verify_importance_report(
    report_rows: list[dict[str, Any]],
    importance: pd.DataFrame,
    metric_columns: tuple[str, ...],
    label: str,
) -> None:
    report_by_feature = {
        row["feature"]: row for row in report_rows if "feature" in row
    }
    if set(report_by_feature) != set(importance["feature"].astype(str)):
        raise ValueError(f"{label} report and CSV importance features do not match.")
    for row in importance.to_dict(orient="records"):
        expected = report_by_feature[str(row["feature"])]
        for column in metric_columns:
            _assert_close(
                f"{label} importance {row['feature']} {column}",
                float(row[column]),
                float(expected[column]),
            )


def _verify_regression_evidence(
    report: dict[str, Any],
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
    model_bundle: dict[str, Any],
    processed_data_path: Path,
) -> dict[str, Any]:
    if not {"actual_minutes", "predicted_minutes"}.issubset(predictions.columns):
        raise ValueError("Regression prediction artifact lacks actual/predicted minutes.")
    selected_name = report["metadata"]["selected_model"]
    metadata = model_bundle.get("metadata", {})
    if metadata.get("selected_model") != selected_name:
        raise ValueError("Regression model artifact does not match the saved report.")
    if report["metadata"]["training_configuration"].get("test_set_used_for_tuning") is not False:
        raise ValueError("Regression selection must not use held-out test results.")
    feature_columns = metadata.get("feature_columns", [])
    if "Time_Order_picked" in feature_columns:
        raise ValueError("Pickup time must not be used as an order-placement feature.")
    if "Time_Order_picked" in importance["feature"].astype(str).tolist():
        raise ValueError("Pickup time must not be reported as a regression predictor.")
    if int(metadata.get("sla_threshold_minutes", -1)) != SLA_THRESHOLD_MINUTES:
        raise ValueError("Regression report SLA does not match the shared configuration.")
    if metadata.get("dataset", {}).get("processed_sha256") != sha256_file(
        processed_data_path
    ):
        raise ValueError("Regression model artifact was trained on a different dataset.")

    pipeline = model_bundle.get("pipeline")
    if pipeline is None or "source_row_number" not in predictions:
        raise ValueError("Regression model or test-row provenance is unavailable.")
    actual = pd.to_numeric(predictions["actual_minutes"], errors="coerce").to_numpy()
    predicted = pd.to_numeric(predictions["predicted_minutes"], errors="coerce").to_numpy()
    dataset = load_regression_dataset(processed_data_path)
    row_positions = predictions["source_row_number"].map(
        pd.Series(
            np.arange(len(dataset.source_row_numbers)),
            index=dataset.source_row_numbers.astype("int64"),
        )
    )
    if row_positions.isna().any():
        raise ValueError("Regression test rows do not map to the processed dataset.")
    positions = row_positions.to_numpy(dtype="int64")
    expected_actual = dataset.target.iloc[positions].to_numpy(dtype=float)
    if not np.allclose(expected_actual, actual, rtol=0.0, atol=1e-10):
        raise ValueError("Regression test targets do not match the processed dataset.")
    features = dataset.features.iloc[positions].loc[:, feature_columns]
    reproduced = np.asarray(pipeline.predict(features), dtype=float)
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(predicted)):
        raise ValueError("Regression test predictions contain non-finite values.")
    if not np.allclose(reproduced, predicted, rtol=1e-7, atol=1e-8):
        raise ValueError("Regression predictions do not match the saved fitted model.")

    metrics = {
        "order_count": int(len(actual)),
        "mae_minutes": float(mean_absolute_error(actual, predicted)),
        "rmse_minutes": float(mean_squared_error(actual, predicted) ** 0.5),
        "r2": float(r2_score(actual, predicted)),
    }
    saved_metrics = report["selected_model_test_metrics"]
    for key in ("mae_minutes", "rmse_minutes", "r2"):
        _assert_close(f"regression test {key}", metrics[key], float(saved_metrics[key]))

    comparison = report["model_comparison"]
    if not any(row["model"] == selected_name for row in comparison):
        raise ValueError("Selected regression model is absent from the model comparison.")
    cv_selected_name = min(
        comparison,
        key=lambda row: (row["cv_mae_minutes"], row["model"]),
    )["model"]
    if cv_selected_name != selected_name:
        raise ValueError("Selected regression model does not match the saved CV criterion.")
    _verify_importance_report(
        report["feature_importance"],
        importance,
        ("mae_increase_minutes", "standard_deviation_minutes"),
        "Regression",
    )
    return {
        "selected_model": selected_name,
        "selection_criterion": report["metadata"]["training_configuration"][
            "selection_criterion"
        ],
        "model_comparison": comparison,
        "baseline_test_metrics": report["baseline_test_metrics"],
        "selected_test_metrics": {**saved_metrics, "order_count": metrics["order_count"]},
        "test_metrics_recomputed_from_predictions": metrics,
        "mae_improvement_vs_baseline_minutes": report[
            "selected_model_mae_improvement_minutes"
        ],
        "mae_improvement_vs_baseline_percent": report[
            "selected_model_mae_improvement_percent"
        ],
        "temporal_selected_model_metrics": report["temporal_selected_model_metrics"],
        "feature_importance": _records(importance),
        "interpretation": (
            "MAE is the average absolute error in minutes. RMSE is in minutes and "
            "penalizes larger misses more strongly. R-squared describes test-set "
            "variation explained relative to a mean baseline; it is not a causal or "
            "individual-order guarantee."
        ),
    }


def _verify_classification_evidence(
    report: dict[str, Any],
    predictions: pd.DataFrame,
    importance: pd.DataFrame,
    model_bundle: dict[str, Any],
    processed_data_path: Path,
) -> dict[str, Any]:
    selected = report["selected_model"]
    selected_name = selected["name"]
    threshold = float(selected["threshold"])
    metadata = model_bundle.get("metadata", {})
    if metadata.get("selected_model") != selected_name:
        raise ValueError("Classifier artifact does not match the saved evaluation report.")
    if not math.isclose(float(metadata.get("operating_threshold", math.nan)), threshold):
        raise ValueError("Classifier artifact threshold does not match the saved report.")
    if int(metadata.get("sla_threshold_minutes", -1)) != SLA_THRESHOLD_MINUTES:
        raise ValueError("Classification report SLA does not match shared configuration.")
    if metadata.get("processed_dataset", {}).get("sha256") != sha256_file(
        processed_data_path
    ):
        raise ValueError("Classifier model artifact was trained on a different dataset.")
    if metadata["training_configuration"].get(
        "test_set_used_for_tuning_or_threshold_selection"
    ) is not False:
        raise ValueError("Classification test labels must not affect model or threshold selection.")
    if "Time_Order_picked" in metadata.get("feature_columns", []):
        raise ValueError("Pickup time must not be used as an order-placement feature.")
    if "Time_Order_picked" in importance["feature"].astype(str).tolist():
        raise ValueError("Pickup time must not be reported as a classifier predictor.")

    required = {"actual_is_late", "risk_score", "predicted_is_late"}
    if not required.issubset(predictions.columns):
        raise ValueError("Classification test predictions are missing required columns.")
    actual = pd.to_numeric(predictions["actual_is_late"], errors="coerce").to_numpy()
    scores = pd.to_numeric(predictions["risk_score"], errors="coerce").to_numpy()
    predicted = pd.to_numeric(predictions["predicted_is_late"], errors="coerce").to_numpy()
    if not np.all(np.isfinite(actual)) or not np.all(np.isfinite(scores)):
        raise ValueError("Classification test predictions contain non-finite values.")
    if not np.isin(actual, (0, 1)).all():
        raise ValueError("Classification test labels must be binary.")
    if not np.array_equal(predicted.astype("int8"), (scores >= threshold).astype("int8")):
        raise ValueError("Saved test classifications do not use the locked threshold.")

    feature_columns = metadata.get("feature_columns", [])
    pipeline = model_bundle.get("pipeline")
    if pipeline is None or not set(feature_columns).issubset(predictions.columns):
        raise ValueError("Classifier model or its saved prediction features are unavailable.")
    probabilities = np.asarray(
        pipeline.predict_proba(predictions.loc[:, feature_columns])[:, 1],
        dtype=float,
    )
    if not np.allclose(probabilities, scores, rtol=1e-7, atol=1e-8):
        raise ValueError("Classifier risk scores do not match the saved fitted model.")

    tn, fp, fn, tp = confusion_matrix(actual, predicted, labels=[0, 1]).ravel()
    metrics = {
        "threshold": threshold,
        "order_count": int(len(actual)),
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "flagged_orders": int(predicted.sum()),
        "flagged_share": float(predicted.mean()),
        "precision": float(precision_score(actual, predicted, zero_division=0)),
        "recall": float(recall_score(actual, predicted, zero_division=0)),
        "f1": float(f1_score(actual, predicted, zero_division=0)),
        "f2": float(fbeta_score(actual, predicted, beta=2, zero_division=0)),
        "roc_auc": float(roc_auc_score(actual, scores)),
        "pr_auc": float(average_precision_score(actual, scores)),
        "brier_score": float(brier_score_loss(actual, scores)),
        "accuracy": float(accuracy_score(actual, predicted)),
    }
    saved_metrics = selected["test_metrics"]
    _assert_close("classification test threshold", threshold, float(saved_metrics["threshold"]))
    for key in (
        "true_negatives",
        "false_positives",
        "false_negatives",
        "true_positives",
        "flagged_orders",
    ):
        if metrics[key] != int(saved_metrics[key]):
            raise ValueError(f"Saved classification test {key} does not match predictions.")
    for key in (
        "precision",
        "recall",
        "f1",
        "f_beta",
        "roc_auc",
        "pr_auc",
        "brier_score",
        "accuracy",
    ):
        report_key = "f2" if key == "f_beta" else key
        _assert_close(
            f"classification test {key}",
            metrics[report_key],
            float(saved_metrics[key]),
        )
    expected_confusion = [
        [metrics["true_negatives"], metrics["false_positives"]],
        [metrics["false_negatives"], metrics["true_positives"]],
    ]
    if saved_metrics["confusion_matrix"]["values"] != expected_confusion:
        raise ValueError("Saved classification confusion matrix does not match predictions.")

    test_late_share = float(np.mean(actual))
    comparison = report["model_comparison"]
    if not any(row["model"] == selected_name for row in comparison):
        raise ValueError("Selected classifier is absent from the model comparison.")
    validation_selected = max(
        comparison,
        key=lambda row: (
            row["validation_metrics"]["f_beta"],
            row["validation_metrics"]["recall"],
            row["validation_metrics"]["precision"],
            row["validation_metrics"]["pr_auc"],
        ),
    )["model"]
    if validation_selected != selected_name:
        raise ValueError("Selected classifier does not match the validation model-selection rule.")
    if float(selected["validation_metrics"]["threshold"]) != threshold:
        raise ValueError("Selected classifier threshold differs from its validation result.")
    class_distribution = report["class_distribution"]
    if int(class_distribution["on_time"]) + int(class_distribution["late"]) != int(
        class_distribution["total"]
    ):
        raise ValueError("Classification class distribution counts do not sum to total.")
    _verify_importance_report(
        report["feature_importance"],
        importance,
        ("pr_auc_decrease", "standard_deviation"),
        "Classification",
    )
    return {
        "selected_model": selected_name,
        "selected_threshold": threshold,
        "threshold_selection": metadata["training_configuration"][
            "threshold_selection"
        ],
        "test_set_used_for_tuning_or_threshold_selection": metadata[
            "training_configuration"
        ]["test_set_used_for_tuning_or_threshold_selection"],
        "model_comparison": comparison,
        "selected_validation_metrics": selected["validation_metrics"],
        "validation_threshold_comparison": selected[
            "validation_threshold_comparison"
        ],
        "selected_test_metrics": metrics,
        "test_late_share": test_late_share,
        "feature_importance": _records(importance),
        "confusion_matrix_definition": (
            "Rows are actual [on-time, late], columns are predicted [on-time, late]. "
            "A false positive is an on-time delivery flagged late; a false negative "
            "is a late delivery not flagged."
        ),
    }


def _priority_evidence(
    delay: dict[str, Any],
    regression: dict[str, Any],
    classification: dict[str, Any],
    riders: dict[str, Any],
    zones: dict[str, Any],
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    traffic_rows = [
        row for row in delay["traffic"]
        if row.get("analysis_traffic") != "(Missing)" and row.get("late_rate") is not None
    ]
    traffic_high = max(traffic_rows, key=lambda row: row["late_rate"])
    traffic_low = min(traffic_rows, key=lambda row: row["late_rate"])
    long_distance = next(
        row
        for row in delay["distance_bands"]
        if row["analysis_distance_band"] == "12+ km"
    )
    regression_importance = {
        row["feature"]: row for row in regression["feature_importance"]
    }
    rider_segments = riders["segments"]
    high_late_segment = max(rider_segments, key=lambda row: row["late_rate"])
    low_late_segment = min(rider_segments, key=lambda row: row["late_rate"])
    zone_rankings = zones["candidate_zone_ranking"]
    if not zone_rankings:
        raise ValueError("PS6 candidate-zone ranking is empty.")
    leading_zone = zone_rankings[0]
    zone_evaluations = zones["cluster_count_evaluation"]
    selected_k = int(zones["metadata"]["selected_cluster_count"])
    max_evaluated_k = max(int(row["cluster_count"]) for row in zone_evaluations)
    selection_warning = (
        f"PS6 selected K={selected_k}, which is the upper boundary of the evaluated "
        f"range (maximum K={max_evaluated_k}); the silhouette result does not establish "
        "a globally optimal or stable site partition."
        if selected_k == max_evaluated_k
        else (
            f"PS6 selected K={selected_k}; K-selection is exploratory and does not "
            "establish a globally optimal or stable site partition."
        )
    )
    evidence = {
        "ps3": {
            "highest_late_rate_traffic_category": traffic_high,
            "lowest_late_rate_traffic_category": traffic_low,
            "long_distance_band": long_distance,
        },
        "ps4_permutation_importance": regression_importance,
        "ps6_leading_candidate_zone": leading_zone,
        "ps6_selected_cluster_count": selected_k,
        "ps6_max_evaluated_cluster_count": max_evaluated_k,
        "ps6_selection_warning": selection_warning,
        "ps7_highest_late_rate_segment": high_late_segment,
        "ps7_lowest_late_rate_segment": low_late_segment,
    }
    classifier_test = classification["selected_test_metrics"]
    recommendations = [
        {
            "id": "REC-1",
            "title": "Pilot human-reviewed late-order risk triage",
            "priority": "High",
            "evidence": (
                f"On {classifier_test['order_count']:,} held-out orders, "
                f"the selected classifier at its training-validation threshold of "
                f"{classification['selected_threshold']:.2f} achieved "
                f"{100 * classifier_test['recall']:.1f}% recall and "
                f"{100 * classifier_test['precision']:.1f}% precision. It flagged "
                f"{classifier_test['flagged_orders']:,} orders, including "
                f"{classifier_test['false_positives']:,} "
                "false positives. The observed late-order share was "
                f"{100 * delay['delivery_time']['late_rate']:.1f}%."
            ),
            "problem": (
                "The retrospective classifier catches many late orders, but its low "
                "precision means a large share of alerts would be unnecessary."
            ),
            "action": (
                "Evaluate the locked order-placement classifier in shadow mode, then "
                "consider a controlled human-review pilot only after operations defines "
                "alert capacity and the official SLA. Keep decisions with staff; do not "
                "automatically change delivery promises, dispatch, or rider treatment."
            ),
            "expected_benefit": (
                "Could help direct attention toward orders with higher estimated late risk. "
                "The dataset does not measure intervention effectiveness, staffing capacity, "
                "or savings, so no numerical benefit is claimed."
            ),
            "implementation": [
                "Confirm the operational SLA and that each predictor is available at the proposed scoring time.",
                f"Use the saved model and locked {classification['selected_threshold']:.2f} "
                "threshold for the initial offline/shadow evaluation; do not tune against "
                "the held-out test labels.",
                "Have operations define a review workflow and alert capacity before any live pilot.",
                "Log scores, review actions, subsequent outcomes, and model/data version for audit.",
            ],
            "risks": [
                f"At the selected test operating point, {classifier_test['false_positives']:,} "
                f"of {classifier_test['flagged_orders']:,} flags were false positives; "
                f"observed test precision was {100 * classifier_test['precision']:.1f}%.",
                "The evaluation is retrospective on a food-delivery proxy dataset; calibration and feature availability may change in a live service.",
                "The threshold optimizes validation F2 rather than verified operational costs or staffing capacity.",
            ],
            "monitoring_kpi": (
                "Flagged-order share, precision, recall, false-positive and false-negative counts, "
                "review response time, and late rate among reviewed versus comparable unreviewed orders."
            ),
            "validation": (
                "Start with shadow scoring. If operationally feasible, run a pre-registered "
                "controlled pilot with a suitable comparison group and confidence intervals; "
                "evaluate outcome changes and alert workload without using test labels to retune."
            ),
        },
        {
            "id": "REC-2",
            "title": "Investigate peak-period, distance, and workload processes",
            "priority": "High",
            "evidence": (
                f"PS3 observed a {100 * traffic_high['late_rate']:.1f}% late rate for "
                f"{traffic_high['analysis_traffic']} traffic "
                f"(n={traffic_high['order_count']:,}) versus "
                f"{100 * traffic_low['late_rate']:.1f}% for "
                f"{traffic_low['analysis_traffic']} traffic "
                f"(n={traffic_low['order_count']:,}); orders in the 12+ km band had a "
                f"{100 * long_distance['late_rate']:.1f}% late rate "
                f"(n={long_distance['order_count']:,}). PS4 held-out permutation importance "
                f"was {regression_importance['is_peak_hour']['mae_increase_minutes']:.3f} "
                "minutes MAE increase for peak-hour and "
                f"{regression_importance['delivery_distance_km']['mae_increase_minutes']:.3f} "
                "for distance. PS7's segment late rates ranged from "
                f"{100 * low_late_segment['late_rate']:.1f}% to "
                f"{100 * high_late_segment['late_rate']:.1f}%; these groups are descriptive "
                "and formed partly from late-rate and volume features."
            ),
            "problem": (
                "Observed delivery outcomes vary across traffic, distance, and time groups, "
                "but the dataset cannot establish which process or resource constraints cause the differences."
            ),
            "action": (
                "Conduct a non-punitive operational review of peak-period and longer-distance "
                "orders, recording dispatch/load conditions and available rider workload. "
                "Use PS7 segment profiles only to frame aggregate support questions, never to "
                "rank or discipline individuals."
            ),
            "expected_benefit": (
                "May identify process constraints worth testing. No causal effect, improvement "
                "size, or rider-level explanation is established by these retrospective associations."
            ),
            "implementation": [
                "Confirm traffic/weather data definitions and timestamps; they were retrospective context, not model inputs.",
                "Validate the configured peak-hour windows and compare like-for-like locations, dates, and distance bands.",
                "Review dispatch timing, order load, and operating conditions at aggregate cohort level before selecting a specific intervention.",
                "Keep sufficient-observation flags visible and do not infer rider tenure from age.",
            ],
            "risks": [
                "Traffic and weather categories may be missing, subjective, confounded, or recorded after the decision point.",
                "PS7 uses a 30-delivery screening floor, not a guarantee of stable rider estimates; its silhouette score is modest.",
                "The PS3 subgroup differences are not causal estimates and may not transfer to another service or time period.",
            ],
            "monitoring_kpi": (
                "Delivery-time median and p90, late-order rate, distance-adjusted outcomes, "
                "and workload/assignment measures by pre-defined time and operating-condition cohort."
            ),
            "validation": (
                "Predefine comparable pilot and control cohorts, measure a baseline period, "
                "then use a randomized or defensible phased rollout where feasible. Report "
                "sample sizes and uncertainty and check that any aggregate differences persist."
            ),
        },
        {
            "id": "REC-3",
            "title": "Validate geographic evidence before site commitments",
            "priority": "Medium",
            "evidence": (
                f"The highest PS6 relative candidate score is zone "
                f"{leading_zone['cluster_id']} (score "
                f"{leading_zone['candidate_priority_score']:.1f}), with "
                f"{leading_zone['order_count']:,} orders "
                f"({100 * leading_zone['demand_share']:.1f}% of clustered demand), "
                f"{100 * leading_zone['late_rate']:.1f}% late, and a nearest supplied "
                f"restaurant distance of {leading_zone['nearest_restaurant_km']:.1f} km. "
                f"{selection_warning}"
            ),
            "problem": (
                "The exploratory zones are derived from food-delivery customer coordinates; "
                "restaurants are only a service-coverage proxy, and the score is not a site-feasibility measure."
            ),
            "action": (
                "Treat the top-ranked zone as a validation candidate for further study, not "
                "a dark-store recommendation. Reconcile source coordinates and time coverage, "
                "then assess actual candidate properties and operating feasibility before "
                "any investment decision."
            ),
            "expected_benefit": (
                "Could reduce the chance of acting on a geographic artifact and focus field "
                "research on measured demand. No site benefit, savings, or commercial return "
                "can be estimated from the available data."
            ),
            "implementation": [
                "Verify customer and existing-location coordinates against trusted operational records and current service boundaries.",
                "Recompute demand using recent time windows and sensitivity checks across plausible cluster counts.",
                "Collect candidate property availability, rent, inventory, road access, staffing, capacity, and operational-cost data.",
                "Keep the weighted zone score labeled as an exploratory ranking, separate from a feasibility or investment score.",
            ],
            "risks": [
                "The selected PS6 K is at the upper end of the evaluated range, so finer partitions may score better.",
                "The source is a food-delivery dataset, not verified dark-store operations; restaurant locations do not prove dark-store coverage.",
                "No property, rent, road-access, inventory, or operating-cost data is available.",
            ],
            "monitoring_kpi": (
                "Coordinate validation rate, demand-share stability by time window and K, "
                "candidate-zone order coverage, and independently verified service-area coverage."
            ),
            "validation": (
                "Reconcile a sample of locations with ground-truth records, test spatial and "
                "temporal stability on a separate period, and complete field/property feasibility "
                "checks before proposing a site."
            ),
        },
    ]
    return evidence, recommendations


def build_management_report(project_root: Path | None = None) -> dict[str, Any]:
    """Verify saved PS1/PS3–PS7 evidence and synthesize the PS8 management report."""
    root = project_root or PROJECT_ROOT

    def resolve(configured_path: Path) -> Path:
        return root / configured_path.relative_to(PROJECT_ROOT)

    required_paths = (
        resolve(REGRESSION_REPORT_PATH),
        resolve(REGRESSION_PREDICTIONS_PATH),
        resolve(REGRESSION_IMPORTANCE_PATH),
        resolve(REGRESSION_MODEL_PATH),
        resolve(CLASSIFICATION_REPORT_PATH),
        resolve(CLASSIFICATION_TEST_PREDICTIONS_PATH),
        resolve(CLASSIFICATION_IMPORTANCE_PATH),
        resolve(CLASSIFICATION_MODEL_PATH),
        resolve(DARK_STORE_REPORT_PATH),
        resolve(RIDER_SEGMENT_REPORT_PATH),
        resolve(PROCESSED_DATA_PATH),
        root / "reports" / "data_quality_report.json",
        root / "reports" / "delay_analysis.json",
    )
    missing = [file_path for file_path in required_paths if not file_path.is_file()]
    if missing:
        raise FileNotFoundError(
            "PS8 cannot build a complete report; required artifacts are missing: "
            + ", ".join(str(item) for item in missing)
        )

    quality = _read_json(root / "reports" / "data_quality_report.json")
    delay = _read_json(root / "reports" / "delay_analysis.json")
    regression_source = _read_json(resolve(REGRESSION_REPORT_PATH))
    classification_source = _read_json(resolve(CLASSIFICATION_REPORT_PATH))
    zones = _read_json(resolve(DARK_STORE_REPORT_PATH))
    riders = _read_json(resolve(RIDER_SEGMENT_REPORT_PATH))
    current_hash = sha256_file(resolve(PROCESSED_DATA_PATH))

    for name, report in (
        ("PS1 data-quality", quality),
        ("PS3 delay", delay),
        ("PS4 regression", regression_source),
        ("PS5 classification", classification_source),
        ("PS6 demand zones", zones),
        ("PS7 rider segmentation", riders),
    ):
        metadata = report.get("metadata", {})
        configured_sla = metadata.get(
            "sla_threshold_minutes",
            report.get("sla_threshold_minutes"),
        )
        if configured_sla is not None and int(configured_sla) != SLA_THRESHOLD_MINUTES:
            raise ValueError(f"{name} report uses an inconsistent SLA threshold.")
        fingerprint = metadata.get("processed_dataset_sha256")
        if fingerprint is None:
            fingerprint = (
                metadata.get("dataset", {}).get("processed_sha256")
                if isinstance(metadata.get("dataset"), dict)
                else None
            )
        if fingerprint and fingerprint != current_hash:
            raise ValueError(f"{name} report was produced from a different processed dataset.")
    if int(delay["sla_threshold_minutes"]) != SLA_THRESHOLD_MINUTES:
        raise ValueError("PS3 delay report uses an inconsistent SLA threshold.")

    regression_model = joblib.load(resolve(REGRESSION_MODEL_PATH))
    classification_model = joblib.load(resolve(CLASSIFICATION_MODEL_PATH))
    regression = _verify_regression_evidence(
        regression_source,
        pd.read_csv(resolve(REGRESSION_PREDICTIONS_PATH)),
        pd.read_csv(resolve(REGRESSION_IMPORTANCE_PATH)),
        regression_model,
        resolve(PROCESSED_DATA_PATH),
    )
    classification = _verify_classification_evidence(
        classification_source,
        pd.read_csv(resolve(CLASSIFICATION_TEST_PREDICTIONS_PATH)),
        pd.read_csv(resolve(CLASSIFICATION_IMPORTANCE_PATH)),
        classification_model,
        resolve(PROCESSED_DATA_PATH),
    )

    evidence, recommendations = _priority_evidence(
        delay,
        regression,
        classification,
        riders,
        zones,
    )

    overview = quality["dataset_overview"]
    records = quality["records"]
    duplicate = quality["duplicate_analysis"]
    coordinate_audit = quality["coordinate_validation"]
    geographic_metadata = zones["metadata"]
    cluster_evaluations = zones["cluster_count_evaluation"]
    rider_audit = riders["data_audit"]
    report: dict[str, Any] = {
        "metadata": {
            "report_version": "1.0.0",
            "source_dataset_sha256": current_hash,
            "source_dataset_path": str(resolve(PROCESSED_DATA_PATH)),
            "official_challenge_pdf_found_in_project": False,
            "official_challenge_guide_found_in_workspace": True,
            "official_challenge_guide_filename": "QUICK_COMMERCEDELIVERY_INTELLIGENCE.txt",
            "sla_threshold_minutes": SLA_THRESHOLD_MINUTES,
            "sla_status": "provisional",
            "created_from_saved_outputs": True,
            "models_retrained": False,
            "recommendation_count": len(recommendations),
        },
        "problem_statement_status": [
            {"id": "PS1", "status": "Implemented", "evidence": "Data audit, quality reports, and validated preprocessing are present."},
            {"id": "PS2", "status": "Documented against official guide", "evidence": "The standalone contract and saved-model prediction demo cover the target, provisional SLA/late definition, Haversine distance, and prediction-time leakage exclusions."},
            {"id": "PS3", "status": "Implemented with data limitation", "evidence": "Saved delay analysis and dashboard; the dataset has no verified rider experience/tenure field, so that comparison is unavailable."},
            {"id": "PS4", "status": "Implemented", "evidence": "Saved regression model comparisons, test predictions, importance, and dashboard."},
            {"id": "PS5", "status": "Implemented with cost limitation", "evidence": "Saved classifier, validation-only threshold selection, held-out predictions, and dashboard; no measured intervention-cost ratio is available to claim a cost-optimal threshold."},
            {"id": "PS6", "status": "Implemented with exploratory limitations", "evidence": "Saved customer-coordinate clustering, diagnostics, candidate ranking, and dashboard."},
            {"id": "PS7", "status": "Implemented with descriptive limitations", "evidence": "Saved rider aggregation, K-Means segmentation, measured profiles, assignments, and dashboard; segment-specific support/roster/training actions are not supplied."},
            {"id": "PS8", "status": "Implemented", "evidence": "This report, explainability evidence, exactly three management actions, and the final dashboard."},
        ],
        "dataset": {
            "description": "Supplied food-delivery data used as a quick-commerce proxy; not verified dark-store operations data.",
            "rows": int(overview["rows"]),
            "columns": int(overview["columns"]),
            "retained_rows": int(records["retained_records"]),
            "excluded_rows": int(records["excluded_records"]),
            "duplicate_order_id_rows_after_first": int(
                duplicate["duplicate_order_id_rows_after_first"]
            ),
            "valid_distance_observations": int(
                coordinate_audit["distance_km"]["model_eligible_pair_count"]
            ),
            "invalid_or_unavailable_distance_observations": int(
                coordinate_audit["distance_km"]["not_model_eligible_pair_count"]
            ),
            "late_orders": int(delay["delivery_time"]["late_orders"]),
            "late_rate": float(delay["delivery_time"]["late_rate"]),
            "mean_delivery_minutes": float(delay["delivery_time"]["mean_minutes"]),
            "median_delivery_minutes": float(delay["delivery_time"]["median_minutes"]),
            "sla_status": delay["sla_status"],
            "source_sha256": overview["source_sha256"],
        },
        "regression": regression,
        "classification": classification,
        "explainability": {
            "regression_importance": regression["feature_importance"],
            "classification_importance": classification["feature_importance"],
            "methodology": (
                "Uses the saved held-out permutation-importance artifacts: regression reports "
                "the increase in MAE after a feature is permuted; classification reports "
                "the decrease in average precision. These are predictive reliance measures, "
                "not causal effects. Correlated/derived features can share or obscure importance."
            ),
            "feature_availability_warning": (
                "Pickup time is observed after order placement and is excluded from both models. "
                "Weather, traffic, and festival fields are retrospective context because their "
                "availability at order placement is unverified. Distance also requires validated "
                "coordinates to be available at scoring time."
            ),
        },
        "ps3_findings": {
            "mean_delivery_minutes": float(delay["delivery_time"]["mean_minutes"]),
            "median_delivery_minutes": float(delay["delivery_time"]["median_minutes"]),
            "late_rate": float(delay["delivery_time"]["late_rate"]),
            "traffic": delay["traffic"],
            "distance_bands": delay["distance_bands"],
            "weather": delay["weather"],
            "interpretation": (
                "Group differences are observational associations; they do not prove that "
                "traffic, weather, distance, or any rider characteristic caused a delay."
            ),
        },
        "geographic_analysis": {
            "selected_cluster_count": int(geographic_metadata["selected_cluster_count"]),
            "selected_silhouette_score": float(
                geographic_metadata["selected_silhouette_score"]
            ),
            "inertia_elbow_suggestion": int(
                geographic_metadata["inertia_elbow_suggestion"]
            ),
            "cluster_count_evaluation": cluster_evaluations,
            "customer_coordinate_observations": int(
                zones["data_audit"]["customer_cluster_observations"]
            ),
            "customer_coordinate_records_excluded": int(
                zones["data_audit"]["orders_excluded_from_customer_clustering"]
            ),
            "candidate_zone_ranking": zones["candidate_zone_ranking"][:5],
            "selection_warning": evidence["ps6_selection_warning"],
            "limitations": zones.get("limitations", []),
        },
        "rider_analysis": {
            "total_riders": int(rider_audit["distinct_riders_with_completed_deliveries"]),
            "segmented_riders": int(rider_audit["riders_meeting_minimum_deliveries"]),
            "insufficient_sample_riders": int(
                rider_audit["riders_flagged_insufficient_deliveries"]
            ),
            "minimum_deliveries": int(rider_audit["minimum_deliveries_for_segmentation"]),
            "selected_cluster_count": int(riders["metadata"]["selected_cluster_count"]),
            "selected_silhouette_score": float(
                riders["metadata"]["selected_silhouette_score"]
            ),
            "segments": riders["segments"],
            "limitations": riders.get("limitations", []),
        },
        "cross_statement_evidence": {
            "highest_late_rate_traffic_category": evidence[
                "ps3"
            ]["highest_late_rate_traffic_category"],
            "lowest_late_rate_traffic_category": evidence[
                "ps3"
            ]["lowest_late_rate_traffic_category"],
            "long_distance_band": evidence["ps3"]["long_distance_band"],
            "highest_late_rate_rider_segment": evidence["ps7_highest_late_rate_segment"],
            "lowest_late_rate_rider_segment": evidence["ps7_lowest_late_rate_segment"],
        },
        "recommendations": recommendations,
        "limitations": [
            "The official guide was reviewed from QUICK_COMMERCEDELIVERY_INTELLIGENCE.txt in the workspace root.",
            "The 30-minute SLA remains provisional because the guide does not publish an approved threshold.",
            "The 30-minute SLA is provisional and must be confirmed before operational use.",
            "The food-delivery dataset is only a quick-commerce proxy and may not represent the intended operation or current conditions.",
            "All model metrics are retrospective; performance, calibration, and feature availability may change in deployment.",
            "The classification threshold is based on validation F2, not verified operating capacity or measured false-positive/false-negative costs.",
            "PS6 zones and restaurant locations do not establish dark-store suitability; the selected K is at the evaluated boundary.",
            "PS7 segments are descriptive groups and must not be used as individual rankings or employment judgments; the proposed support actions are hypotheses requiring validation.",
            "No intervention has been tested; the report makes no quantified savings, ROI, delivery-improvement, or causal claims.",
        ],
    }
    if len(recommendations) != 3:
        raise AssertionError("The final management report must contain exactly three actions.")
    return report


def render_management_markdown(report: dict[str, Any]) -> str:
    """Render the verified PS8 report as readable judge/management documentation."""
    regression = report["regression"]
    classification = report["classification"]
    test = classification["selected_test_metrics"]
    xgboost_result = next(
        row for row in regression["model_comparison"] if row["model"] == "XGBoost"
    )
    xgboost_gap = abs(
        xgboost_result["test_metrics"]["mae_minutes"]
        - regression["selected_test_metrics"]["mae_minutes"]
    )
    default_threshold = next(
        row
        for row in classification["validation_threshold_comparison"]
        if math.isclose(row["threshold"], 0.5)
    )
    lines = [
        "# Quick-Commerce Delivery Intelligence — Final Management Report",
        "",
        "## Project status",
        "",
        "| Problem statement | Status | Evidence / qualification |",
        "|---|---|---|",
    ]
    lines.extend(
        f"| {row['id']} | {row['status']} | {row['evidence']} |"
        for row in report["problem_statement_status"]
    )
    lines.extend(
        [
            "",
            "The official challenge guide was reviewed from the workspace-root TXT file. PS2's "
            "target, SLA/late definition, Haversine distance, and leakage exclusions are documented; "
            "the 30-minute SLA remains provisional.",
            "",
            "## Dataset and SLA",
            "",
            f"- {report['dataset']['rows']:,} food-delivery records; "
            f"{report['dataset']['retained_rows']:,} retained by preprocessing.",
            f"- {report['dataset']['duplicate_order_id_rows_after_first']:,} duplicate order IDs after the first row.",
            f"- {report['dataset']['valid_distance_observations']:,} valid distance observations; "
            f"{report['dataset']['invalid_or_unavailable_distance_observations']:,} invalid or unavailable.",
            f"- Observed late-order share: {report['dataset']['late_rate']:.1%} "
            f"({report['dataset']['late_orders']:,} orders).",
            f"- SLA: {report['dataset']['sla_status']}. A delivery is late only when time "
            f"is strictly greater than {report['metadata']['sla_threshold_minutes']} minutes.",
            "- The source is a food-delivery dataset used as a proxy, not verified quick-commerce or dark-store operations data.",
            "",
            "## Regression model selection",
            "",
            f"Selected **{regression['selected_model']}** using the saved training-only "
            "cross-validation MAE criterion. No model was retrained for this report.",
            "",
            "| Model | CV MAE (min) | Test MAE (min) | Test RMSE (min) | Test R² |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for row in regression["model_comparison"]:
        metric = row["test_metrics"]
        lines.append(
            f"| {row['model']} | {row['cv_mae_minutes']:.3f} | "
            f"{metric['mae_minutes']:.3f} | {metric['rmse_minutes']:.3f} | "
            f"{metric['r2']:.3f} |"
        )
    lines.extend(
        [
            "",
            f"The mean baseline test MAE was "
            f"{regression['baseline_test_metrics']['mae_minutes']:.3f} minutes. "
            f"Selected-model MAE improved by "
            f"{regression['mae_improvement_vs_baseline_minutes']:.3f} minutes "
            f"({regression['mae_improvement_vs_baseline_percent']:.1f}%). "
            "Gradient Boosting and XGBoost test MAEs differ by only "
            f"{xgboost_gap:.4f} "
            "minutes; this small difference does not establish a meaningful deployment advantage.",
            "",
            "MAE is average absolute delivery-time error; RMSE is more sensitive to larger "
            "errors; R² is the share of test-target variation accounted for relative to a "
            "mean baseline. The selected tree ensemble is less directly interpretable than "
            "linear regression and still leaves substantial error. These are offline estimates, "
            "not a guarantee for an individual order. Gradient Boosting is a nonlinear ensemble "
            "with more implementation/inference complexity than linear regression; XGBoost's "
            "test MAE is effectively tied, while the saved sklearn Gradient Boosting model "
            "avoids requiring XGBoost for the selected estimator. No serving-latency or "
            "deployment-cost benchmark was measured.",
            "",
            "## Late-order classification",
            "",
            f"Selected **{classification['selected_model']}** at the fixed "
            f"**{classification['selected_threshold']:.2f}** threshold. Threshold selection "
            "used training-only validation and maximized F2; no held-out test label was used "
            "to choose or change it.",
            "",
            "| Model | CV average precision | Validation precision | Validation recall | Validation F2 | Validation ROC-AUC |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for row in classification["model_comparison"]:
        metrics = row["validation_metrics"]
        lines.append(
            f"| {row['model']} | {row['cv_average_precision']:.3f} | "
            f"{metrics['precision']:.3f} | {metrics['recall']:.3f} | "
            f"{metrics['f_beta']:.3f} | {metrics['roc_auc']:.3f} |"
        )
    lines.extend(
        [
            "",
            "Model comparisons above use validation results. Only the selected model's "
            "held-out test evaluation is reported; test labels were not used for model or "
            "threshold selection.",
            "",
            f"On {test['order_count']:,} test orders: precision "
            f"{test['precision']:.1%}, recall {test['recall']:.1%}, F1 {test['f1']:.3f}, "
            f"F2 {test['f2']:.3f}, ROC-AUC {test['roc_auc']:.3f}.",
            f"Confusion matrix (rows actual on-time/late; columns predicted on-time/late): "
            f"[[{test['true_negatives']:,}, {test['false_positives']:,}], "
            f"[{test['false_negatives']:,}, {test['true_positives']:,}]]. "
            f"The model flagged {test['flagged_orders']:,} orders "
            f"({test['flagged_share']:.1%}); false positives were "
            f"{test['false_positives']:,}, and false negatives were {test['false_negatives']:,}.",
            "",
            f"Threshold rationale: {classification['selected_threshold']:.2f} maximized "
            f"F2 on the training-only validation split (F2 "
            f"{classification['selected_validation_metrics']['f_beta']:.3f}, recall "
            f"{classification['selected_validation_metrics']['recall']:.1%}, precision "
            f"{classification['selected_validation_metrics']['precision']:.1%}). F2 weights "
            "recall more heavily than precision because missed late orders matter for early "
            "identification; the trade-off is additional false-positive intervention flags. "
            f"At the unchanged threshold, the held-out test flagged "
            f"{test['flagged_orders']:,}/{test['order_count']:,} orders "
            f"({test['flagged_share']:.1%}); {test['false_positives']:,} were false "
            f"positives and {test['false_negatives']:,} were false negatives. "
            "This is not a cost-optimal threshold: intervention costs and alert capacity "
            "are not measured, and held-out test labels were not used to select it. "
            "High recall means most observed late orders were flagged, not that alerts are "
            "operationally efficient. No alert costs or capacity were invented. "
            f"On validation, the {classification['selected_threshold']:.2f} threshold had precision "
            f"{classification['selected_validation_metrics']['precision']:.1%}, recall "
            f"{classification['selected_validation_metrics']['recall']:.1%}, and F2 "
            f"{classification['selected_validation_metrics']['f_beta']:.3f}; at 0.50, precision "
            f"was {default_threshold['precision']:.1%}, recall {default_threshold['recall']:.1%}, "
            f"and F2 {default_threshold['f_beta']:.3f}. The classifier identifies "
            "retrospective risk and does not prevent delays. Precision is the share of flagged "
            "orders that were actually late; recall is the share of actual late orders flagged. "
            "F1 is their harmonic mean, and F2 weights recall more heavily. ROC-AUC measures "
            "ranking discrimination across thresholds; it does not select an operating point "
            "or establish probability calibration. The confusion matrix gives the true/false "
            "positive and negative counts at the fixed threshold.",
            "",
            "## Explainability",
            "",
            "Permutation importance is calculated on the saved held-out evaluation data. "
            "Regression values are MAE increase in minutes; classification values are average "
            "precision decrease. They measure predictive reliance, not causal influence.",
            "",
            "| Regression feature | MAE increase (min) |",
            "|---|---:|",
        ]
    )
    for row in regression["feature_importance"][:5]:
        lines.append(f"| {row['feature']} | {row['mae_increase_minutes']:.4f} |")
    lines.extend(
        [
            "",
            "| Classification feature | Average-precision decrease |",
            "|---|---:|",
        ]
    )
    for row in classification["feature_importance"][:5]:
        lines.append(f"| {row['feature']} | {row['pr_auc_decrease']:.4f} |")
    lines.extend(
        [
            "",
            "Peak-hour and distance features rank highly in both saved analyses. Pickup time "
            "is unavailable at order placement and excluded. Weather/traffic are retrospective "
            "context only; distance is usable only if validated coordinates are available when scored. "
            "Correlated/derived features can split or obscure permutation importance.",
            "",
            "## PS3–PS7 evidence",
            "",
            f"- PS3: average/median delivery time is "
            f"{report['ps3_findings']['mean_delivery_minutes']:.2f}/"
            f"{report['ps3_findings']['median_delivery_minutes']:.1f} minutes. Traffic, "
            "weather, time, and distance differences are observational associations.",
            f"- PS6: selected K={report['geographic_analysis']['selected_cluster_count']}, "
            f"silhouette {report['geographic_analysis']['selected_silhouette_score']:.3f}; "
            f"the inertia elbow suggests K={report['geographic_analysis']['inertia_elbow_suggestion']}. "
            f"{report['geographic_analysis']['selection_warning']}",
            f"- PS7: {report['rider_analysis']['segmented_riders']:,} of "
            f"{report['rider_analysis']['total_riders']:,} riders met the "
            f"{report['rider_analysis']['minimum_deliveries']}-delivery screen; "
            f"{report['rider_analysis']['insufficient_sample_riders']:,} were flagged "
            "for insufficient observations. Segments are descriptive, not individual rankings.",
            "- Restaurant coordinates are only a coverage proxy; actual property, rent, inventory, road-access, staffing, and operational-cost data are absent.",
            "",
            "## Exactly three prioritised management actions",
            "",
        ]
    )
    for index, recommendation in enumerate(report["recommendations"], start=1):
        lines.extend(
            [
                f"### {index}. {recommendation['title']} — {recommendation['priority']} priority",
                "",
                f"**Evidence:** {recommendation['evidence']}",
                "",
                f"**Operational problem:** {recommendation['problem']}",
                "",
                f"**Action:** {recommendation['action']}",
                "",
                f"**Expected benefit:** {recommendation['expected_benefit']}",
                "",
                "**Implementation requirements:**",
                *[f"- {item}" for item in recommendation["implementation"]],
                "",
                "**Risks and limitations:**",
                *[f"- {item}" for item in recommendation["risks"]],
                "",
                f"**Monitoring KPI:** {recommendation['monitoring_kpi']}",
                "",
                f"**Validation:** {recommendation['validation']}",
                "",
            ]
        )
    lines.extend(["## Remaining limitations", ""])
    lines.extend(f"- {item}" for item in report["limitations"])
    lines.extend(
        [
            "",
            "## Reproducibility",
            "",
            "The report is generated from saved PS1, PS3, PS4, PS5, PS6, and PS7 reports, "
            "saved predictions, importance tables, and fitted regression/classification models. "
            "The generation step checks dataset provenance, SLA consistency, model/report "
            "identity, model predictions, and recomputed test metrics. It does not retrain models.",
            "",
        ]
    )
    return "\n".join(lines)


def write_management_report(
    report: dict[str, Any],
    *,
    json_path: Path = MANAGEMENT_INSIGHTS_REPORT_PATH,
    markdown_path: Path = MANAGEMENT_REPORT_MARKDOWN_PATH,
) -> tuple[Path, Path]:
    """Persist the verified structured report and judge-readable Markdown."""
    json_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    markdown_path.write_text(render_management_markdown(report), encoding="utf-8")
    return json_path, markdown_path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=None, help="Project root directory.")
    args = parser.parse_args()
    root = args.root or PROJECT_ROOT
    report = build_management_report(project_root=root)
    json_path, markdown_path = write_management_report(
        report,
        json_path=root / "reports" / MANAGEMENT_INSIGHTS_REPORT_PATH.name,
        markdown_path=root / "reports" / MANAGEMENT_REPORT_MARKDOWN_PATH.name,
    )
    print(f"Verified management report: {json_path}")
    print(f"Readable management report: {markdown_path}")
    print(f"Recommendations: {len(report['recommendations'])}")


if __name__ == "__main__":
    main()
