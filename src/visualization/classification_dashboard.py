"""Streamlit dashboard for held-out late-order risk evaluation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    CLASSIFICATION_IMPORTANCE_PATH,
    CLASSIFICATION_MODEL_PATH,
    CLASSIFICATION_REPORT_PATH,
    CLASSIFICATION_TEST_PREDICTIONS_PATH,
    CLASSIFICATION_VALIDATION_PREDICTIONS_PATH,
)
from src.models.classification import evaluate_classification_metrics

CHART_CONFIG = {"displaylogo": False, "responsive": True}
BLUE = "#245B8F"
TEAL = "#168A8A"


@st.cache_data(show_spinner=False)
def _load_artifacts(
    report_path: str,
    report_mtime_ns: int,
    test_path: str,
    test_mtime_ns: int,
    validation_path: str,
    validation_mtime_ns: int,
    importance_path: str,
    importance_mtime_ns: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    del report_mtime_ns, test_mtime_ns, validation_mtime_ns, importance_mtime_ns
    with Path(report_path).open(encoding="utf-8") as file_handle:
        report = json.load(file_handle)
    return (
        report,
        pd.read_csv(test_path),
        pd.read_csv(validation_path),
        pd.read_csv(importance_path),
    )


def render_classification_dashboard() -> None:
    """Render validation-based threshold exploration and fixed test results."""
    st.header("Late-Order Risk Intelligence")
    st.caption(
        "Risk scores flag deliveries expected to exceed the provisional 30-minute SLA. "
        "The model predicts from order-placement features only."
    )
    paths = (
        CLASSIFICATION_REPORT_PATH,
        CLASSIFICATION_TEST_PREDICTIONS_PATH,
        CLASSIFICATION_VALIDATION_PREDICTIONS_PATH,
        CLASSIFICATION_IMPORTANCE_PATH,
        CLASSIFICATION_MODEL_PATH,
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        st.info(
            "Classification artifacts are not available. From the project directory, "
            "run `python -m src.models.classification`."
        )
        return

    report, test, validation, importance = _load_artifacts(
        str(CLASSIFICATION_REPORT_PATH),
        CLASSIFICATION_REPORT_PATH.stat().st_mtime_ns,
        str(CLASSIFICATION_TEST_PREDICTIONS_PATH),
        CLASSIFICATION_TEST_PREDICTIONS_PATH.stat().st_mtime_ns,
        str(CLASSIFICATION_VALIDATION_PREDICTIONS_PATH),
        CLASSIFICATION_VALIDATION_PREDICTIONS_PATH.stat().st_mtime_ns,
        str(CLASSIFICATION_IMPORTANCE_PATH),
        CLASSIFICATION_IMPORTANCE_PATH.stat().st_mtime_ns,
    )
    selected = report["selected_model"]
    locked_threshold = float(selected["threshold"])
    test_metrics = selected["test_metrics"]
    confusion = test_metrics["confusion_matrix"]["values"]

    st.subheader("Held-out test results")
    cards = st.columns(6)
    cards[0].metric("Orders analysed", f"{test_metrics['order_count']:,}")
    cards[1].metric(
        "Observed late rate",
        f"{report['class_distribution']['late_rate']:.1%}",
    )
    cards[2].metric("Late orders flagged", f"{test_metrics['true_positives']:,}")
    cards[3].metric("Recall", f"{test_metrics['recall']:.1%}")
    cards[4].metric("Precision", f"{test_metrics['precision']:.1%}")
    cards[5].metric("F1", f"{test_metrics['f1']:.3f}")
    st.caption(
        f"Selected: {selected['name']} at validation-selected threshold "
        f"{locked_threshold:.2f}. Test set: {test_metrics['order_count']:,} orders. "
        "False negatives are missed late orders; false positives are unnecessary risk flags."
    )
    validation_metrics = selected["validation_metrics"]
    st.info(
        f"**Why 0.15:** it maximized F2 on the training-only validation split "
        f"(F2 {validation_metrics['f_beta']:.3f}; recall "
        f"{validation_metrics['recall']:.1%}; precision "
        f"{validation_metrics['precision']:.1%}). F2 weights recall more heavily "
        "than precision because missed late orders matter for early identification; "
        "lowering the threshold also creates more false-positive intervention flags. "
        f"At the unchanged threshold, the held-out test flagged "
        f"{test_metrics['flagged_orders']:,}/{test_metrics['order_count']:,} orders "
        f"({test_metrics['flagged_orders'] / test_metrics['order_count']:.1%}); "
        f"{test_metrics['false_positives']:,} were false positives and "
        f"{test_metrics['false_negatives']:,} were false negatives. This is not a "
        "cost-optimal threshold: intervention costs and alert capacity are not measured, "
        "and held-out test labels were not used to select it."
    )

    st.subheader("Model comparison")
    st.dataframe(_comparison_table(report["model_comparison"]), hide_index=True, width="stretch")
    st.caption(
        "Comparison metrics and operating thresholds use the stratified validation split. "
        "Final test metrics below use the locked selected threshold."
    )

    st.subheader("Test-set classification")
    left, right = st.columns(2)
    with left:
        _render_confusion_matrix(confusion)
    with right:
        _render_test_curves(selected["test_curves"], test_metrics)

    st.subheader("Validation threshold exploration")
    st.caption(
        "The control changes validation-set counts only. The final test result and selected "
        "threshold remain fixed; do not use the test set to choose a threshold."
    )
    selected_at_validation = evaluate_classification_metrics(
        validation["actual_is_late"],
        validation["risk_score"],
        locked_threshold,
    )
    threshold = st.slider(
        "Validation risk threshold",
        min_value=0.0,
        max_value=1.0,
        value=locked_threshold,
        step=0.01,
        key="late_classification_validation_threshold",
    )
    live = evaluate_classification_metrics(
        validation["actual_is_late"], validation["risk_score"], threshold
    )
    metrics = st.columns(7)
    metrics[0].metric("Flagged", f"{live['flagged_orders']:,}")
    metrics[1].metric("True positives", f"{live['true_positives']:,}")
    metrics[2].metric("False positives", f"{live['false_positives']:,}")
    metrics[3].metric("False negatives", f"{live['false_negatives']:,}")
    metrics[4].metric("Precision", f"{live['precision']:.1%}")
    metrics[5].metric("Recall", f"{live['recall']:.1%}")
    metrics[6].metric("F1", f"{live['f1']:.3f}")
    st.caption(
        f"Locked validation operating point: {selected_at_validation['flagged_orders']:,} "
        f"flagged; recall {selected_at_validation['recall']:.1%}; "
        f"precision {selected_at_validation['precision']:.1%}."
    )
    _render_threshold_table(selected["validation_threshold_comparison"])

    st.subheader("Probability calibration")
    _render_calibration(
        selected["validation_calibration"],
        selected["test_calibration"],
    )

    st.subheader("Risk patterns and model reliance")
    left, right = st.columns(2)
    with left:
        _render_risk_distribution(test)
    with right:
        _render_feature_importance(importance)
    _render_risk_breakdowns(test)
    st.subheader("Highest-scored held-out orders")
    display_columns = [
        "risk_score",
        "actual_is_late",
        "order_hour",
        "delivery_distance_km",
        "distance_band",
        "City",
        "Type_of_order",
    ]
    st.dataframe(
        test.nlargest(10, "risk_score").loc[:, display_columns].rename(
            columns={
                "risk_score": "Risk score",
                "actual_is_late": "Actually late",
                "order_hour": "Order hour",
                "delivery_distance_km": "Distance (km)",
                "distance_band": "Distance band",
                "City": "City",
                "Type_of_order": "Order type",
            }
        ).round({"Risk score": 3, "Distance (km)": 2}),
        hide_index=True,
        width="stretch",
    )
    with st.expander("Classification report and feature availability"):
        st.dataframe(
            pd.DataFrame(selected["test_classification_report"]).T.round(3),
            width="stretch",
        )
        st.json(report["metadata"]["training_configuration"])
        st.dataframe(
            pd.DataFrame(
                [
                    {"Feature": feature, "Availability": reason}
                    for feature, reason in report["metadata"][
                        "feature_availability_policy"
                    ].items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    st.caption(report["interpretation"])


def _comparison_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "Model": row["model"],
                "CV PR-AUC": row["cv_average_precision"],
                "Threshold": row["validation_threshold"],
                "Precision": row["validation_metrics"]["precision"],
                "Recall": row["validation_metrics"]["recall"],
                "F1": row["validation_metrics"]["f1"],
                "F2": row["validation_metrics"]["f_beta"],
                "ROC-AUC": row["validation_metrics"]["roc_auc"],
                "PR-AUC": row["validation_metrics"]["pr_auc"],
                "Balanced accuracy": row["validation_metrics"]["balanced_accuracy"],
            }
            for row in rows
        ]
    ).sort_values(["F2", "Recall"], ascending=False, ignore_index=True).round(3)


def _render_confusion_matrix(matrix: list[list[int]]) -> None:
    figure = go.Figure(
        go.Heatmap(
            z=matrix,
            x=["Predicted on time", "Predicted late"],
            y=["Actually on time", "Actually late"],
            text=matrix,
            texttemplate="%{text:,}",
            colorscale=[[0, "#EAF0F5"], [1, BLUE]],
            showscale=False,
            hovertemplate="%{y}<br>%{x}<br>Orders: %{z:,}<extra></extra>",
        )
    )
    figure.update_layout(
        title="Held-out confusion matrix",
        xaxis_title="Prediction",
        yaxis_title="Observed outcome",
        margin={"l": 115, "b": 70},
    )
    st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)


def _render_test_curves(curves: dict[str, list[float]], metrics: dict[str, Any]) -> None:
    roc = go.Figure()
    roc.add_scatter(
        x=curves["roc_false_positive_rate"],
        y=curves["roc_true_positive_rate"],
        mode="lines",
        name=f"ROC-AUC {metrics['roc_auc']:.3f}",
        line={"color": BLUE},
    )
    roc.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Chance",
        line={"color": TEAL, "dash": "dash"},
    )
    roc.update_layout(
        title="Held-out ROC curve",
        xaxis_title="False positive rate",
        yaxis_title="Recall",
        margin={"b": 55},
        legend={"orientation": "h", "y": -0.3},
    )
    st.plotly_chart(roc, width="stretch", config=CHART_CONFIG)

    pr = go.Figure()
    pr.add_scatter(
        x=curves["precision_recall_recall"],
        y=curves["precision_recall_precision"],
        mode="lines",
        name=f"PR-AUC {metrics['pr_auc']:.3f}",
        line={"color": TEAL},
    )
    pr.update_layout(
        title="Held-out precision-recall curve",
        xaxis_title="Recall",
        yaxis_title="Precision",
        margin={"b": 55},
        legend={"orientation": "h", "y": -0.3},
    )
    st.plotly_chart(pr, width="stretch", config=CHART_CONFIG)


def _render_threshold_table(rows: list[dict[str, Any]]) -> None:
    frame = pd.DataFrame(rows)
    figure = go.Figure()
    for column, label, color in (
        ("precision", "Precision", BLUE),
        ("recall", "Recall", TEAL),
        ("f1", "F1", "#7A5AA6"),
    ):
        figure.add_scatter(
            x=frame["threshold"],
            y=frame[column],
            mode="lines+markers",
            name=label,
            line={"color": color},
        )
    figure.update_layout(
        title="Validation metrics across risk thresholds",
        xaxis_title="Risk threshold",
        yaxis_title="Score",
        yaxis={"range": [0, 1]},
        legend={"orientation": "h", "y": -0.25},
        margin={"b": 65},
    )
    st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)
    st.dataframe(
        frame.rename(
            columns={
                "threshold": "Threshold",
                "flagged_orders": "Flagged",
                "true_positives": "True positives",
                "false_positives": "False positives",
                "false_negatives": "False negatives",
                "precision": "Precision",
                "recall": "Recall",
                "f1": "F1",
                "f_beta": "F2",
                "specificity": "Specificity",
            }
        ).round(3),
        hide_index=True,
        width="stretch",
    )


def _render_calibration(
    validation: dict[str, Any], test: dict[str, Any]
) -> None:
    figure = go.Figure()
    figure.add_scatter(
        x=[0, 1],
        y=[0, 1],
        mode="lines",
        name="Perfect calibration",
        line={"color": "#777777", "dash": "dash"},
    )
    for label, summary, color in (
        ("Validation", validation, BLUE),
        ("Test", test, TEAL),
    ):
        bins = summary["bins"]
        if bins:
            frame = pd.DataFrame(bins)
            figure.add_scatter(
                x=frame["mean_predicted_probability"],
                y=frame["observed_late_rate"],
                mode="lines+markers",
                name=f"{label}, Brier {summary['brier_score']:.3f}",
                marker={"size": 7, "color": color},
                line={"color": color},
                customdata=frame[["order_count"]],
                hovertemplate=(
                    "Mean predicted: %{x:.3f}<br>Observed late rate: %{y:.3f}"
                    "<br>Orders: %{customdata[0]:,}<extra></extra>"
                ),
            )
    figure.update_layout(
        title="Reliability by risk-score bin",
        xaxis_title="Mean predicted risk",
        yaxis_title="Observed late-order rate",
        xaxis={"range": [0, 1]},
        yaxis={"range": [0, 1]},
        legend={"orientation": "h", "y": -0.25},
        margin={"b": 65},
    )
    st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)
    st.caption(
        f"Validation Brier score: {validation['brier_score']:.3f} "
        f"(prevalence-only reference {validation['prevalence_only_brier_score']:.3f}); "
        f"test Brier score: {test['brier_score']:.3f}. Calibration is sample-dependent."
    )


def _render_risk_distribution(predictions: pd.DataFrame) -> None:
    figure = px.histogram(
        predictions,
        x="risk_score",
        color="actual_is_late",
        nbins=25,
        barmode="overlay",
        opacity=0.75,
        labels={"risk_score": "Predicted late-order risk", "actual_is_late": "Actually late"},
        title="Held-out risk-score distribution",
        color_discrete_sequence=[BLUE, TEAL],
    )
    figure.update_layout(yaxis_title="Orders", legend_title="Observed outcome")
    st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)


def _render_feature_importance(importance: pd.DataFrame) -> None:
    if importance.empty:
        st.info("Validation permutation importance is unavailable.")
        return
    chart_data = importance.sort_values("pr_auc_decrease", ascending=True)
    figure = px.bar(
        chart_data,
        x="pr_auc_decrease",
        y="feature",
        orientation="h",
        error_x="standard_deviation",
        labels={
            "feature": "Model feature",
            "pr_auc_decrease": "Decrease in validation PR-AUC when shuffled",
            "standard_deviation": "Across permutation repeats",
        },
        title="Validation permutation importance",
        color_discrete_sequence=[BLUE],
    )
    st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)
    st.caption("Importance indicates model reliance, not a causal effect.")


def _render_risk_breakdowns(predictions: pd.DataFrame) -> None:
    breakdown_columns = (
        ("order_hour", "Order hour"),
        ("distance_band", "Distance band"),
        ("weather_context", "Weather (retrospective context)"),
        ("traffic_context", "Traffic (retrospective context)"),
    )
    for column, title in breakdown_columns:
        if column not in predictions:
            continue
        grouped = (
            predictions.groupby(column, dropna=False, observed=True)
            .agg(
                orders=("actual_is_late", "size"),
                observed_late_rate=("actual_is_late", "mean"),
                mean_predicted_risk=("risk_score", "mean"),
            )
            .reset_index()
            .sort_values("orders", ascending=False)
        )
        if column == "order_hour":
            grouped = grouped.sort_values(column, na_position="last")
        st.subheader(f"Late-order risk by {title.lower()}")
        figure = go.Figure()
        figure.add_bar(
            x=grouped[column].astype("string"),
            y=grouped["observed_late_rate"],
            name="Observed late rate",
            marker_color=BLUE,
            customdata=grouped[["orders", "mean_predicted_risk"]],
            hovertemplate=(
                f"{title}: %{{x}}<br>Observed late rate: %{{y:.1%}}"
                "<br>Orders: %{customdata[0]:,}"
                "<br>Mean score: %{customdata[1]:.1%}<extra></extra>"
            ),
        )
        figure.update_layout(
            title=f"Held-out late-order outcomes by {title.lower()}",
            xaxis_title=title,
            yaxis_title="Observed late-order rate",
            yaxis={"tickformat": ".0%"},
        )
        st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)
        if column in {"weather_context", "traffic_context"}:
            st.caption(
                "Retrospective grouping only. This field was excluded from model inputs "
                "because availability at order placement is unverified."
            )
