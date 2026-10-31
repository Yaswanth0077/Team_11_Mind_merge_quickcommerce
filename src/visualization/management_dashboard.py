"""Final evidence-integrated management dashboard for PS8."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import streamlit as st

from config import MANAGEMENT_INSIGHTS_REPORT_PATH
from src.analytics.management_insights import build_management_report, write_management_report

CHART_CONFIG = {"displaylogo": False, "responsive": True}
BLUE = "#245B8F"
TEAL = "#168A8A"


@st.cache_data(show_spinner=False)
def _load_management_report(
    report_path: str,
    report_mtime_ns: int,
) -> dict[str, Any]:
    del report_mtime_ns
    with Path(report_path).open(encoding="utf-8") as file_handle:
        return json.load(file_handle)


def render_management_dashboard() -> None:
    """Render model evidence, PS1–PS7 synthesis, and the three approved actions."""
    st.header("Management Insights")
    st.caption(
        "Final synthesis of saved PS1–PS7 evidence. Retrospective associations are not "
        "causal findings; recommendations are hypotheses to validate, not guaranteed benefits."
    )
    if not MANAGEMENT_INSIGHTS_REPORT_PATH.is_file():
        st.info(
            "The verified PS8 report has not been generated. From the project directory, "
            "run `python -m src.analytics.management_insights`."
        )
        return
    report = _load_management_report(
        str(MANAGEMENT_INSIGHTS_REPORT_PATH),
        MANAGEMENT_INSIGHTS_REPORT_PATH.stat().st_mtime_ns,
    )
    _render_dataset_summary(report)
    _render_statement_status(report)
    _render_regression(report)
    _render_classification(report)
    _render_explainability(report)
    _render_prior_work_evidence(report)
    _render_recommendations(report)
    _render_limitations(report)


def _render_dataset_summary(report: dict[str, Any]) -> None:
    dataset = report["dataset"]
    st.subheader("Evidence base")
    metrics = st.columns(4)
    metrics[0].metric("Supplied records", f"{dataset['rows']:,}")
    metrics[1].metric("Observed late orders", f"{dataset['late_rate']:.1%}")
    metrics[2].metric("Mean delivery time", f"{dataset['mean_delivery_minutes']:.2f} min")
    metrics[3].metric(
        "Valid delivery distances",
        f"{dataset['valid_distance_observations']:,}",
        help=(
            f"{dataset['invalid_or_unavailable_distance_observations']:,} records had "
            "invalid or unavailable distance."
        ),
    )
    st.warning(
        f"The source is food-delivery data used as a quick-commerce proxy. "
        f"The {report['metadata']['sla_threshold_minutes']}-minute SLA remains "
        f"{dataset['sla_status']}."
    )


def _render_statement_status(report: dict[str, Any]) -> None:
    with st.expander("PS1–PS8 scope and completion status", expanded=False):
        st.dataframe(
            pd.DataFrame(report["problem_statement_status"]),
            hide_index=True,
            width="stretch",
        )
        st.caption(
            "The official challenge guide was reviewed from the workspace-root TXT file. PS2 has "
            "a standalone prediction contract and saved-model demo; the 30-minute SLA remains "
            "provisional because the guide does not publish an approved threshold."
        )


def _render_regression(report: dict[str, Any]) -> None:
    regression = report["regression"]
    st.subheader("Delivery-time model selection")
    comparison = pd.DataFrame(
        [
            {
                "Model": row["model"],
                "Training CV MAE (min)": row["cv_mae_minutes"],
                "Held-out test MAE (min)": row["test_metrics"]["mae_minutes"],
                "Held-out test RMSE (min)": row["test_metrics"]["rmse_minutes"],
                "Held-out test R²": row["test_metrics"]["r2"],
            }
            for row in regression["model_comparison"]
        ]
    )
    st.dataframe(comparison.round(3), hide_index=True, width="stretch")
    selected_name = st.selectbox(
        "Inspect regression candidate",
        options=[row["model"] for row in regression["model_comparison"]],
        index=[row["model"] for row in regression["model_comparison"]].index(
            regression["selected_model"]
        ),
        key="management_regression_model",
    )
    row = next(item for item in regression["model_comparison"] if item["model"] == selected_name)
    details = st.columns(4)
    details[0].metric("CV MAE", f"{row['cv_mae_minutes']:.2f} min")
    details[1].metric("Test MAE", f"{row['test_metrics']['mae_minutes']:.2f} min")
    details[2].metric("Test RMSE", f"{row['test_metrics']['rmse_minutes']:.2f} min")
    details[3].metric("Test R²", f"{row['test_metrics']['r2']:.3f}")
    st.caption(
        f"Selected model: {regression['selected_model']}. "
        f"{regression['selection_criterion']} Mean-baseline test MAE was "
        f"{regression['baseline_test_metrics']['mae_minutes']:.2f} minutes; "
        f"the selected model improves MAE by "
        f"{regression['mae_improvement_vs_baseline_minutes']:.2f} minutes "
        f"({regression['mae_improvement_vs_baseline_percent']:.1f}%). "
        "Gradient Boosting and XGBoost test results are nearly tied; the project choice "
        "follows training-only cross-validation, not the held-out test."
    )
    figure = px.bar(
        comparison,
        x="Model",
        y=["Training CV MAE (min)", "Held-out test MAE (min)"],
        barmode="group",
        title="Regression error across evaluated models",
        color_discrete_sequence=[BLUE, TEAL],
    )
    figure.update_layout(yaxis_title="Mean absolute error (minutes)", legend_title_text="")
    st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)
    st.caption(
        "MAE is average absolute error in minutes. RMSE also uses minutes but weights "
        "larger errors more. R² compares explained test variation with a mean baseline. "
        "The selected tree ensemble is less directly interpretable than a linear model and "
        "does not guarantee an accurate estimate for an individual delivery."
    )


def _render_classification(report: dict[str, Any]) -> None:
    classification = report["classification"]
    st.subheader("Late-order model and operating point")
    comparison = pd.DataFrame(
        [
            {
                "Model": row["model"],
                "CV average precision": row["cv_average_precision"],
                "Validation threshold": row["validation_metrics"]["threshold"],
                "Validation precision": row["validation_metrics"]["precision"],
                "Validation recall": row["validation_metrics"]["recall"],
                "Validation F1": row["validation_metrics"]["f1"],
                "Validation F2": row["validation_metrics"]["f_beta"],
                "Validation ROC-AUC": row["validation_metrics"]["roc_auc"],
            }
            for row in classification["model_comparison"]
        ]
    )
    st.dataframe(comparison.round(3), hide_index=True, width="stretch")
    selected_name = st.selectbox(
        "Inspect classifier validation results",
        options=[row["model"] for row in classification["model_comparison"]],
        index=[row["model"] for row in classification["model_comparison"]].index(
            classification["selected_model"]
        ),
        key="management_classification_model",
    )
    validation_row = next(
        item for item in classification["model_comparison"] if item["model"] == selected_name
    )
    validation_metrics = validation_row["validation_metrics"]
    validation_cards = st.columns(4)
    validation_cards[0].metric("Validation threshold", f"{validation_metrics['threshold']:.2f}")
    validation_cards[1].metric("Validation precision", f"{validation_metrics['precision']:.1%}")
    validation_cards[2].metric("Validation recall", f"{validation_metrics['recall']:.1%}")
    validation_cards[3].metric("Validation F2", f"{validation_metrics['f_beta']:.3f}")
    st.caption(
        f"{classification['selected_model']} is selected because it achieved the highest "
        "validation F2, with recall as the next tie-break criterion. XGBoost had the highest "
        "cross-validation average precision, but validation F2—not the test set—determined "
        "the selected model. F2 weights recall more heavily than precision."
    )
    threshold_table = pd.DataFrame(
        [
            {
                "Validation threshold": row["threshold"],
                "Flagged orders": row["flagged_orders"],
                "Precision": row["precision"],
                "Recall": row["recall"],
                "F1": row["f1"],
                "F2": row["f_beta"],
                "False positives": row["false_positives"],
                "False negatives": row["false_negatives"],
            }
            for row in classification["validation_threshold_comparison"]
        ]
    )
    with st.expander("Training-only validation threshold comparison", expanded=False):
        st.dataframe(threshold_table.round(3), hide_index=True, width="stretch")
        selected_validation = classification["selected_validation_metrics"]
        conventional = next(
            row
            for row in classification["validation_threshold_comparison"]
            if row["threshold"] == 0.5
        )
        st.caption(
            f"The selected {classification['selected_threshold']:.2f} validation threshold "
            f"achieved precision {selected_validation['precision']:.1%}, recall "
            f"{selected_validation['recall']:.1%}, and F2 {selected_validation['f_beta']:.3f}. "
            f"At 0.50, precision was {conventional['precision']:.1%}, recall "
            f"{conventional['recall']:.1%}, and F2 {conventional['f_beta']:.3f}. "
            "The lower threshold accepts more false-positive alerts to reduce missed late orders."
        )
        st.caption(
            "The selected threshold is fixed at the value chosen using training-only "
            "validation. Test labels are not available to the threshold selector."
        )

    test = classification["selected_test_metrics"]
    st.markdown("**Locked selected-model held-out test result**")
    st.caption(
        f"{classification['selected_model']} at threshold "
        f"{classification['selected_threshold']:.2f}; this test result is for evaluation, "
        "not threshold selection."
    )
    test_cards = st.columns(5)
    test_cards[0].metric("Precision", f"{test['precision']:.1%}")
    test_cards[1].metric("Recall", f"{test['recall']:.1%}")
    test_cards[2].metric("F1", f"{test['f1']:.3f}")
    test_cards[3].metric("ROC-AUC", f"{test['roc_auc']:.3f}")
    test_cards[4].metric(
        "Flagged",
        f"{test['flagged_orders']:,} ({test['flagged_share']:.1%})",
    )
    confusion = pd.DataFrame(
        [
            [test["true_negatives"], test["false_positives"]],
            [test["false_negatives"], test["true_positives"]],
        ],
        index=["Actual on time", "Actual late"],
        columns=["Predicted on time", "Predicted late"],
    )
    st.dataframe(confusion, width="stretch")
    st.caption(
        f"Test false positives: {test['false_positives']:,}; test false negatives: "
        f"{test['false_negatives']:,}. Precision is the share of alerts that were late; "
        "recall is the share of late orders that were flagged. F1 is their harmonic mean; "
        "F2 weights recall more heavily. ROC-AUC measures ranking discrimination across "
        "thresholds, not calibration or an operating-point decision. Confusion-matrix rows "
        "are actual on-time/late and columns are predicted on-time/late. High recall does not "
        "by itself establish operational efficiency. The classifier flags risk; it does not "
        "prevent delays."
    )


def _render_explainability(report: dict[str, Any]) -> None:
    st.subheader("Model explainability")
    explainability = report["explainability"]
    regression = pd.DataFrame(explainability["regression_importance"])
    classification = pd.DataFrame(explainability["classification_importance"])
    left, right = st.columns(2)
    with left:
        figure = px.bar(
            regression.sort_values("mae_increase_minutes"),
            x="mae_increase_minutes",
            y="feature",
            orientation="h",
            title="Regression: held-out MAE increase after permutation",
            labels={
                "mae_increase_minutes": "MAE increase (minutes)",
                "feature": "Feature",
            },
            color_discrete_sequence=[BLUE],
        )
        st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)
    with right:
        figure = px.bar(
            classification.sort_values("pr_auc_decrease"),
            x="pr_auc_decrease",
            y="feature",
            orientation="h",
            title="Classification: held-out average-precision decrease",
            labels={
                "pr_auc_decrease": "Average-precision decrease",
                "feature": "Feature",
            },
            color_discrete_sequence=[TEAL],
        )
        st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)
    st.caption(explainability["methodology"])
    st.warning(explainability["feature_availability_warning"])


def _render_prior_work_evidence(report: dict[str, Any]) -> None:
    st.subheader("Evidence from PS3–PS7")
    delay = report["ps3_findings"]
    left, right = st.columns(2)
    with left:
        traffic = pd.DataFrame(
            [row for row in delay["traffic"] if row["analysis_traffic"] != "(Missing)"]
        )
        figure = px.bar(
            traffic,
            x="analysis_traffic",
            y="late_rate",
            hover_data=["order_count", "mean_delivery_minutes"],
            title="PS3 observed late rate by recorded traffic category",
            labels={"analysis_traffic": "Recorded traffic", "late_rate": "Late-order rate"},
            color_discrete_sequence=[BLUE],
        )
        figure.update_yaxes(tickformat=".0%")
        st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)
    with right:
        distance = pd.DataFrame(delay["distance_bands"])
        figure = px.bar(
            distance,
            x="analysis_distance_band",
            y="late_rate",
            hover_data=["order_count", "mean_delivery_minutes"],
            title="PS3 observed late rate by valid distance band",
            labels={
                "analysis_distance_band": "Distance band",
                "late_rate": "Late-order rate",
            },
            color_discrete_sequence=[TEAL],
        )
        figure.update_yaxes(tickformat=".0%")
        st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)
    st.caption(delay["interpretation"])

    zones = report["geographic_analysis"]
    st.markdown("**PS6 exploratory zone ranking**")
    st.dataframe(
        pd.DataFrame(zones["candidate_zone_ranking"])[
            [
                "cluster_id",
                "candidate_priority_score",
                "order_count",
                "demand_share",
                "mean_delivery_minutes",
                "late_rate",
                "average_delivery_distance_km",
                "nearest_restaurant_km",
            ]
        ].round(3),
        hide_index=True,
        width="stretch",
    )
    st.warning(
        f"{zones['selection_warning']} Restaurants are not verified dark stores, and the "
        "weighted candidate score is not a site-feasibility or financial-return measure."
    )

    riders = report["rider_analysis"]
    segments = riders["segments"]
    st.markdown("**PS7 rider-segment profile**")
    segment_ids = [int(row["segment_id"]) for row in segments]
    chosen_segment = st.selectbox(
        "Inspect descriptive rider segment",
        segment_ids,
        key="management_rider_segment",
    )
    selected = next(row for row in segments if int(row["segment_id"]) == chosen_segment)
    segment_metrics = st.columns(4)
    segment_metrics[0].metric("Riders", f"{selected['rider_count']:,}")
    segment_metrics[1].metric("Deliveries", f"{selected['total_deliveries']:,}")
    segment_metrics[2].metric(
        "Mean delivery time",
        f"{selected['mean_delivery_minutes']:.2f} min",
    )
    segment_metrics[3].metric("Late rate", f"{selected['late_rate']:.1%}")
    st.dataframe(
        pd.DataFrame(segments)[
            [
                "segment_id",
                "rider_count",
                "rider_share",
                "total_deliveries",
                "mean_delivery_minutes",
                "median_delivery_minutes",
                "late_rate",
                "mean_delivery_distance_km",
                "mean_rider_rating",
                "mean_multiple_deliveries",
            ]
        ].round(3),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"{riders['segmented_riders']:,} of {riders['total_riders']:,} riders met the "
        f"{riders['minimum_deliveries']}-delivery threshold; "
        f"{riders['insufficient_sample_riders']:,} were flagged as insufficient. "
        f"Silhouette={riders['selected_silhouette_score']:.3f}. These segments are "
        "descriptive and are not individual rider rankings or employment judgments."
    )


def _render_recommendations(report: dict[str, Any]) -> None:
    st.subheader("Prioritised management actions")
    for index, recommendation in enumerate(report["recommendations"], start=1):
        with st.container(border=True):
            st.markdown(
                f"**{index}. {recommendation['title']}**  \n"
                f"Priority: **{recommendation['priority']}**"
            )
            st.markdown(f"**Evidence:** {recommendation['evidence']}")
            st.markdown(f"**Problem:** {recommendation['problem']}")
            st.markdown(f"**Action:** {recommendation['action']}")
            st.markdown(f"**Expected benefit:** {recommendation['expected_benefit']}")
            st.markdown("**Implementation requirements:**")
            for item in recommendation["implementation"]:
                st.markdown(f"- {item}")
            st.markdown("**Risks and limitations:**")
            for item in recommendation["risks"]:
                st.markdown(f"- {item}")
            st.markdown(f"**Monitoring KPI:** {recommendation['monitoring_kpi']}")
            st.markdown(f"**Validation:** {recommendation['validation']}")


def _render_limitations(report: dict[str, Any]) -> None:
    with st.expander("Methodology, assumptions, and limitations"):
        for item in report["limitations"]:
            st.markdown(f"- {item}")
        st.caption(
            "The management report verifies existing saved artifacts; it does not retrain "
            "the regression/classification models or change the selected threshold."
        )
