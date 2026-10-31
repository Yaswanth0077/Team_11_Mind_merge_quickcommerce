"""Streamlit model-lab page for the trained ETA regression experiment."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    REGRESSION_IMPORTANCE_PATH,
    REGRESSION_MODEL_PATH,
    REGRESSION_PREDICTIONS_PATH,
    REGRESSION_REPORT_PATH,
)

CHART_CONFIG = {"displaylogo": False, "responsive": True}
BLUE = "#245B8F"
TEAL = "#168A8A"


@st.cache_data(show_spinner=False)
def _load_results(
    report_path: str,
    report_mtime_ns: int,
    predictions_path: str,
    predictions_mtime_ns: int,
    importance_path: str,
    importance_mtime_ns: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    del report_mtime_ns, predictions_mtime_ns, importance_mtime_ns
    with Path(report_path).open(encoding="utf-8") as file_handle:
        report = json.load(file_handle)
    predictions = pd.read_csv(predictions_path)
    importance = pd.read_csv(importance_path)
    return report, predictions, importance


def render_regression_dashboard() -> None:
    """Render the saved experiment metrics and held-out diagnostic plots."""
    st.header("Delivery-Time Regression Laboratory")
    st.caption(
        "Offline evaluation of ETA regression models. Results are not a live operational estimate."
    )
    paths = (
        REGRESSION_REPORT_PATH,
        REGRESSION_PREDICTIONS_PATH,
        REGRESSION_IMPORTANCE_PATH,
        REGRESSION_MODEL_PATH,
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        st.info(
            "Regression artifacts are not available yet. From the project directory, "
            "run `python -m src.models.regression` to train and evaluate the models."
        )
        return
    report, predictions, importance = _load_results(
        str(REGRESSION_REPORT_PATH),
        REGRESSION_REPORT_PATH.stat().st_mtime_ns,
        str(REGRESSION_PREDICTIONS_PATH),
        REGRESSION_PREDICTIONS_PATH.stat().st_mtime_ns,
        str(REGRESSION_IMPORTANCE_PATH),
        REGRESSION_IMPORTANCE_PATH.stat().st_mtime_ns,
    )
    selected_name = report["metadata"]["selected_model"]
    selected_metrics = report["selected_model_test_metrics"]
    baseline_metrics = report["baseline_test_metrics"]
    comparison = _comparison_table(report["model_comparison"])

    st.subheader("Held-out test performance")
    metrics = st.columns(5)
    metrics[0].metric("Selected model", selected_name)
    metrics[1].metric("MAE", f"{selected_metrics['mae_minutes']:.2f} min")
    metrics[2].metric("RMSE", f"{selected_metrics['rmse_minutes']:.2f} min")
    metrics[3].metric("R²", f"{selected_metrics['r2']:.3f}")
    improvement = report["selected_model_mae_improvement_percent"]
    metrics[4].metric(
        "MAE vs mean baseline",
        f"{improvement:+.1f}%" if improvement is not None else "Unavailable",
        help="Positive means lower MAE than the mean-prediction baseline.",
    )
    st.caption(
        f"Mean baseline test MAE: {baseline_metrics['mae_minutes']:.2f} min; "
        f"selected-model MAE improvement: "
        f"{report['selected_model_mae_improvement_minutes']:.2f} min. "
        f"Test rows: {report['metadata']['training_configuration']['test_rows']:,}. "
        "Model selection used cross-validation on training data, not held-out test results."
    )

    st.subheader("Model comparison")
    st.dataframe(comparison, hide_index=True, width="stretch")
    st.caption(
        "MAE was the primary selection criterion. RMSE penalizes larger errors; R² is supporting "
        "context, not the sole selection rule. Temporal validation uses only the training partition."
    )

    st.subheader("Selected model diagnostics")
    left, right = st.columns(2)
    with left:
        _render_actual_vs_predicted(predictions)
    with right:
        _render_residual_distribution(predictions)
    _render_error_ranges(report["error_by_delivery_range"])
    _render_feature_importance(importance)

    residuals = report["selected_model_residual_summary"]
    st.subheader("Residual summary")
    residual_metrics = st.columns(4)
    residual_metrics[0].metric("Mean residual", f"{residuals['mean_minutes']:+.2f} min")
    residual_metrics[1].metric("Median residual", f"{residuals['median_minutes']:+.2f} min")
    residual_metrics[2].metric(
        "Residual standard deviation",
        f"{residuals['standard_deviation_minutes']:.2f} min",
    )
    residual_metrics[3].metric(
        "95th percentile absolute error",
        f"{residuals['p95_absolute_minutes']:.2f} min",
    )

    with st.expander("Feature availability and experiment configuration"):
        policy = report["metadata"]["feature_availability_policy"]
        st.dataframe(
            pd.DataFrame(
                [
                    {
                        "Feature": feature,
                        "Policy": policy[feature],
                        "Used": feature in report["metadata"]["feature_columns"],
                    }
                    for feature in policy
                ]
            ),
            hide_index=True,
            width="stretch",
        )
        st.json(report["metadata"]["training_configuration"])
    st.caption(report["interpretation"])


def _comparison_table(rows: list[dict[str, Any]]) -> pd.DataFrame:
    records = []
    for row in rows:
        metrics = row["test_metrics"]
        temporal = row["temporal_validation_metrics"] or {}
        records.append(
            {
                "Model": row["model"],
                "CV MAE (min)": row["cv_mae_minutes"],
                "Temporal MAE (min)": temporal.get("mae_minutes"),
                "Test MAE (min)": metrics["mae_minutes"],
                "Test RMSE (min)": metrics["rmse_minutes"],
                "Test R²": metrics["r2"],
                "Test median absolute error (min)": metrics[
                    "median_absolute_error_minutes"
                ],
            }
        )
    return pd.DataFrame(records).sort_values(
        ["CV MAE (min)", "Model"], ignore_index=True
    ).round(3)


def _render_actual_vs_predicted(predictions: pd.DataFrame) -> None:
    figure = go.Figure(
        go.Scattergl(
            x=predictions["actual_minutes"],
            y=predictions["predicted_minutes"],
            mode="markers",
            marker={"size": 6, "opacity": 0.45, "color": BLUE},
            customdata=predictions[["residual_minutes"]],
            hovertemplate=(
                "Actual: %{x:.1f} min<br>Predicted: %{y:.1f} min"
                "<br>Residual: %{customdata[0]:+.1f} min<extra></extra>"
            ),
            name="Held-out orders",
        )
    )
    low = float(
        min(predictions["actual_minutes"].min(), predictions["predicted_minutes"].min())
    )
    high = float(
        max(predictions["actual_minutes"].max(), predictions["predicted_minutes"].max())
    )
    figure.add_trace(
        go.Scatter(
            x=[low, high],
            y=[low, high],
            mode="lines",
            line={"color": TEAL, "dash": "dash"},
            name="Perfect prediction",
            hoverinfo="skip",
        )
    )
    figure.update_layout(
        title="Actual versus predicted delivery time",
        xaxis_title="Actual delivery time (minutes)",
        yaxis_title="Predicted delivery time (minutes)",
        legend={"orientation": "h", "y": -0.25},
        margin={"b": 65},
    )
    st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)


def _render_residual_distribution(predictions: pd.DataFrame) -> None:
    figure = px.histogram(
        predictions,
        x="residual_minutes",
        nbins=35,
        labels={"residual_minutes": "Residual (actual − predicted, minutes)"},
        title="Held-out residual distribution",
        color_discrete_sequence=[BLUE],
    )
    figure.add_vline(x=0, line_dash="dash", line_color=TEAL)
    figure.update_layout(yaxis_title="Orders", bargap=0.08)
    st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)


def _render_error_ranges(ranges: list[dict[str, Any]]) -> None:
    st.subheader("Prediction error by actual delivery-time range")
    if not ranges:
        st.info("No held-out predictions are available for this comparison.")
        return
    frame = pd.DataFrame(ranges)
    figure = go.Figure()
    figure.add_bar(
        x=frame["delivery_range"],
        y=frame["mae_minutes"],
        name="MAE",
        marker_color=BLUE,
        customdata=frame[["order_count"]],
        hovertemplate="Range: %{x} min<br>MAE: %{y:.2f} min<br>Orders: %{customdata[0]:,}<extra></extra>",
    )
    figure.add_scatter(
        x=frame["delivery_range"],
        y=frame["rmse_minutes"],
        mode="lines+markers",
        name="RMSE",
        line={"color": TEAL},
        hovertemplate="Range: %{x} min<br>RMSE: %{y:.2f} min<extra></extra>",
    )
    figure.update_layout(
        title="Error across delivery-time ranges",
        xaxis_title="Actual delivery-time range (minutes)",
        yaxis_title="Prediction error (minutes)",
        legend={"orientation": "h", "y": -0.25},
        margin={"b": 65},
    )
    st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)
    st.dataframe(
        frame.rename(
            columns={
                "delivery_range": "Actual delivery range (min)",
                "order_count": "Orders",
                "mae_minutes": "MAE (min)",
                "rmse_minutes": "RMSE (min)",
            }
        ).round(3),
        hide_index=True,
        width="stretch",
    )


def _render_feature_importance(importance: pd.DataFrame) -> None:
    st.subheader("Permutation importance")
    if importance.empty:
        st.info("Permutation importance was not calculated.")
        return
    chart_data = importance.sort_values("mae_increase_minutes", ascending=True)
    figure = px.bar(
        chart_data,
        x="mae_increase_minutes",
        y="feature",
        orientation="h",
        error_x="standard_deviation_minutes",
        labels={
            "feature": "Model feature",
            "mae_increase_minutes": "Increase in test MAE when shuffled (minutes)",
            "standard_deviation_minutes": "Across permutation repeats",
        },
        title="Held-out permutation importance",
        color_discrete_sequence=[BLUE],
    )
    st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)
    st.caption(
        "A larger MAE increase indicates greater model reliance in this test sample. "
        "Importance does not establish a causal effect."
    )
