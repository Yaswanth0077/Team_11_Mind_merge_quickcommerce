"""Interactive Streamlit dashboard for PS7 rider performance segmentation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    RIDER_ORDER_ASSIGNMENTS_PATH,
    RIDER_SEGMENT_ASSIGNMENTS_PATH,
    RIDER_SEGMENT_MODEL_PATH,
    RIDER_SEGMENT_REPORT_PATH,
)
from src.analytics.rider_support import SEGMENT_SUPPORT_ACTIONS

CHART_CONFIG = {"displaylogo": False, "responsive": True}
COLORS = px.colors.qualitative.Safe
ALL_RIDERS = "All riders"
ELIGIBLE_RIDERS = "Eligible segments"
INSUFFICIENT_RIDERS = "Insufficient observations"

@st.cache_data(show_spinner=False)
def _load_rider_artifacts(
    report_path: str,
    report_mtime_ns: int,
    assignments_path: str,
    assignments_mtime_ns: int,
    orders_path: str,
    orders_mtime_ns: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    del report_mtime_ns, assignments_mtime_ns, orders_mtime_ns
    with Path(report_path).open(encoding="utf-8") as file_handle:
        report = json.load(file_handle)
    riders = pd.read_csv(assignments_path)
    orders = pd.read_csv(orders_path)
    if "segment_id" in riders:
        riders["segment_id"] = pd.to_numeric(riders["segment_id"], errors="coerce")
    if "segment_id" in orders:
        orders["segment_id"] = pd.to_numeric(orders["segment_id"], errors="coerce")
    return report, riders, orders


def filter_rider_assignments(
    riders: pd.DataFrame,
    *,
    segment: int | str | None = None,
    delivery_count_range: tuple[int, int] | None = None,
) -> pd.DataFrame:
    """Apply one segment choice and delivery-volume range to rider-level records."""
    selected = riders.copy()
    if segment in (None, ALL_RIDERS):
        pass
    elif segment == ELIGIBLE_RIDERS:
        selected = selected.loc[selected["segment_id"].notna()]
    elif segment == INSUFFICIENT_RIDERS:
        selected = selected.loc[selected["segment_id"].isna()]
    else:
        if isinstance(segment, str) and segment.startswith("Segment "):
            try:
                segment = int(segment.removeprefix("Segment "))
            except ValueError as error:
                raise ValueError(f"Unknown rider segment selection: {segment}") from error
        valid_ids = set(pd.to_numeric(riders["segment_id"], errors="coerce").dropna().astype(int))
        if not isinstance(segment, int) or segment not in valid_ids:
            raise ValueError(f"Unknown rider segment: {segment}")
        selected = selected.loc[selected["segment_id"].eq(segment)]
    if delivery_count_range is not None:
        lower, upper = delivery_count_range
        if lower < 0 or lower > upper:
            raise ValueError("Delivery-count range must satisfy 0 <= minimum <= maximum.")
        selected = selected.loc[
            pd.to_numeric(selected["delivery_count"], errors="coerce").between(
                lower, upper
            )
        ]
    return selected.reset_index(drop=True)


def summarize_selected_segments(
    selected_riders: pd.DataFrame,
    order_assignments: pd.DataFrame,
) -> pd.DataFrame:
    """Recompute segment outcome and rider-volume metrics for the selected population."""
    eligible = selected_riders.loc[selected_riders["segment_id"].notna()].copy()
    if eligible.empty:
        return pd.DataFrame(
            columns=[
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
                "median_deliveries_per_rider",
                "p10_deliveries_per_rider",
                "p90_deliveries_per_rider",
            ]
        )
    rider_ids = set(eligible["rider_id"].astype(str))
    selected_orders = order_assignments.loc[
        order_assignments["rider_id"].astype(str).isin(rider_ids)
    ].copy()
    selected_orders["delivery_minutes"] = pd.to_numeric(
        selected_orders["delivery_minutes"], errors="coerce"
    )
    selected_orders["is_late"] = pd.to_numeric(
        selected_orders["is_late"], errors="coerce"
    )
    selected_orders["delivery_distance_km"] = pd.to_numeric(
        selected_orders["delivery_distance_km"], errors="coerce"
    )
    rider_metrics = eligible.groupby("segment_id", observed=True).agg(
        rider_count=("rider_id", "nunique"),
        mean_rider_rating=("mean_rider_rating", "mean"),
        mean_multiple_deliveries=("mean_multiple_deliveries", "mean"),
        median_deliveries_per_rider=("delivery_count", "median"),
        p10_deliveries_per_rider=("delivery_count", lambda values: values.quantile(0.10)),
        p90_deliveries_per_rider=("delivery_count", lambda values: values.quantile(0.90)),
    )
    order_metrics = (
        selected_orders.groupby("segment_id", observed=True)
        .agg(
            total_deliveries=("delivery_minutes", "count"),
            mean_delivery_minutes=("delivery_minutes", "mean"),
            median_delivery_minutes=("delivery_minutes", "median"),
            late_rate=("is_late", "mean"),
            mean_delivery_distance_km=("delivery_distance_km", "mean"),
        )
    )
    result = rider_metrics.join(order_metrics, how="left").reset_index()
    result["rider_share"] = result["rider_count"] / len(eligible)
    return result.sort_values("segment_id", ignore_index=True)


def render_rider_segmentation_dashboard() -> None:
    """Render the filterable PS7 analysis from saved, reproducible artifacts."""
    st.header("Rider Performance Segmentation")
    st.caption(
        "Retrospective grouping of riders with enough completed deliveries for a more "
        "stable descriptive profile. Segments are not rider rankings or employment judgments."
    )
    paths = (
        RIDER_SEGMENT_REPORT_PATH,
        RIDER_SEGMENT_ASSIGNMENTS_PATH,
        RIDER_ORDER_ASSIGNMENTS_PATH,
        RIDER_SEGMENT_MODEL_PATH,
    )
    if any(not path.is_file() for path in paths):
        st.info(
            "Rider-segmentation artifacts are not available. From the project directory, "
            "run `python -m src.analytics.rider_segmentation`."
        )
        return
    report, riders, orders = _load_rider_artifacts(
        str(RIDER_SEGMENT_REPORT_PATH),
        RIDER_SEGMENT_REPORT_PATH.stat().st_mtime_ns,
        str(RIDER_SEGMENT_ASSIGNMENTS_PATH),
        RIDER_SEGMENT_ASSIGNMENTS_PATH.stat().st_mtime_ns,
        str(RIDER_ORDER_ASSIGNMENTS_PATH),
        RIDER_ORDER_ASSIGNMENTS_PATH.stat().st_mtime_ns,
    )
    segments = sorted(
        pd.to_numeric(riders["segment_id"], errors="coerce").dropna().astype(int).unique()
    )
    options: list[str] = [
        ALL_RIDERS,
        ELIGIBLE_RIDERS,
        INSUFFICIENT_RIDERS,
        *[f"Segment {segment_id}" for segment_id in segments],
    ]
    maximum_deliveries = int(riders["delivery_count"].max())
    first, second = st.columns(2)
    segment_selection = first.selectbox(
        "Rider population or segment",
        options,
        key="rider_segment_filter",
    )
    volume_range = second.slider(
        "Completed deliveries per rider",
        min_value=0,
        max_value=maximum_deliveries,
        value=(0, maximum_deliveries),
        key="rider_volume_filter",
    )
    selected = filter_rider_assignments(
        riders,
        segment=segment_selection,
        delivery_count_range=volume_range,
    )
    selected_segment_metrics = summarize_selected_segments(selected, orders)
    _render_population_summary(riders, orders, report, selected)
    _render_diagnostics(report)
    _render_segment_distribution(selected)
    _render_segment_comparison(selected_segment_metrics, selected)
    _render_segment_metrics(selected_segment_metrics)
    _render_segment_support_actions(selected_segment_metrics)
    _render_rider_details(selected)
    _render_methodology(report, selected_segment_metrics)


def _render_segment_support_actions(metrics: pd.DataFrame) -> None:
    st.subheader("Segment-level support hypotheses")
    st.caption(
        "These are group-level ideas to validate, not proven solutions, individual "
        "rankings, performance judgments, or employment recommendations. Segment "
        "membership must not be used for penalties or dismissal."
    )
    if metrics.empty:
        st.info("Select an eligible rider segment to view its support hypotheses.")
        return
    st.dataframe(build_segment_support_table(metrics), hide_index=True, width="stretch")
    st.caption(
        "The existing segmentation is retrospective, uses a 30-delivery eligibility "
        "screen and has a modest silhouette (0.212). The dataset contains no verified "
        "rider tenure or experience field; age is not used as a substitute."
    )


def build_segment_support_table(metrics: pd.DataFrame) -> pd.DataFrame:
    """Combine measured filtered segment metrics with non-ranking test hypotheses."""
    rows = []
    for segment in metrics.itertuples(index=False):
        segment_id = int(segment.segment_id)
        action = SEGMENT_SUPPORT_ACTIONS.get(segment_id)
        if action is None:
            raise ValueError(f"No support guidance is defined for segment {segment_id}.")
        rows.append(
            {
                "Segment (not a rank)": segment_id,
                "Observed group profile": (
                    f"{int(segment.rider_count):,} riders; "
                    f"{segment.late_rate:.1%} observed late rate; "
                    f"{segment.mean_delivery_minutes:.2f} min mean delivery; "
                    f"{segment.median_deliveries_per_rider:.1f} median deliveries/rider"
                ),
                "Potential operational support": action["support"],
                "Suggested training / process review": action["training"],
                "Roster / workload consideration": action["roster"],
                "Monitoring KPI": action["kpi"],
                "Limitations / validation": action["validation"],
            }
        )
    return pd.DataFrame(
        rows,
        columns=[
            "Segment (not a rank)",
            "Observed group profile",
            "Potential operational support",
            "Suggested training / process review",
            "Roster / workload consideration",
            "Monitoring KPI",
            "Limitations / validation",
        ],
    )


def _render_population_summary(
    riders: pd.DataFrame,
    orders: pd.DataFrame,
    report: dict[str, Any],
    selected: pd.DataFrame,
) -> None:
    audit = report["data_audit"]
    st.subheader("Rider population and selected records")
    population = st.columns(3)
    population[0].metric(
        "Riders with completed deliveries",
        f"{audit['distinct_riders_with_completed_deliveries']:,}",
    )
    population[1].metric(
        "Completed deliveries",
        f"{audit['completed_deliveries_with_valid_rider_id']:,}",
    )
    population[2].metric(
        f"Below {report['metadata']['minimum_deliveries_for_segmentation']}-delivery minimum",
        f"{audit['riders_flagged_insufficient_deliveries']:,}",
    )
    filtered = st.columns(3)
    filtered[0].metric("Riders in selection", f"{selected['rider_id'].nunique():,}")
    selected_riders = set(selected["rider_id"].astype(str))
    delivery_count = int(orders["rider_id"].astype(str).isin(selected_riders).sum())
    filtered[1].metric("Deliveries in selection", f"{delivery_count:,}")
    filtered[2].metric(
        "Riders meeting segmentation minimum",
        f"{int(selected['sufficient_observations'].sum()):,}",
    )
    st.caption(
        "Population counts are fixed; the lower row reflects the current segment and "
        "delivery-volume filters."
    )


def _render_diagnostics(report: dict[str, Any]) -> None:
    metadata = report["metadata"]
    evaluation = pd.DataFrame(report["cluster_count_evaluation"])
    st.subheader("Selecting the number of segments")
    left, right = st.columns(2)
    left.metric("Selected K", str(metadata["selected_cluster_count"]))
    right.metric(
        "Elbow indication",
        str(metadata["inertia_elbow_suggestion"]),
    )
    st.caption(
        f"Selected silhouette: {metadata['selected_silhouette_score']:.3f}. "
        f"Evaluation uses a reproducible sample of "
        f"{metadata['silhouette_sample']['sample_size']:,} eligible riders. "
        "Silhouette and the inertia elbow are diagnostics; inspect the segment sizes "
        "and measured profiles before interpreting them operationally."
    )
    if metadata["selected_silhouette_score"] < 0.25:
        st.warning(
            "The selected silhouette is modest, indicating substantial overlap in "
            "the measured rider profiles. Treat the segments as exploratory summaries, "
            "not sharply separated or naturally occurring rider types."
        )
    if metadata["selection_at_candidate_boundary"]:
        st.warning(
            "The best silhouette is at the highest evaluated K. The tested range may "
            "not contain the best resolution; treat the selected K as exploratory."
        )
    elbow, silhouette = st.columns(2)
    with elbow:
        figure = px.line(
            evaluation,
            x="cluster_count",
            y="inertia",
            markers=True,
            labels={
                "cluster_count": "Number of segments (K)",
                "inertia": "Within-cluster sum of squares",
            },
            title="Inertia / elbow analysis",
            color_discrete_sequence=["#245B8F"],
        )
        figure.add_vline(
            x=metadata["inertia_elbow_suggestion"],
            line_dash="dash",
            line_color="#D58B24",
        )
        st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)
    with silhouette:
        figure = px.line(
            evaluation,
            x="cluster_count",
            y="silhouette_score",
            markers=True,
            labels={
                "cluster_count": "Number of segments (K)",
                "silhouette_score": "Sample silhouette score",
            },
            title="Silhouette analysis",
            color_discrete_sequence=["#168A8A"],
        )
        figure.add_vline(
            x=metadata["selected_cluster_count"],
            line_dash="dash",
            line_color="#D58B24",
        )
        st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)
    st.dataframe(evaluation.round(4), hide_index=True, width="stretch")


def _render_segment_distribution(selected: pd.DataFrame) -> None:
    st.subheader("Rider distribution by segment")
    counts = (
        selected.assign(
            segment_label=selected["segment_id"].map(
                lambda value: f"Segment {int(value)}" if pd.notna(value) else INSUFFICIENT_RIDERS
            )
        )
        .groupby("segment_label", observed=True)
        .size()
        .rename("riders")
        .reset_index()
    )
    if counts.empty:
        st.info("No riders match the current filters.")
        return
    figure = px.bar(
        counts,
        x="segment_label",
        y="riders",
        color="segment_label",
        labels={"segment_label": "Rider group", "riders": "Riders"},
        title="Riders in each selected group",
        color_discrete_sequence=COLORS,
    )
    figure.update_layout(showlegend=False)
    st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)


def _render_segment_comparison(
    metrics: pd.DataFrame,
    selected_riders: pd.DataFrame,
) -> None:
    st.subheader("Segment performance comparisons")
    if metrics.empty:
        st.info(
            "The current selection contains no riders meeting the segmentation minimum."
        )
        return
    left, right = st.columns(2)
    with left:
        figure = go.Figure()
        figure.add_bar(
            x=metrics["segment_id"].map(lambda value: f"Segment {int(value)}"),
            y=metrics["mean_delivery_minutes"],
            name="Mean delivery time",
            marker_color="#245B8F",
            customdata=metrics[["median_delivery_minutes", "total_deliveries"]],
            hovertemplate=(
                "%{x}<br>Mean: %{y:.1f} min"
                "<br>Median: %{customdata[0]:.1f} min"
                "<br>Deliveries: %{customdata[1]:,}<extra></extra>"
            ),
        )
        figure.update_layout(
            title="Delivery time by segment",
            xaxis_title="Segment",
            yaxis_title="Minutes",
        )
        st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)
    with right:
        figure = go.Figure()
        figure.add_bar(
            x=metrics["segment_id"].map(lambda value: f"Segment {int(value)}"),
            y=metrics["late_rate"],
            name="Observed late-delivery rate",
            marker_color="#168A8A",
            customdata=metrics[["rider_count", "total_deliveries"]],
            hovertemplate=(
                "%{x}<br>Late rate: %{y:.1%}"
                "<br>Riders: %{customdata[0]:,}"
                "<br>Deliveries: %{customdata[1]:,}<extra></extra>"
            ),
        )
        figure.update_layout(
            title="Late-delivery rate by segment",
            xaxis_title="Segment",
            yaxis_title="Observed late-delivery rate",
            yaxis={"tickformat": ".0%"},
        )
        st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)
    volume = selected_riders.loc[selected_riders["segment_id"].notna()].copy()
    volume["segment_label"] = volume["segment_id"].map(
        lambda value: f"Segment {int(value)}"
    )
    figure = px.box(
        volume,
        x="segment_label",
        y="delivery_count",
        points="all",
        labels={
            "x": "Segment",
            "delivery_count": "Completed deliveries per rider",
        },
        title="Rider delivery-volume distribution by segment",
        color_discrete_sequence=["#D58B24"],
    )
    st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)


def _render_segment_metrics(metrics: pd.DataFrame) -> None:
    st.subheader("Selected segment metrics")
    if metrics.empty:
        st.info("No eligible segment metrics are available for this filter selection.")
        return
    display = metrics.rename(
        columns={
            "segment_id": "Segment (not a rank)",
            "rider_count": "Riders",
            "rider_share": "Share of selected eligible riders",
            "total_deliveries": "Completed deliveries",
            "mean_delivery_minutes": "Mean delivery time (min)",
            "median_delivery_minutes": "Median delivery time (min)",
            "late_rate": "Late-delivery rate",
            "mean_delivery_distance_km": "Mean valid distance (km)",
            "mean_rider_rating": "Mean rider rating",
            "mean_multiple_deliveries": "Mean multiple-delivery field",
            "median_deliveries_per_rider": "Median deliveries per rider",
            "p10_deliveries_per_rider": "P10 deliveries per rider",
            "p90_deliveries_per_rider": "P90 deliveries per rider",
        }
    )
    st.dataframe(display.round(3), hide_index=True, width="stretch")


def _render_rider_details(selected: pd.DataFrame) -> None:
    st.subheader("Rider-level descriptive details")
    if selected.empty:
        st.info("No riders match the current segment and delivery-volume filters.")
        return
    columns = [
        "rider_id",
        "segment_id",
        "segment_description",
        "delivery_count",
        "sufficiency_status",
        "late_rate",
        "late_rate_ci_low",
        "late_rate_ci_high",
        "sufficient_observations",
        "mean_delivery_minutes",
        "median_delivery_minutes",
        "mean_delivery_distance_km",
        "distance_observation_count",
        "distance_data_sufficient",
        "mean_rider_rating",
        "rating_observation_count",
        "rating_data_sufficient",
        "mean_multiple_deliveries",
        "multiple_delivery_observation_count",
        "multiple_delivery_data_sufficient",
    ]
    display = selected.loc[:, columns].rename(
        columns={
            "rider_id": "Rider ID (group key)",
            "segment_id": "Segment (not a rank)",
            "segment_description": "Measured profile relative to eligible population",
            "delivery_count": "Completed deliveries",
            "sufficiency_status": "Delivery sample status",
            "late_rate": "Late rate",
            "late_rate_ci_low": "Late-rate 95% Wilson CI lower",
            "late_rate_ci_high": "Late-rate 95% Wilson CI upper",
            "sufficient_observations": "Delivery sample sufficient",
            "mean_delivery_minutes": "Mean delivery (min)",
            "median_delivery_minutes": "Median delivery (min)",
            "mean_delivery_distance_km": "Mean valid distance (km)",
            "distance_observation_count": "Valid distance observations",
            "distance_data_sufficient": "Distance sample sufficient",
            "mean_rider_rating": "Mean valid rating",
            "rating_observation_count": "Valid rating observations",
            "rating_data_sufficient": "Rating sample sufficient",
            "mean_multiple_deliveries": "Mean multiple-delivery field",
            "multiple_delivery_observation_count": "Valid multiple-delivery observations",
            "multiple_delivery_data_sufficient": "Multiple-delivery sample sufficient",
        }
    )
    st.dataframe(display.round(3), hide_index=True, width="stretch")


def _render_methodology(
    report: dict[str, Any],
    selected_metrics: pd.DataFrame,
) -> None:
    with st.expander("Segment characteristics, methodology, and limitations"):
        st.write("**Measured segment descriptions**")
        descriptions = pd.DataFrame(report["segments"])
        if selected_metrics.empty:
            st.write("No eligible segment matches the current selection.")
        else:
            selected_ids = set(selected_metrics["segment_id"].astype(int))
            for segment in descriptions.loc[
                descriptions["segment_id"].astype(int).isin(selected_ids)
            ].itertuples(index=False):
                st.write(
                    f"**Segment {int(segment.segment_id)}:** "
                    f"{segment.operational_description}."
                )
        st.write("**Features used**")
        st.code(", ".join(report["metadata"]["feature_names"]))
        st.write(report["metadata"]["preprocessing"])
        st.json(report["metadata"]["feature_policy"])
        st.write(
            f"Eligible riders have at least "
            f"{report['metadata']['minimum_deliveries_for_segmentation']} completed "
            "deliveries. A rider's late-rate uncertainty is shown using a Wilson interval. "
            f"Ratings require {report['metadata']['minimum_rating_observations']} "
            "valid observations; the fitted pipeline median-imputes feature values, while "
            "missingness remains visible in rider-level sufficiency fields and does not "
            "itself define a segment."
        )
        st.caption(
            f"Late delivery means completion strictly greater than the configured "
            f"{report['metadata']['sla_threshold_minutes']}-minute SLA; the SLA is provisional."
        )
        st.write(
            "Mean and median delivery times in the segment table are calculated across "
            "the selected completed-order records. Rider-level profiles use each rider's "
            "own observed feature summaries."
        )
        st.write("**Limitations**")
        for limitation in report["limitations"]:
            st.write(f"- {limitation}")
