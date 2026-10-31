"""Interactive Streamlit page for exploratory customer-demand zones."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    DARK_STORE_ASSIGNMENTS_PATH,
    DARK_STORE_MODEL_PATH,
    DARK_STORE_REPORT_PATH,
    DARK_STORE_RESTAURANTS_PATH,
)

CHART_CONFIG = {"displaylogo": False, "responsive": True}
COLORS = px.colors.qualitative.Safe


@st.cache_data(show_spinner=False)
def _load_zone_artifacts(
    report_path: str,
    report_mtime_ns: int,
    assignments_path: str,
    assignments_mtime_ns: int,
    restaurants_path: str,
    restaurants_mtime_ns: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    del report_mtime_ns, assignments_mtime_ns, restaurants_mtime_ns
    with Path(report_path).open(encoding="utf-8") as file_handle:
        report = json.load(file_handle)
    return (
        report,
        pd.read_csv(assignments_path),
        pd.read_csv(restaurants_path),
    )


def filter_zone_assignments(
    assignments: pd.DataFrame, cluster_id: int | None
) -> pd.DataFrame:
    """Return all assigned orders or a single selected cluster."""
    if cluster_id is None:
        return assignments.copy()
    if cluster_id not in set(assignments["cluster_id"].astype(int).unique()):
        raise ValueError(f"Unknown cluster id: {cluster_id}")
    return assignments.loc[assignments["cluster_id"].eq(cluster_id)].copy()


def aggregate_customer_locations(assignments: pd.DataFrame) -> pd.DataFrame:
    """Aggregate coincident order points to weighted markers on the demand map."""
    if assignments.empty:
        return pd.DataFrame(
            columns=[
                "customer_latitude",
                "customer_longitude",
                "cluster_id",
                "order_count",
                "late_rate",
                "mean_delivery_minutes",
            ]
        )
    return (
        assignments.groupby(
            ["customer_latitude", "customer_longitude", "cluster_id"],
            observed=True,
            as_index=False,
        )
        .agg(
            order_count=("source_row_number", "size"),
            late_rate=("is_late", "mean"),
            mean_delivery_minutes=("delivery_minutes", "mean"),
        )
    )


def render_demand_zone_dashboard() -> None:
    """Render cluster selection, map, performance, diagnostics, and evidence ranking."""
    st.header("Dark-Store Demand Zone Intelligence")
    st.caption(
        "Exploratory customer-order density zones from validated coordinates. "
        "Clusters can prioritize further investigation; they do not prove an optimal site."
    )
    paths = (
        DARK_STORE_REPORT_PATH,
        DARK_STORE_ASSIGNMENTS_PATH,
        DARK_STORE_RESTAURANTS_PATH,
        DARK_STORE_MODEL_PATH,
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        st.info(
            "Demand-zone artifacts are not available. From the project directory, "
            "run `python -m src.analytics.demand_zones`."
        )
        return
    report, assignments, restaurants = _load_zone_artifacts(
        str(DARK_STORE_REPORT_PATH),
        DARK_STORE_REPORT_PATH.stat().st_mtime_ns,
        str(DARK_STORE_ASSIGNMENTS_PATH),
        DARK_STORE_ASSIGNMENTS_PATH.stat().st_mtime_ns,
        str(DARK_STORE_RESTAURANTS_PATH),
        DARK_STORE_RESTAURANTS_PATH.stat().st_mtime_ns,
    )
    all_clusters = pd.DataFrame(report["clusters"])
    ranked = pd.DataFrame(report["candidate_zone_ranking"])
    evaluation = pd.DataFrame(report["cluster_count_evaluation"])
    cluster_ids = sorted(assignments["cluster_id"].astype(int).unique().tolist())
    selector_labels = ["All zones"] + [f"Zone {cluster_id}" for cluster_id in cluster_ids]
    selection = st.selectbox("Demand zone", selector_labels, key="demand_zone_selection")
    selected_cluster = (
        None if selection == "All zones" else int(selection.removeprefix("Zone "))
    )
    selected_orders = filter_zone_assignments(assignments, selected_cluster)
    selected_summary = (
        all_clusters
        if selected_cluster is None
        else all_clusters.loc[all_clusters["cluster_id"].eq(selected_cluster)]
    )

    _render_kpis(selected_orders, selected_summary, report["network_benchmarks"])
    _render_geographic_map(
        selected_orders,
        restaurants,
        selected_summary,
    )
    _render_performance(selected_summary, report["network_benchmarks"])
    _render_diagnostics(evaluation, all_clusters, report)
    _render_candidate_ranking(ranked)
    with st.expander("Coordinate audit and scoring assumptions"):
        st.json(report["data_audit"])
        st.json(report["metadata"]["score_components_and_weights"])
        st.write(report["metadata"]["score_interpretation"])
        st.write("**Site-selection limitations**")
        for limitation in report["limitations"]:
            st.write(f"- {limitation}")


def _render_kpis(
    orders: pd.DataFrame,
    clusters: pd.DataFrame,
    network: dict[str, Any],
) -> None:
    count = int(len(orders))
    if count:
        mean_time = float(orders["delivery_minutes"].mean())
        late_rate = float(orders["is_late"].mean())
        distance = pd.to_numeric(orders["delivery_distance_km"], errors="coerce").mean()
        demand_share = count / int(network["clustered_orders"])
    else:
        mean_time = late_rate = demand_share = float("nan")
        distance = float("nan")
    title = "Selected zone" if len(clusters) == 1 else "All valid demand zones"
    st.subheader(title)
    cards = st.columns(5)
    cards[0].metric("Orders", f"{count:,}")
    cards[1].metric("Demand share", _format_metric(demand_share, ".1%"))
    cards[2].metric("Average delivery", _format_metric(mean_time, ".1f", " min"))
    cards[3].metric("Late-order rate", _format_metric(late_rate, ".1%"))
    cards[4].metric(
        "Average valid distance", _format_metric(distance, ".1f", " km")
    )
    if len(clusters) == 1:
        row = clusters.iloc[0]
        st.caption(
            f"Zone {int(row['cluster_id'])} has {int(row['unique_riders']):,} distinct "
            "rider IDs in its orders. Order counts are not unique-customer counts."
        )
    st.caption(
        "Network benchmark: "
        f"{_format_metric(network['mean_delivery_minutes'], '.1f', ' min')} average delivery; "
        f"{_format_metric(network['late_rate'], '.1%')} late-order rate; "
        f"{_format_metric(network['average_delivery_distance_km'], '.1f', ' km')} "
        "average valid distance."
    )


def _format_metric(value: Any, number_format: str, suffix: str = "") -> str:
    return f"{value:{number_format}}{suffix}" if pd.notna(value) else "Not available"


def _render_geographic_map(
    orders: pd.DataFrame,
    restaurants: pd.DataFrame,
    cluster_summary: pd.DataFrame,
) -> None:
    st.subheader("Customer demand and existing restaurant locations")
    points = aggregate_customer_locations(orders)
    if points.empty:
        st.info("No customer locations are available for this selection.")
        return
    figure = go.Figure()
    for cluster_id in sorted(points["cluster_id"].unique()):
        group = points.loc[points["cluster_id"].eq(cluster_id)]
        figure.add_trace(
            go.Scattergeo(
                lat=group["customer_latitude"],
                lon=group["customer_longitude"],
                mode="markers",
                name=f"Zone {int(cluster_id)} customer orders",
                marker={
                    "size": (group["order_count"].clip(lower=1) ** 0.5 * 3).clip(4, 18),
                    "color": COLORS[int(cluster_id) % len(COLORS)],
                    "opacity": 0.72,
                    "line": {"width": 0},
                },
                customdata=group[
                    ["order_count", "late_rate", "mean_delivery_minutes"]
                ],
                hovertemplate=(
                    f"Zone {int(cluster_id)}<br>Latitude: %{{lat:.5f}}"
                    "<br>Longitude: %{lon:.5f}<br>Orders: %{customdata[0]:,}"
                    "<br>Observed late rate: %{customdata[1]:.1%}"
                    "<br>Mean delivery: %{customdata[2]:.1f} min<extra></extra>"
                ),
            )
        )
    if not restaurants.empty:
        figure.add_trace(
            go.Scattergeo(
                lat=restaurants["restaurant_latitude"],
                lon=restaurants["restaurant_longitude"],
                mode="markers",
                name="Supplied restaurant locations",
                marker={
                    "size": 8,
                    "symbol": "square",
                    "color": "#303030",
                    "opacity": 0.72,
                },
                customdata=restaurants[["orders_at_location"]],
                hovertemplate=(
                    "Restaurant location (coverage proxy)<br>Latitude: %{lat:.5f}"
                    "<br>Longitude: %{lon:.5f}<br>Orders at location: %{customdata[0]:,}"
                    "<extra></extra>"
                ),
            )
        )
    if not cluster_summary.empty:
        figure.add_trace(
            go.Scattergeo(
                lat=cluster_summary["centroid_latitude"],
                lon=cluster_summary["centroid_longitude"],
                mode="markers+text",
                text=cluster_summary["cluster_id"].map(lambda value: f"Zone {int(value)}"),
                textposition="top center",
                name="Demand-weighted cluster centroids",
                marker={
                    "size": 15,
                    "symbol": "star",
                    "color": "#D58B24",
                    "line": {"color": "white", "width": 1},
                },
                customdata=cluster_summary[
                    ["order_count", "demand_share", "mean_delivery_minutes", "late_rate"]
                ],
                hovertemplate=(
                    "Demand-weighted centroid<br>Zone: %{text}"
                    "<br>Orders: %{customdata[0]:,}"
                    "<br>Demand share: %{customdata[1]:.1%}"
                    "<br>Mean delivery: %{customdata[2]:.1f} min"
                    "<br>Late rate: %{customdata[3]:.1%}<extra></extra>"
                ),
            )
        )
    all_points = pd.concat(
        [
            points[["customer_latitude", "customer_longitude"]].rename(
                columns={"customer_latitude": "lat", "customer_longitude": "lon"}
            ),
            restaurants[["restaurant_latitude", "restaurant_longitude"]].rename(
                columns={"restaurant_latitude": "lat", "restaurant_longitude": "lon"}
            ),
        ],
        ignore_index=True,
    )
    lat_min, lat_max = all_points["lat"].min(), all_points["lat"].max()
    lon_min, lon_max = all_points["lon"].min(), all_points["lon"].max()
    figure.update_layout(
        title="Validated customer order locations, centroids, and restaurant proxy",
        geo={
            "scope": "asia",
            "projection_type": "mercator",
            "showland": True,
            "landcolor": "#F1F4F7",
            "showocean": True,
            "oceancolor": "#EAF2F8",
            "showcountries": True,
            "fitbounds": "locations",
            "lataxis_range": [lat_min - 0.2, lat_max + 0.2],
            "lonaxis_range": [lon_min - 0.2, lon_max + 0.2],
        },
        legend={"orientation": "h", "y": -0.08},
        margin={"l": 0, "r": 0, "t": 55, "b": 45},
    )
    st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)
    st.caption(
        "Marker size reflects order count at an exact coordinate. Coordinates are locations, "
        "not verified distinct customers. Restaurant points are a separate layer."
    )


def _render_performance(
    clusters: pd.DataFrame, network: dict[str, Any]
) -> None:
    st.subheader("Cluster delivery performance")
    if clusters.empty:
        st.info("No cluster metrics are available.")
        return
    displayed = clusters.sort_values("cluster_id")
    left, right = st.columns(2)
    with left:
        figure = px.bar(
            displayed,
            x="cluster_id",
            y="order_count",
            color="cluster_id",
            labels={"cluster_id": "Demand zone", "order_count": "Orders"},
            title="Orders by demand zone",
            color_discrete_sequence=COLORS,
        )
        figure.update_layout(showlegend=False)
        st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)
    with right:
        figure = go.Figure()
        figure.add_bar(
            x=displayed["cluster_id"],
            y=displayed["mean_delivery_minutes"],
            name="Average delivery time",
            marker_color="#245B8F",
            customdata=displayed[["order_count", "delivery_time_difference_vs_network_minutes"]],
            hovertemplate=(
                "Zone %{x}<br>Average: %{y:.1f} min<br>Orders: %{customdata[0]:,}"
                "<br>Difference vs network: %{customdata[1]:+.1f} min<extra></extra>"
            ),
        )
        figure.add_scatter(
            x=displayed["cluster_id"],
            y=[network["mean_delivery_minutes"]] * len(displayed),
            mode="lines",
            name="Network mean",
            line={"color": "#168A8A", "dash": "dash"},
        )
        figure.update_layout(
            title="Average delivery time by zone",
            xaxis_title="Demand zone",
            yaxis_title="Minutes",
        )
        st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)
    late = go.Figure()
    late.add_bar(
        x=displayed["cluster_id"],
        y=displayed["late_rate"],
        name="Zone late rate",
        marker_color="#D58B24",
        customdata=displayed[
            ["order_count", "late_rate_difference_vs_network_percentage_points"]
        ],
        hovertemplate=(
            "Zone %{x}<br>Late rate: %{y:.1%}<br>Orders: %{customdata[0]:,}"
            "<br>Difference vs network: %{customdata[1]:+.1f} pp<extra></extra>"
        ),
    )
    late.add_scatter(
        x=displayed["cluster_id"],
        y=[network["late_rate"]] * len(displayed),
        mode="lines",
        name="Network late rate",
        line={"color": "#245B8F", "dash": "dash"},
    )
    late.update_layout(
        title="Observed late-order rate by zone",
        xaxis_title="Demand zone",
        yaxis_title="Late-order rate",
        yaxis={"tickformat": ".0%"},
    )
    st.plotly_chart(late, width="stretch", config=CHART_CONFIG)
    distance = go.Figure()
    distance.add_bar(
        x=displayed["cluster_id"],
        y=displayed["average_delivery_distance_km"],
        name="Mean valid delivery distance",
        marker_color="#168A8A",
        customdata=displayed[["order_count", "distance_difference_vs_network_km"]],
        hovertemplate=(
            "Zone %{x}<br>Mean valid distance: %{y:.1f} km"
            "<br>Orders: %{customdata[0]:,}"
            "<br>Difference vs network: %{customdata[1]:+.1f} km<extra></extra>"
        ),
    )
    distance.add_scatter(
        x=displayed["cluster_id"],
        y=[network["average_delivery_distance_km"]] * len(displayed),
        mode="lines",
        name="Network mean",
        line={"color": "#245B8F", "dash": "dash"},
    )
    distance.update_layout(
        title="Mean valid delivery distance by zone",
        xaxis_title="Demand zone",
        yaxis_title="Straight-line distance (km)",
    )
    st.plotly_chart(distance, width="stretch", config=CHART_CONFIG)
    st.dataframe(
        displayed[
            [
                "cluster_id",
                "order_count",
                "observed_delivery_outcomes",
                "demand_share",
                "mean_delivery_minutes",
                "median_delivery_minutes",
                "late_rate",
                "average_delivery_distance_km",
                "unique_riders",
                "unique_customer_locations",
                "nearest_restaurant_km",
                "mean_distance_from_centroid_km",
            ]
        ].rename(
            columns={
                "cluster_id": "Zone",
                "order_count": "Orders",
                "observed_delivery_outcomes": "Observed delivery outcomes",
                "demand_share": "Demand share",
                "mean_delivery_minutes": "Mean delivery (min)",
                "median_delivery_minutes": "Median delivery (min)",
                "late_rate": "Late rate",
                "average_delivery_distance_km": "Mean valid distance (km)",
                "unique_riders": "Distinct rider IDs",
                "unique_customer_locations": "Unique coordinate points",
                "nearest_restaurant_km": "Nearest restaurant (km)",
                "mean_distance_from_centroid_km": "Mean centroid distance (km)",
            }
        ).round(3),
        hide_index=True,
        width="stretch",
    )


def _render_diagnostics(
    evaluation: pd.DataFrame,
    clusters: pd.DataFrame,
    report: dict[str, Any],
) -> None:
    st.subheader("K-Means diagnostics")
    kpi_left, kpi_right = st.columns(2)
    kpi_left.metric(
        "Selected cluster count",
        str(report["metadata"]["selected_cluster_count"]),
    )
    kpi_right.metric(
        "Inertia elbow indication",
        str(report["metadata"]["inertia_elbow_suggestion"]),
    )
    st.caption(
        f"Selected silhouette: {report['metadata']['selected_silhouette_score']:.3f}. "
        f"Silhouette uses a reproducible sample of "
        f"{report['metadata']['silhouette_sample']['sample_size']:,} orders. "
        "Review both diagnostics and the map; the elbow indication is not an automatic rule."
    )
    if report["metadata"]["silhouette_selection_at_candidate_boundary"]:
        st.warning(
            "The best silhouette is at the largest evaluated k, so finer partitions "
            "may score higher. The elbow gives a separate, coarser indication; compare "
            "cluster compactness and map interpretability before using this k."
        )
    left, right = st.columns(2)
    with left:
        figure = px.line(
            evaluation,
            x="cluster_count",
            y="inertia_km_squared",
            markers=True,
            labels={
                "cluster_count": "Cluster count (k)",
                "inertia_km_squared": "Weighted within-cluster sum of squares (km²)",
            },
            title="Elbow analysis",
            color_discrete_sequence=["#245B8F"],
        )
        figure.add_vline(
            x=report["metadata"]["inertia_elbow_suggestion"],
            line_dash="dash",
            line_color="#D58B24",
        )
        st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)
    with right:
        figure = px.line(
            evaluation,
            x="cluster_count",
            y="silhouette_score",
            markers=True,
            labels={
                "cluster_count": "Cluster count (k)",
                "silhouette_score": "Sample silhouette score",
            },
            title="Silhouette comparison",
            color_discrete_sequence=["#168A8A"],
        )
        figure.add_vline(
            x=report["metadata"]["selected_cluster_count"],
            line_dash="dash",
            line_color="#D58B24",
        )
        st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)
    size = clusters.sort_values("cluster_id")
    figure = px.bar(
        size,
        x="cluster_id",
        y="order_count",
        color="cluster_id",
        labels={"cluster_id": "Demand zone", "order_count": "Orders"},
        title="Cluster-size distribution",
        color_discrete_sequence=COLORS,
    )
    figure.update_layout(showlegend=False)
    st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)
    st.dataframe(evaluation.round(4), hide_index=True, width="stretch")


def _render_candidate_ranking(ranked: pd.DataFrame) -> None:
    st.subheader("Candidate zones for further investigation")
    if ranked.empty:
        st.info("Candidate-zone ranking is unavailable.")
        return
    columns = [
        "cluster_id",
        "candidate_priority_score",
        "order_count",
        "demand_share",
        "mean_delivery_minutes",
        "late_rate",
        "average_delivery_distance_km",
        "mean_distance_from_centroid_km",
        "nearest_restaurant_km",
        "centroid_latitude",
        "centroid_longitude",
    ]
    visible = ranked.loc[:, columns].rename(
        columns={
            "cluster_id": "Zone",
            "candidate_priority_score": "Relative priority score (0–100)",
            "order_count": "Orders",
            "demand_share": "Demand share",
            "mean_delivery_minutes": "Mean delivery (min)",
            "late_rate": "Observed late rate",
            "average_delivery_distance_km": "Mean valid distance (km)",
            "mean_distance_from_centroid_km": "Mean centroid distance (km)",
            "nearest_restaurant_km": "Nearest restaurant proxy (km)",
            "centroid_latitude": "Centroid latitude",
            "centroid_longitude": "Centroid longitude",
        }
    )
    st.dataframe(visible.round(3), hide_index=True, width="stretch")
    st.caption(
        "Score weights: demand 35%, late rate 25%, average delivery time 20%, "
        "geographic concentration 10%, and distance to nearest supplied restaurant 10%. "
        "Components are relative percentile ranks across these clusters. The restaurant "
        "gap is only a proxy, and the score is not a forecast of savings or profit."
    )
    st.warning(
        "Site selection still requires property availability, rent, inventory, road access, "
        "staffing, capacity, and operational feasibility data. None are present here."
    )
