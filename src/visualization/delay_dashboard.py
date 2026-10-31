"""Interactive Streamlit page for descriptive delivery-delay analysis."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

from config import (
    LATE_LABEL_COLUMN,
    PROCESSED_DATA_PATH,
    SLA_THRESHOLD_MINUTES,
    TARGET_COLUMN,
)
from src.analytics.delay_analysis import (
    DelayFilters,
    analysis_quality_summary,
    analyze_by_category,
    analyze_by_day,
    analyze_by_hour,
    analyze_distance_bands,
    analyze_ratings,
    analyze_weather_traffic,
    apply_delay_filters,
    benchmark_comparison_to_network,
    delivery_time_summary,
    generate_insights,
    load_delay_data,
    network_benchmark_metrics,
    rider_performance,
)

CHART_CONFIG = {"displaylogo": False, "responsive": True}
COLORS = {"blue": "#245B8F", "teal": "#168A8A", "amber": "#D58B24"}


@st.cache_data(show_spinner="Loading validated processed orders...")
def _cached_delay_data(path: str, modified_time_ns: int) -> pd.DataFrame:
    del modified_time_ns
    return load_delay_data(Path(path), SLA_THRESHOLD_MINUTES)


@st.cache_data(show_spinner=False)
def _cached_network_benchmark(
    path: str, modified_time_ns: int
) -> dict[str, float | int | None]:
    data = _cached_delay_data(path, modified_time_ns)
    return network_benchmark_metrics(data)


def render_delay_dashboard() -> None:
    """Render the delay analysis filters, metrics, comparisons, and charts."""
    st.header("Delivery Delay Intelligence")
    st.caption(
        "Descriptive patterns in the supplied food-delivery data. "
        "Group differences are associations, not causes."
    )
    try:
        modified_time_ns = PROCESSED_DATA_PATH.stat().st_mtime_ns
        data = _cached_delay_data(str(PROCESSED_DATA_PATH), modified_time_ns)
        network = _cached_network_benchmark(
            str(PROCESSED_DATA_PATH), modified_time_ns
        )
    except FileNotFoundError as error:
        st.error(str(error))
        st.info("Generate the Phase 1 dataset with `python -m src.data.pipeline`.")
        return
    except (ValueError, KeyError) as error:
        st.error(f"Processed data could not be prepared for analysis: {error}")
        return

    filters = _render_filters(data)
    selected = apply_delay_filters(data, filters)
    if selected.empty:
        st.warning("No valid delivery orders match the selected filters.")
        st.caption(
            "Broaden the date, hour, category, or distance selection. "
            "Distance filtering excludes orders with invalid coordinates."
        )
        return

    _render_kpis(selected)
    _render_benchmark(selected, network)
    _render_insights(selected)
    _render_charts(selected)
    _render_rider_summary(selected)
    _render_data_coverage(data, selected)


def _render_filters(data: pd.DataFrame) -> DelayFilters:
    st.subheader("Filters")
    with st.container(border=True):
        date_values = data["analysis_date"].dropna()
        if date_values.empty:
            st.info("No order dates could be parsed; date filtering is unavailable.")
            start_date = end_date = None
        else:
            minimum_date = date_values.min()
            maximum_date = date_values.max()
            date_selection = st.date_input(
                "Order date range",
                value=(minimum_date, maximum_date),
                min_value=minimum_date,
                max_value=maximum_date,
                key="delay_date_range",
            )
            if isinstance(date_selection, (tuple, list)) and len(date_selection) == 2:
                start_date, end_date = date_selection
            else:
                start_date = end_date = date_selection

        hour_range = st.slider(
            "Order hour",
            min_value=0,
            max_value=23,
            value=(0, 23),
            format="%d:00",
            key="delay_hour_range",
        )
        weather, traffic, order_type, city = st.columns(4)
        weather_selection = weather.multiselect(
            "Weather",
            options=sorted(data["analysis_weather"].unique()),
            default=sorted(data["analysis_weather"].unique()),
            key="delay_weather_filter",
        )
        traffic_selection = traffic.multiselect(
            "Traffic",
            options=sorted(data["analysis_traffic"].unique()),
            default=sorted(data["analysis_traffic"].unique()),
            key="delay_traffic_filter",
        )
        order_selection = order_type.multiselect(
            "Order type",
            options=sorted(data["analysis_order_type"].unique()),
            default=sorted(data["analysis_order_type"].unique()),
            key="delay_order_type_filter",
        )
        city_selection = city.multiselect(
            "City",
            options=sorted(data["analysis_city"].unique()),
            default=sorted(data["analysis_city"].unique()),
            key="delay_city_filter",
        )

        distances = pd.to_numeric(data["analysis_distance_km"], errors="coerce").dropna()
        apply_distance = st.checkbox(
            "Restrict results to a delivery-distance range",
            value=False,
            key="delay_apply_distance_filter",
        )
        distance_range: tuple[float, float] | None = None
        if distances.empty:
            st.caption("No valid-coordinate distances are available for this dataset.")
        elif apply_distance:
            minimum_distance = float(distances.min())
            maximum_distance = float(distances.max())
            if minimum_distance == maximum_distance:
                distance_range = (minimum_distance, maximum_distance)
                st.caption(f"All valid distances are {minimum_distance:.1f} km.")
            else:
                distance_range = st.slider(
                    "Distance (km)",
                    min_value=minimum_distance,
                    max_value=maximum_distance,
                    value=(minimum_distance, maximum_distance),
                    step=0.5,
                    key="delay_distance_range",
                )
        else:
            st.caption(
                f"Valid delivery distances range from {distances.min():.1f} to "
                f"{distances.max():.1f} km. Enable the checkbox to filter by distance."
            )

    return DelayFilters(
        start_date=start_date,
        end_date=end_date,
        first_hour=int(hour_range[0]),
        last_hour=int(hour_range[1]),
        weather=tuple(weather_selection),
        traffic=tuple(traffic_selection),
        order_type=tuple(order_selection),
        city=tuple(city_selection),
        distance_km=distance_range,
    )


def _render_kpis(selected: pd.DataFrame) -> None:
    metrics = delivery_time_summary(selected)
    distance = pd.to_numeric(
        selected.loc[selected["analysis_valid_order"], "analysis_distance_km"],
        errors="coerce",
    ).dropna()
    hourly = analyze_by_hour(selected)
    minimum_hour_orders = min(50, max(1, int(metrics["order_count"] * 0.01)))
    eligible_hours = hourly.loc[hourly["order_count"] >= minimum_hour_orders]
    peak_hour = (
        eligible_hours.sort_values(["late_rate", "order_count"], ascending=False).iloc[0]
        if not eligible_hours.empty
        else None
    )

    st.subheader("Selected orders")
    kpis = st.columns(6)
    kpis[0].metric("Valid orders", f"{metrics['order_count']:,}")
    kpis[1].metric(
        "Average delivery",
        f"{metrics['mean_minutes']:.1f} min" if metrics["mean_minutes"] is not None else "—",
    )
    kpis[2].metric(
        "Median delivery",
        f"{metrics['median_minutes']:.1f} min"
        if metrics["median_minutes"] is not None
        else "—",
    )
    kpis[3].metric(
        "Late-order rate",
        f"{metrics['late_rate']:.1%}" if metrics["late_rate"] is not None else "—",
    )
    kpis[4].metric(
        "Average valid distance",
        f"{distance.mean():.1f} km" if not distance.empty else "—",
    )
    kpis[5].metric(
        "Peak-risk hour",
        f"{int(peak_hour['hour']):02d}:00"
        if peak_hour is not None
        else "Insufficient volume",
        help=(
            "Hour with the highest observed late rate among hours meeting the displayed "
            "minimum sample size. This is descriptive."
        ),
    )
    st.caption(
        f"Peak-risk eligibility requires at least {minimum_hour_orders} orders per hour. "
        "Average distance uses valid-coordinate orders only."
    )


def _render_benchmark(
    selected: pd.DataFrame,
    network: dict[str, float | int | None],
) -> None:
    st.subheader("Filtered segment versus full-network benchmark")
    comparison = benchmark_comparison_to_network(selected, network)
    display = comparison.copy()
    for column in ("selected", "overall_network", "difference"):
        display[column] = display[column].map(_format_benchmark_value)
    display["percent_difference"] = comparison["percent_difference"].map(
        lambda value: f"{value:+.1f}%" if pd.notna(value) else "—"
    )
    st.dataframe(
        display.rename(
            columns={
                "metric": "Metric",
                "selected": "Selected",
                "overall_network": "Overall network",
                "difference": "Difference",
                "percent_difference": "Difference vs network",
                "difference_unit": "Difference unit",
            }
        ),
        hide_index=True,
        width="stretch",
    )
    st.caption(
        f"The overall benchmark uses all {int(network['order_count'] or 0):,} orders "
        "with valid delivery times "
        "and does not change with dashboard filters. Distance metrics use valid coordinate pairs."
    )


def _format_benchmark_value(value: object) -> str:
    if pd.isna(value):
        return "—"
    if isinstance(value, (int, np.integer)):
        return f"{value:,}"
    return f"{float(value):.2f}"


def _render_insights(selected: pd.DataFrame) -> None:
    st.subheader("Observed patterns to review")
    insights = generate_insights(selected)
    if not insights:
        st.info(
            "No category met the 100-order minimum sample and exceeded the selected "
            "orders' late-rate benchmark."
        )
        return
    for insight in insights:
        st.markdown(
            f"- **{insight['dimension'].title()} — {insight['segment']}:** "
            f"{insight['finding']} {insight['operational_relevance']}"
        )


def _render_charts(selected: pd.DataFrame) -> None:
    st.subheader("Delivery-time distribution")
    distribution = px.histogram(
        selected,
        x=TARGET_COLUMN,
        nbins=30,
        labels={TARGET_COLUMN: "Delivery time (minutes)", "count": "Orders"},
        title="Observed delivery times",
        color_discrete_sequence=[COLORS["blue"]],
    )
    distribution.update_layout(bargap=0.08, yaxis_title="Orders")
    _chart(distribution)

    st.subheader("When orders are delayed")
    left, right = st.columns(2)
    with left:
        hourly = analyze_by_hour(selected)
        figure = _performance_chart(
            hourly,
            "hour",
            "Delivery performance by order hour",
            "Order hour",
            "Hour",
        )
        _chart(figure)
    with right:
        daily = analyze_by_day(selected)
        figure = _performance_chart(
            daily,
            "day_name",
            "Delivery performance by weekday",
            "Day of week",
            "Weekday",
        )
        _chart(figure)
        weekend = selected["analysis_is_weekend"].fillna(False)
        weekday_comparison = _weekend_summary(selected, weekend)
        st.caption(weekday_comparison)

    st.subheader("Weather and road conditions")
    weather_data = analyze_by_category(selected, "weather")
    left, right = st.columns(2)
    with left:
        weather_chart = px.bar(
            weather_data,
            x="analysis_weather",
            y="late_rate",
            color="late_rate",
            color_continuous_scale="Blues",
            hover_data={
                "order_count": ":,",
                "mean_delivery_minutes": ":.1f",
                "late_rate": ":.1%",
            },
            labels={
                "analysis_weather": "Weather",
                "late_rate": "Late-order rate",
                "mean_delivery_minutes": "Average delivery (min)",
                "order_count": "Orders",
            },
            title="Late-order rate by weather",
        )
        weather_chart.update_yaxes(tickformat=".0%")
        _chart(weather_chart)
    with right:
        heatmap_data = analyze_weather_traffic(selected)
        _render_heatmap(heatmap_data)

    st.subheader("Distance and delivery performance")
    valid_distances = selected.loc[selected["analysis_distance_km"].notna()]
    left, right = st.columns(2)
    with left:
        if valid_distances.empty:
            st.info("No valid-coordinate distances are available in this selection.")
        else:
            scatter = go.Figure(
                go.Scattergl(
                    x=valid_distances["analysis_distance_km"],
                    y=valid_distances[TARGET_COLUMN],
                    mode="markers",
                    marker={"size": 5, "opacity": 0.45, "color": COLORS["blue"]},
                    customdata=valid_distances[
                        ["analysis_weather", "analysis_traffic", LATE_LABEL_COLUMN]
                    ],
                    hovertemplate=(
                        "Distance: %{x:.2f} km<br>Delivery: %{y} min"
                        "<br>Weather: %{customdata[0]}<br>Traffic: %{customdata[1]}"
                        "<br>Late: %{customdata[2]}<extra></extra>"
                    ),
                )
            )
            scatter.update_layout(
                title="Delivery time versus straight-line distance",
                xaxis_title="Haversine distance (km)",
                yaxis_title="Delivery time (minutes)",
            )
            _chart(scatter)
            correlation = valid_distances[
                ["analysis_distance_km", TARGET_COLUMN]
            ].corr(method="spearman").iloc[0, 1]
            st.caption(
                f"Spearman rank correlation: {correlation:.3f} "
                f"({len(valid_distances):,} valid-coordinate orders). "
                "This is an association, not evidence that distance alone explains delivery time."
            )
    with right:
        band_data = analyze_distance_bands(selected)
        _chart(
            _performance_chart(
                band_data,
                "analysis_distance_band",
                "Delivery performance by distance band",
                "Distance band",
                "Distance",
            )
        )

    st.subheader("Order and rider characteristics")
    left, right = st.columns(2)
    with left:
        order_data = analyze_by_category(selected, "order_type")
        _chart(
            _performance_chart(
                order_data,
                "analysis_order_type",
                "Delivery performance by order type",
                "Order type",
                "Order type",
            )
        )
    with right:
        rating_data = analyze_ratings(selected)
        if rating_data.empty:
            st.info("No rider ratings within the valid 1–5 scale are available.")
        else:
            rating_chart = px.scatter(
                rating_data,
                x="rating",
                y="mean_delivery_minutes",
                size="order_count",
                color="late_rate",
                color_continuous_scale="Blues",
                hover_data={
                    "order_count": ":,",
                    "late_rate": ":.1%",
                    "mean_delivery_minutes": ":.1f",
                },
                labels={
                    "rating": "Rider rating (1–5)",
                    "mean_delivery_minutes": "Average delivery (min)",
                    "order_count": "Orders",
                    "late_rate": "Late-order rate",
                },
                title="Observed delivery performance by rider rating",
            )
            _chart(rating_chart)
            st.caption(
                "Ratings above 5 or missing ratings are excluded from this comparison. "
                "The association does not establish that rating causes faster delivery."
            )


def _performance_chart(
    grouped: pd.DataFrame,
    category: str,
    title: str,
    category_label: str,
    category_hover: str,
) -> go.Figure:
    figure = go.Figure()
    if grouped.empty:
        figure.update_layout(title=title, annotations=[_empty_annotation()])
        return figure
    figure.add_bar(
        x=grouped[category],
        y=grouped["mean_delivery_minutes"],
        name="Average delivery (min)",
        marker_color=COLORS["blue"],
        customdata=grouped[["order_count", "late_rate", "median_delivery_minutes"]],
        hovertemplate=(
            f"{category_hover}: %{{x}}<br>Average: %{{y:.1f}} min"
            "<br>Orders: %{customdata[0]:,}<br>Late rate: %{customdata[1]:.1%}"
            "<br>Median: %{customdata[2]:.1f} min<extra></extra>"
        ),
    )
    figure.add_scatter(
        x=grouped[category],
        y=grouped["late_rate"],
        name="Late-order rate",
        mode="lines+markers",
        yaxis="y2",
        line={"color": COLORS["teal"], "width": 2},
        hovertemplate=f"{category_hover}: %{{x}}<br>Late rate: %{{y:.1%}}<extra></extra>",
    )
    figure.update_layout(
        title=title,
        xaxis_title=category_label,
        yaxis={"title": "Average delivery (minutes)", "rangemode": "tozero"},
        yaxis2={
            "title": "Late-order rate",
            "overlaying": "y",
            "side": "right",
            "tickformat": ".0%",
            "rangemode": "tozero",
        },
        legend={"orientation": "h", "y": -0.25},
        margin={"b": 70},
    )
    return figure


def _render_heatmap(data: pd.DataFrame) -> None:
    if data.empty:
        st.info("No orders have both weather and traffic values in this selection.")
        return
    pivot = data.pivot(
        index="analysis_weather", columns="analysis_traffic", values="late_rate"
    )
    counts = data.pivot(
        index="analysis_weather", columns="analysis_traffic", values="order_count"
    )
    traffic_order = ("Low", "Medium", "High", "Jam", "(Missing)")
    weather_order = sorted(
        value for value in pivot.index if value != "(Missing)"
    ) + (["(Missing)"] if "(Missing)" in pivot.index else [])
    pivot = pivot.reindex(
        index=weather_order,
        columns=[value for value in traffic_order if value in pivot.columns]
        + sorted(value for value in pivot.columns if value not in traffic_order),
    )
    counts = counts.reindex(index=pivot.index, columns=pivot.columns)
    figure = go.Figure(
        go.Heatmap(
            z=pivot.to_numpy() * 100,
            x=pivot.columns,
            y=pivot.index,
            customdata=counts.to_numpy(),
            colorscale="Blues",
            colorbar={"title": "Late rate (%)"},
            hovertemplate=(
                "Weather: %{y}<br>Traffic: %{x}<br>Late rate: %{z:.1f}%"
                "<br>Orders: %{customdata:,}<extra></extra>"
            ),
            text=[
                [
                    f"{pivot.iloc[row, column]:.1%}<br>n={counts.iloc[row, column]:,}"
                    if pd.notna(pivot.iloc[row, column])
                    else ""
                    for column in range(pivot.shape[1])
                ]
                for row in range(pivot.shape[0])
            ],
            texttemplate="%{text}",
        )
    )
    figure.update_layout(
        title="Late-order rate by weather and traffic",
        xaxis_title="Road traffic density",
        yaxis_title="Weather condition",
    )
    _chart(figure)


def _weekend_summary(selected: pd.DataFrame, weekend: pd.Series) -> str:
    valid = selected.loc[selected["analysis_valid_order"]].copy()
    valid["weekend_group"] = weekend.loc[valid.index].map(
        {True: "Weekend", False: "Weekday"}
    )
    groups = valid.groupby("weekend_group", observed=True).agg(
        orders=(TARGET_COLUMN, "count"),
        average_minutes=(TARGET_COLUMN, "mean"),
        late_rate=(LATE_LABEL_COLUMN, "mean"),
    )
    if not {"Weekday", "Weekend"}.issubset(groups.index):
        return "Weekend/weekday comparison requires orders in both groups."
    weekday = groups.loc["Weekday"]
    weekend_row = groups.loc["Weekend"]
    return (
        f"Weekend: {int(weekend_row['orders']):,} orders, "
        f"{weekend_row['average_minutes']:.1f} min average, "
        f"{weekend_row['late_rate']:.1%} late. Weekday: "
        f"{int(weekday['orders']):,} orders, {weekday['average_minutes']:.1f} min average, "
        f"{weekday['late_rate']:.1%} late."
    )


def _render_rider_summary(selected: pd.DataFrame) -> None:
    st.subheader("Rider-level descriptive comparison")
    st.caption(
        "Only riders with at least 30 valid orders are shown. Rider tenure is not in this dataset; "
        "these summaries do not establish experience or causation."
    )
    riders = rider_performance(selected)
    qualified = riders.loc[riders["meets_minimum_sample"]]
    if qualified.empty:
        st.info("No rider meets the 30-order minimum within the current filter selection.")
        return
    display = qualified.head(15).copy()
    display["late_rate"] = display["late_rate"].map(lambda value: f"{value:.1%}")
    display["mean_delivery_minutes"] = display["mean_delivery_minutes"].map(
        lambda value: f"{value:.1f}"
    )
    display["average_rating"] = display["average_rating"].map(
        lambda value: f"{value:.2f}" if pd.notna(value) else "—"
    )
    st.dataframe(
        display[
            [
                "Delivery_person_ID",
                "order_count",
                "mean_delivery_minutes",
                "late_rate",
                "average_rating",
            ]
        ].rename(
            columns={
                "Delivery_person_ID": "Rider ID",
                "order_count": "Orders",
                "mean_delivery_minutes": "Average delivery (min)",
                "late_rate": "Late-order rate",
                "average_rating": "Average rating",
            }
        ),
        hide_index=True,
        width="stretch",
    )


def _render_data_coverage(overall: pd.DataFrame, selected: pd.DataFrame) -> None:
    quality = analysis_quality_summary(overall)
    st.subheader("Analysis coverage")
    coverage = st.columns(4)
    coverage[0].metric(
        "Missing/invalid delivery times",
        f"{quality['missing_or_invalid_delivery_time_records']:,}",
    )
    coverage[1].metric(
        "Invalid coordinate pairs",
        f"{quality['invalid_coordinate_records']:,}",
    )
    coverage[2].metric(
        "Missing coordinate pairs",
        f"{quality['missing_coordinate_records']:,}",
    )
    coverage[3].metric(
        "Missing/invalid ratings",
        f"{quality['invalid_or_missing_rating_records']:,}",
    )
    with st.expander("Other comparison coverage"):
        coverage_details = pd.DataFrame(
            [
                ("Missing order dates", quality["missing_order_date_records"]),
                ("Missing order times", quality["missing_order_time_records"]),
                ("Unavailable distances", quality["unavailable_distance_records"]),
                ("Missing weather", quality["missing_weather_records"]),
                ("Missing traffic", quality["missing_traffic_records"]),
                ("Missing order type", quality["missing_order_type_records"]),
                ("Missing city", quality["missing_city_records"]),
                ("Missing ratings", quality["missing_rating_records"]),
                ("Invalid ratings outside 1–5", quality["invalid_rating_records"]),
            ],
            columns=["Field", "Affected valid orders"],
        )
        st.dataframe(coverage_details, hide_index=True, width="stretch")
    st.caption(
        f"Current selection contains {len(selected):,} valid-target orders. "
        "Preparation does not delete source rows. "
        "Distance and rating analyses use their respective valid observations; "
        "other comparisons retain orders with valid delivery times."
    )


def _chart(figure: go.Figure) -> None:
    st.plotly_chart(figure, width="stretch", config=CHART_CONFIG)


def _empty_annotation() -> dict[str, str | int]:
    return {
        "text": "No matching observations",
        "xref": "paper",
        "yref": "paper",
        "x": 0.5,
        "y": 0.5,
        "showarrow": False,
    }
