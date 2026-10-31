"""Live order-placement demo using the saved PS4 and PS5 model pipelines."""

from __future__ import annotations

from datetime import date, time

import pandas as pd
import streamlit as st

from config import PROCESSED_DATA_PATH, SLA_THRESHOLD_MINUTES
from src.models.order_prediction import (
    COORDINATE_INPUTS,
    predict_order_outcomes,
)


@st.cache_data(show_spinner=False)
def _load_input_choices(path: str, modified_ns: int) -> dict[str, list[str]]:
    del modified_ns
    columns = ["City", "Type_of_order"]
    frame = pd.read_csv(path, usecols=columns, dtype="string")
    choices: dict[str, list[str]] = {}
    for column in columns:
        values = (
            frame[column]
            .dropna()
            .str.strip()
            .loc[lambda series: series.ne("")]
            .drop_duplicates()
            .sort_values()
            .tolist()
        )
        if not values:
            raise ValueError(f"No valid values are available for {column}.")
        choices[column] = values
    return choices


def render_order_prediction_demo() -> None:
    """Render a form that returns saved-model ETA and late-order risk."""
    st.header("New Order Prediction Demo")
    st.caption(
        "Enter information available at order placement to estimate delivery time "
        "and late-order risk using the saved PS4 and PS5 models."
    )
    st.info(
        f"The project uses a {SLA_THRESHOLD_MINUTES}-minute provisional SLA. "
        "Predictions are estimates, not guarantees; the SLA has not been confirmed "
        "by the challenge organizer."
    )

    choices = _load_input_choices(
        str(PROCESSED_DATA_PATH),
        PROCESSED_DATA_PATH.stat().st_mtime_ns,
    )
    with st.form("new_order_prediction_form"):
        order_date = st.date_input("Order date", value=date.today())
        order_time = st.time_input("Order time", value=time(11, 30))
        city = st.selectbox("Destination city", choices["City"])
        order_type = st.selectbox("Order type", choices["Type_of_order"])
        include_coordinates = st.checkbox(
            "Enter store and customer coordinates to calculate distance",
            value=False,
        )
        coordinate_text: dict[str, str] = {}
        if include_coordinates:
            first_row = st.columns(2)
            coordinate_text[COORDINATE_INPUTS[0]] = first_row[0].text_input(
                "Store latitude"
            )
            coordinate_text[COORDINATE_INPUTS[1]] = first_row[1].text_input(
                "Store longitude"
            )
            second_row = st.columns(2)
            coordinate_text[COORDINATE_INPUTS[2]] = second_row[0].text_input(
                "Customer latitude"
            )
            coordinate_text[COORDINATE_INPUTS[3]] = second_row[1].text_input(
                "Customer longitude"
            )
        submitted = st.form_submit_button("Estimate delivery")

    if not submitted:
        st.caption(
            "Pickup time, actual delivery time, rider details, and the late label "
            "are not requested because they are unavailable at order placement or "
            "would leak outcome information."
        )
        return

    order = {
        "Order_Date": order_date.strftime("%d-%m-%Y"),
        "Time_Orderd": order_time.strftime("%H:%M:%S"),
        "City": city,
        "Type_of_order": order_type,
    }
    if include_coordinates:
        if any(not value.strip() for value in coordinate_text.values()):
            st.error("Enter all four coordinates, or uncheck the coordinate option.")
            return
        order.update(coordinate_text)

    try:
        result = predict_order_outcomes(order)
    except (FileNotFoundError, ValueError, OSError) as exc:
        st.error(f"Could not produce an estimate: {exc}")
        return

    metric_columns = st.columns(2)
    metric_columns[0].metric(
        "Estimated delivery time",
        f"{result['predicted_delivery_time_minutes']:.1f} min",
    )
    metric_columns[1].metric(
        "Estimated late-order risk",
        f"{result['late_order_risk']:.1%}",
        help=(
            f"The saved model classifies an order as late at risk "
            f"≥ {result['classification_threshold']:.0%}; this operating "
            "threshold is unchanged."
        ),
    )
    st.write(
        f"Saved classifier status at the {result['classification_threshold']:.0%} "
        f"operating threshold: **{result['predicted_status'].replace('_', ' ')}**."
    )
    st.caption(
        f"Models: ETA — {result['regression_model']}; late risk — "
        f"{result['classification_model']}. Late means delivery time strictly "
        f"greater than {result['sla_threshold_minutes']} minutes."
    )
    if result["warnings"]:
        for warning in result["warnings"]:
            st.warning(warning)
    st.caption(
        "Estimates reflect the supplied historical food-delivery dataset and "
        "should not be treated as operational guarantees."
    )
