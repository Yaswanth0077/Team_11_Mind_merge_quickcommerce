"""Read-only Streamlit viewer for the Phase 1 data-quality report."""

import json

import pandas as pd
import streamlit as st

from config import REPORT_DIR

REPORT_PATH = REPORT_DIR / "data_quality_report.json"

st.set_page_config(page_title="Delivery Data Readiness", layout="wide")
st.title("Quick-Commerce Delivery Intelligence")
st.caption("Data readiness — supplied food-delivery dataset used as a quick-commerce proxy")

if not REPORT_PATH.is_file():
    st.info("No audit report exists yet. Run `python -m src.data.pipeline` from the project directory.")
else:
    with REPORT_PATH.open(encoding="utf-8") as report_file:
        report = json.load(report_file)
    overview = report["dataset_overview"]
    records = report["records"]
    delay_lens = report["delay_lens"]
    missing = pd.DataFrame(report["missing_values"])
    missing_cells = int(missing["normalized_missing_values"].sum())
    coordinate_audit = report["coordinate_validation"]
    first, second, third, fourth = st.columns(4)
    first.metric("Rows inspected", f"{overview['rows']:,}")
    second.metric("Missing cells", f"{missing_cells:,}")
    third.metric(
        "(0,0) store coordinates",
        f"{coordinate_audit['zero_store_coordinate_pair_count']:,}",
    )
    fourth.metric(
        "Orders over provisional SLA",
        f"{delay_lens['observed_outcomes']['late_orders']:,}",
    )

    st.warning(
        f"SLA: {delay_lens['sla_threshold_minutes']} minutes — provisional, "
        "pending confirmation. The order count above is an observed label count, not a prediction."
    )
    st.caption(
        f"Rows retained: {records['retained_records']:,} of {records['input_records']:,}; "
        f"rows excluded: {records['excluded_records']:,}. "
        f"Duplicate order IDs: "
        f"{report['duplicate_analysis']['duplicate_order_id_rows_after_first']:,}."
    )

    left, right = st.columns(2)
    with left:
        st.subheader("Missing values after normalization")
        st.dataframe(
            missing.loc[
                missing["normalized_missing_values"] > 0,
                ["column", "normalized_missing_values", "missing_percentage"],
            ],
            hide_index=True,
            width="stretch",
        )
    with right:
        st.subheader("Coordinate review")
        st.write(
            f"{coordinate_audit['expected_region_envelope']['geographically_implausible_pair_count']:,} "
            "coordinate pairs fall outside the broad India screening envelope. "
            "They are retained, but their distances are not used as model features."
        )
        st.write(
            f"{coordinate_audit['distance_km']['over_100_km_count']:,} raw distances "
            f"exceed the {coordinate_audit['distance_km']['review_distance_threshold_km']:g} km review threshold."
        )

    with st.expander("Order-placement feature policy"):
        st.write("**Allowed feature columns**")
        st.code(", ".join(delay_lens["prediction_feature_columns"]))
        st.write("**Excluded fields**")
        st.code(", ".join(delay_lens["excluded_columns"]))
        st.dataframe(
            pd.DataFrame(
                [
                    {"Field": field, "Availability": reason}
                    for field, reason in delay_lens["feature_availability_policy"].items()
                ]
            ),
            hide_index=True,
            width="stretch",
        )
    st.caption(
        "The SLA, peak hours, and India coordinate envelope are working assumptions. "
        "Delay analysis, regression, and late-order risk are separate Streamlit pages; "
        "this page remains the data audit."
    )
