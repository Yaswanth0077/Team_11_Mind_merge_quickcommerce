"""Tests for placement-time input validation and saved-model demo inference."""

from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from config import PROCESSED_DATA_PATH, SLA_THRESHOLD_MINUTES
from config import PROJECT_ROOT
from src.models.order_prediction import (
    COORDINATE_INPUTS,
    predict_order_outcomes,
    validate_order_placement_inputs,
)
from streamlit.testing.v1 import AppTest

BASE_ORDER = {
    "Order_Date": "19-03-2022",
    "Time_Orderd": "11:30:00",
    "City": "Urban",
    "Type_of_order": "Snack",
}
VALID_COORDINATES = {
    "Restaurant_latitude": 22.745049,
    "Restaurant_longitude": 75.892471,
    "Delivery_location_latitude": 22.765049,
    "Delivery_location_longitude": 75.912471,
}


class OrderPredictionTests(unittest.TestCase):
    def test_order_placement_inputs_are_normalized(self) -> None:
        normalized = validate_order_placement_inputs(
            {**BASE_ORDER, **{key: str(value) for key, value in VALID_COORDINATES.items()}}
        )
        self.assertEqual(normalized["Order_Date"], "19-03-2022")
        for field in COORDINATE_INPUTS:
            self.assertIsInstance(normalized[field], float)

    def test_missing_coordinates_are_allowed_as_a_complete_absence(self) -> None:
        normalized = validate_order_placement_inputs(BASE_ORDER)
        self.assertTrue(set(COORDINATE_INPUTS).isdisjoint(normalized))

    def test_partial_or_invalid_coordinates_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "all four store/customer"):
            validate_order_placement_inputs(
                {**BASE_ORDER, "Restaurant_latitude": 22.7}
            )
        with self.assertRaisesRegex(ValueError, "screening bounds"):
            validate_order_placement_inputs(
                {**BASE_ORDER, **VALID_COORDINATES, "Restaurant_latitude": 0}
            )
        with self.assertRaisesRegex(ValueError, "valid number"):
            validate_order_placement_inputs(
                {**BASE_ORDER, **VALID_COORDINATES, "Restaurant_latitude": "north"}
            )

    def test_missing_and_malformed_required_inputs_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "City is required"):
            validate_order_placement_inputs({**BASE_ORDER, "City": ""})
        with self.assertRaisesRegex(ValueError, "DD-MM-YYYY"):
            validate_order_placement_inputs({**BASE_ORDER, "Order_Date": "2022-03-19"})
        with self.assertRaisesRegex(ValueError, "HH:MM:SS"):
            validate_order_placement_inputs({**BASE_ORDER, "Time_Orderd": "11:30"})

    def test_outcome_or_unavailable_fields_cannot_enter_prediction(self) -> None:
        for key in ("Time_taken(min)", "Time_Order_picked", "Delivery_person_ID"):
            with self.subTest(field=key), self.assertRaisesRegex(
                ValueError, "Unsupported or post-placement"
            ):
                validate_order_placement_inputs({**BASE_ORDER, key: "30"})

    def test_saved_models_produce_finite_estimates_and_keep_threshold(self) -> None:
        result = predict_order_outcomes({**BASE_ORDER, **VALID_COORDINATES})
        self.assertTrue(np.isfinite(result["predicted_delivery_time_minutes"]))
        self.assertGreaterEqual(result["predicted_delivery_time_minutes"], 0)
        self.assertTrue(np.isfinite(result["late_order_risk"]))
        self.assertGreaterEqual(result["late_order_risk"], 0)
        self.assertLessEqual(result["late_order_risk"], 1)
        self.assertEqual(result["classification_threshold"], 0.15)
        self.assertEqual(result["sla_threshold_minutes"], SLA_THRESHOLD_MINUTES)

    def test_saved_models_infer_from_a_real_processed_order(self) -> None:
        frame = pd.read_csv(
            PROCESSED_DATA_PATH,
            usecols=[
                "Order_Date",
                "Time_Orderd",
                *COORDINATE_INPUTS,
                "City",
                "Type_of_order",
            ],
            dtype="string",
        )
        candidate = None
        for _, row in frame.iterrows():
            order = row.dropna().to_dict()
            if not set(BASE_ORDER).issubset(order):
                continue
            try:
                candidate = validate_order_placement_inputs(order)
                if set(COORDINATE_INPUTS).issubset(candidate):
                    break
            except ValueError:
                continue
        self.assertIsNotNone(candidate, "No complete valid processed order was found.")
        result = predict_order_outcomes(candidate)
        self.assertIn(result["predicted_status"], {"late", "on_time"})

    def test_demo_renders_and_recomputes_after_a_form_selection_changes(self) -> None:
        page = PROJECT_ROOT / "pages" / "Order_Prediction_Demo.py"
        app = AppTest.from_file(str(page), default_timeout=60).run()
        self.assertFalse(app.exception, [error.message for error in app.exception])

        app.button[0].click().run()
        self.assertFalse(app.exception, [error.message for error in app.exception])
        self.assertEqual(
            {metric.label for metric in app.metric},
            {"Estimated delivery time", "Estimated late-order risk"},
        )
        initial_estimates = {
            metric.label: metric.value for metric in app.metric
        }

        cities = app.selectbox[0].options
        alternative = next((value for value in cities if value != app.selectbox[0].value), None)
        if alternative is not None:
            app.selectbox[0].select(alternative)
            app.button[0].click().run()
            self.assertFalse(app.exception, [error.message for error in app.exception])
            self.assertEqual(len(app.metric), 2)
            changed_estimates = {
                metric.label: metric.value for metric in app.metric
            }
            self.assertNotEqual(initial_estimates, changed_estimates)


if __name__ == "__main__":
    unittest.main()
