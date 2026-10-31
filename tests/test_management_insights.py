"""Tests for the PS8 evidence-integrated management report."""

from __future__ import annotations

import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from config import PROJECT_ROOT
from src.analytics.management_insights import (
    build_management_report,
    render_management_markdown,
)


class ManagementInsightsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.report = build_management_report()

    def test_saved_model_metrics_and_threshold_are_verified(self) -> None:
        regression = self.report["regression"]
        self.assertEqual(regression["selected_model"], "Gradient Boosting")
        self.assertAlmostEqual(
            regression["selected_test_metrics"]["mae_minutes"],
            6.4055333166,
            places=8,
        )
        self.assertEqual(len(regression["model_comparison"]), 5)

        classification = self.report["classification"]
        self.assertEqual(classification["selected_model"], "Gradient Boosting")
        self.assertEqual(classification["selected_threshold"], 0.15)
        test = classification["selected_test_metrics"]
        self.assertEqual(
            (
                test["true_negatives"],
                test["false_positives"],
                test["false_negatives"],
                test["true_positives"],
            ),
            (2431, 3967, 146, 2575),
        )
        self.assertAlmostEqual(test["recall"], 0.9463432562, places=8)

    def test_report_contains_exactly_three_complete_recommendations(self) -> None:
        recommendations = self.report["recommendations"]
        self.assertEqual(len(recommendations), 3)
        required_fields = {
            "priority",
            "evidence",
            "problem",
            "action",
            "expected_benefit",
            "implementation",
            "risks",
            "monitoring_kpi",
            "validation",
        }
        for recommendation in recommendations:
            self.assertTrue(required_fields.issubset(recommendation))
            self.assertTrue(recommendation["evidence"])
            self.assertTrue(recommendation["risks"])
        self.assertEqual(
            [item["priority"] for item in recommendations],
            ["High", "High", "Medium"],
        )
        markdown = render_management_markdown(self.report)
        self.assertEqual(markdown.count("### 1. "), 1)
        self.assertEqual(markdown.count("### 2. "), 1)
        self.assertEqual(markdown.count("### 3. "), 1)
        self.assertNotIn("### 4. ", markdown)
        self.assertIn("maximized F2 on the training-only validation split", markdown)
        self.assertIn("6,542/9,119 orders (71.7%)", markdown)
        self.assertIn("not a cost-optimal threshold", markdown)
        self.assertIn("intervention costs and alert capacity are not measured", markdown)

    def test_prior_statement_statuses_and_limits_remain_visible(self) -> None:
        statuses = {
            item["id"]: item["status"]
            for item in self.report["problem_statement_status"]
        }
        self.assertIn("PS2", statuses)
        self.assertIn("PS8", statuses)
        self.assertIn("provisional", self.report["dataset"]["sla_status"].lower())
        self.assertIn(
            "boundary",
            self.report["geographic_analysis"]["selection_warning"].lower(),
        )

    def test_explainability_uses_saved_permutation_importance(self) -> None:
        regression_features = {
            row["feature"]
            for row in self.report["explainability"]["regression_importance"]
        }
        classification_features = {
            row["feature"]
            for row in self.report["explainability"]["classification_importance"]
        }
        self.assertIn("is_peak_hour", regression_features)
        self.assertIn("delivery_distance_km", regression_features)
        self.assertIn("is_peak_hour", classification_features)
        self.assertNotIn("Time_Order_picked", regression_features)
        self.assertNotIn("Time_Order_picked", classification_features)

    def test_management_dashboard_selectors_update_metrics(self) -> None:
        page = PROJECT_ROOT / "pages" / "Management_Insights.py"
        app = AppTest.from_file(str(page), default_timeout=60).run()
        self.assertFalse(app.exception, [error.message for error in app.exception])

        app.selectbox(key="management_regression_model").select("Linear Regression").run()
        self.assertTrue(
            any(metric.label == "Test MAE" and "6.63" in str(metric.value) for metric in app.metric)
        )

        app.selectbox(key="management_classification_model").select(
            "Logistic Regression"
        ).run()
        self.assertTrue(
            any(
                metric.label == "Validation recall" and "90.6%" in str(metric.value)
                for metric in app.metric
            )
        )

        app.selectbox(key="management_rider_segment").select(1).run()
        self.assertTrue(
            any(metric.label == "Riders" and str(metric.value) == "213" for metric in app.metric)
        )

    def test_classifier_dashboard_explains_locked_threshold_and_tradeoff(self) -> None:
        page = PROJECT_ROOT / "pages" / "Late_Order_Risk_Intelligence.py"
        app = AppTest.from_file(str(page), default_timeout=60).run()
        self.assertFalse(app.exception, [error.message for error in app.exception])
        messages = [element.value for element in app.info]
        self.assertTrue(any("maximized F2" in message for message in messages))
        self.assertTrue(any("6,542/9,119" in message for message in messages))
        self.assertTrue(any("not a cost-optimal threshold" in message for message in messages))


if __name__ == "__main__":
    unittest.main()
