"""Build the two submission DOCX files from verified project artifacts."""

from __future__ import annotations

import json
from io import BytesIO
from pathlib import Path
from typing import Any, Iterable

from docx import Document
from docx.enum.table import WD_CELL_VERTICAL_ALIGNMENT, WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Inches, Pt, RGBColor
from PIL import Image, ImageDraw, ImageFont

from src.analytics.rider_support import SEGMENT_SUPPORT_ACTIONS

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = Path(__file__).resolve().parent
INK = "18324B"
BLUE = "245B8F"
TEAL = "168A8A"
PALE_BLUE = "EAF1F7"
PALE_TEAL = "E6F3F2"
MID_GRAY = "627384"
LIGHT_GRAY = "F3F6F8"
WHITE = "FFFFFF"


def _read_json(relative_path: str) -> dict[str, Any]:
    path = ROOT / relative_path
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def _set_cell_shading(cell: Any, fill: str) -> None:
    properties = cell._tc.get_or_add_tcPr()
    shading = OxmlElement("w:shd")
    shading.set(qn("w:fill"), fill)
    properties.append(shading)


def _set_cell_margins(cell: Any, top: int = 70, start: int = 90, bottom: int = 70, end: int = 90) -> None:
    properties = cell._tc.get_or_add_tcPr()
    margins = OxmlElement("w:tcMar")
    for name, value in (("top", top), ("start", start), ("bottom", bottom), ("end", end)):
        item = OxmlElement(f"w:{name}")
        item.set(qn("w:w"), str(value))
        item.set(qn("w:type"), "dxa")
        margins.append(item)
    properties.append(margins)


def _repeat_table_header(row: Any) -> None:
    properties = row._tr.get_or_add_trPr()
    repeat = OxmlElement("w:tblHeader")
    repeat.set(qn("w:val"), "true")
    properties.append(repeat)


def _add_page_field(paragraph: Any) -> None:
    run = paragraph.add_run()
    begin = OxmlElement("w:fldChar")
    begin.set(qn("w:fldCharType"), "begin")
    instruction = OxmlElement("w:instrText")
    instruction.set(qn("xml:space"), "preserve")
    instruction.text = " PAGE "
    separate = OxmlElement("w:fldChar")
    separate.set(qn("w:fldCharType"), "separate")
    text = OxmlElement("w:t")
    text.text = "1"
    end = OxmlElement("w:fldChar")
    end.set(qn("w:fldCharType"), "end")
    for element in (begin, instruction, separate, text, end):
        run._r.append(element)


def _make_document(title: str, subtitle: str) -> Document:
    document = Document()
    section = document.sections[0]
    section.top_margin = Inches(0.58)
    section.bottom_margin = Inches(0.58)
    section.left_margin = Inches(0.68)
    section.right_margin = Inches(0.68)
    styles = document.styles
    normal = styles["Normal"]
    normal.font.name = "Aptos"
    normal.font.size = Pt(9.5)
    normal.font.color.rgb = RGBColor.from_string(INK)
    normal.paragraph_format.space_after = Pt(5)
    normal.paragraph_format.line_spacing = 1.05
    for name, size, color in (
        ("Title", 25, INK),
        ("Heading 1", 17, INK),
        ("Heading 2", 12, BLUE),
        ("Heading 3", 10, TEAL),
    ):
        style = styles[name]
        style.font.name = "Aptos Display" if name != "Heading 3" else "Aptos"
        style.font.size = Pt(size)
        style.font.bold = True
        style.font.color.rgb = RGBColor.from_string(color)
        style.paragraph_format.space_before = Pt(5)
        style.paragraph_format.space_after = Pt(4)
        style.paragraph_format.keep_with_next = True

    header = section.header.paragraphs[0]
    header.text = "QUICK-COMMERCE DELIVERY INTELLIGENCE  |  SUBMISSION"
    header.style = styles["Caption"]
    header.runs[0].font.color.rgb = RGBColor.from_string(MID_GRAY)
    header.runs[0].font.size = Pt(8)
    footer = section.footer.paragraphs[0]
    footer.alignment = WD_ALIGN_PARAGRAPH.RIGHT
    footer.add_run("Evidence-based analysis  •  Page ")
    _add_page_field(footer)

    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_before = Pt(5)
    paragraph.paragraph_format.space_after = Pt(2)
    run = paragraph.add_run(title)
    run.bold = True
    run.font.name = "Aptos Display"
    run.font.size = Pt(24)
    run.font.color.rgb = RGBColor.from_string(INK)
    paragraph = document.add_paragraph()
    paragraph.paragraph_format.space_after = Pt(10)
    run = paragraph.add_run(subtitle)
    run.font.size = Pt(10)
    run.font.color.rgb = RGBColor.from_string(MID_GRAY)
    return document


def _add_paragraph(document: Document, text: str, *, bold_lead: str | None = None) -> None:
    paragraph = document.add_paragraph()
    if bold_lead:
        lead = paragraph.add_run(bold_lead + " ")
        lead.bold = True
        paragraph.add_run(text)
    else:
        paragraph.add_run(text)


def _add_bullets(document: Document, items: Iterable[str]) -> None:
    for item in items:
        paragraph = document.add_paragraph(style="List Bullet")
        paragraph.paragraph_format.space_after = Pt(2)
        paragraph.add_run(item)


def _add_table(
    document: Document,
    headers: list[str],
    rows: Iterable[Iterable[Any]],
    *,
    font_size: float = 8,
    widths: list[float] | None = None,
) -> None:
    rows = [list(row) for row in rows]
    table = document.add_table(rows=1, cols=len(headers))
    table.alignment = WD_TABLE_ALIGNMENT.CENTER
    table.autofit = widths is None
    table.style = "Table Grid"
    _repeat_table_header(table.rows[0])
    for index, header in enumerate(headers):
        cell = table.rows[0].cells[index]
        cell.text = str(header)
        _set_cell_shading(cell, INK)
    for row_index, values in enumerate(rows):
        cells = table.add_row().cells
        for column, value in enumerate(values):
            cells[column].text = str(value)
            if row_index % 2 == 1:
                _set_cell_shading(cells[column], LIGHT_GRAY)
    for row in table.rows:
        for index, cell in enumerate(row.cells):
            cell.vertical_alignment = WD_CELL_VERTICAL_ALIGNMENT.CENTER
            _set_cell_margins(cell)
            if widths:
                cell.width = Inches(widths[index])
            for paragraph in cell.paragraphs:
                paragraph.paragraph_format.space_after = Pt(0)
                paragraph.paragraph_format.line_spacing = 1.0
                for run in paragraph.runs:
                    run.font.name = "Aptos"
                    run.font.size = Pt(font_size)
                    if row is table.rows[0]:
                        run.font.bold = True
                        run.font.color.rgb = RGBColor.from_string(WHITE)
    document.add_paragraph().paragraph_format.space_after = Pt(1)


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    path = Path(r"C:\Windows\Fonts\arialbd.ttf" if bold else r"C:\Windows\Fonts\arial.ttf")
    if path.is_file():
        return ImageFont.truetype(str(path), size=size)
    return ImageFont.load_default()


def _bar_chart(title: str, values: list[tuple[str, float]], suffix: str) -> BytesIO:
    width, height = 1200, max(280, 100 + len(values) * 55)
    image = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(image)
    draw.text((28, 20), title, fill=f"#{INK}", font=_font(28, bold=True))
    label_font, value_font = _font(20), _font(18, bold=True)
    start_y, row_height = 78, 47
    max_value = max(value for _, value in values) or 1.0
    for index, (label, value) in enumerate(values):
        y = start_y + index * row_height
        draw.text((28, y + 7), label, fill=f"#{INK}", font=label_font)
        left, right = 370, 1020
        draw.rounded_rectangle((left, y, right, y + 32), radius=7, fill=f"#{LIGHT_GRAY}")
        fill_right = left + int((right - left) * value / max_value)
        color = f"#{TEAL if index % 2 else BLUE}"
        draw.rounded_rectangle((left, y, fill_right, y + 32), radius=7, fill=color)
        draw.text((1040, y + 5), f"{value:.1f}{suffix}", fill=f"#{INK}", font=value_font)
    image_buffer = BytesIO()
    image.save(image_buffer, format="PNG", optimize=True)
    image_buffer.seek(0)
    return image_buffer


def _add_chart(document: Document, image: BytesIO) -> None:
    paragraph = document.add_paragraph()
    paragraph.alignment = WD_ALIGN_PARAGRAPH.CENTER
    paragraph.add_run().add_picture(image, width=Inches(6.75))
    paragraph.paragraph_format.space_after = Pt(2)


def _add_callout(document: Document, title: str, text: str, fill: str = PALE_BLUE) -> None:
    table = document.add_table(rows=1, cols=1)
    cell = table.cell(0, 0)
    _set_cell_shading(cell, fill)
    _set_cell_margins(cell, top=120, start=150, bottom=120, end=150)
    paragraph = cell.paragraphs[0]
    lead = paragraph.add_run(title + "  ")
    lead.bold = True
    lead.font.color.rgb = RGBColor.from_string(BLUE)
    paragraph.add_run(text)
    document.add_paragraph().paragraph_format.space_after = Pt(0)


def _page_break(document: Document) -> None:
    document.add_page_break()


def _quality_values() -> dict[str, Any]:
    data = _read_json("reports/management_insights.json")
    quality = _read_json("reports/data_quality_report.json")
    return data, quality


def build_business_report() -> Path:
    report, quality = _quality_values()
    reg = report["regression"]
    clf = report["classification"]
    test = clf["selected_test_metrics"]
    ps3 = report["ps3_findings"]
    geo = report["geographic_analysis"]
    rider = report["rider_analysis"]
    solution = _read_json("reports/management_insights.json")
    document = _make_document(
        "Business Report",
        "Quick-Commerce Delivery Intelligence  |  Evidence brief for operations leadership",
    )

    document.add_heading("Executive summary", level=1)
    _add_paragraph(
        document,
        "This analysis turns 45,593 historical food-delivery records into a cautious "
        "delivery-time estimate, an order-placement late-risk signal, exploratory demand "
        "zones and descriptive rider groups. The dataset is a proxy, not verified "
        "quick-commerce or dark-store operations data. The evidence supports testing and "
        "investigation—not claims of causality, guaranteed improvement, or financial return.",
    )
    _add_table(
        document,
        ["Evidence snapshot", "Measured result"],
        [
            ("Delivery time", f"Mean {report['dataset']['mean_delivery_minutes']:.2f} min; median {report['dataset']['median_delivery_minutes']:.0f} min"),
            ("Late-order share", f"{report['dataset']['late_rate']:.1%} at the provisional 30-minute SLA"),
            ("PS4 selected model", f"{reg['selected_model']}; test MAE {reg['selected_test_metrics']['mae_minutes']:.2f} min"),
            ("PS5 fixed operating point", f"{clf['selected_model']} at 0.15; test recall {test['recall']:.1%}, precision {test['precision']:.1%}"),
            ("PS6 / PS7", f"Geographic K={geo['selected_cluster_count']} (boundary candidate); rider K={rider['selected_cluster_count']}"),
        ],
        font_size=8.5,
    )
    document.add_heading("Business problem and objectives", level=2)
    _add_paragraph(
        document,
        "The operating question is how to recognize delivery delay patterns early enough "
        "to investigate them while distinguishing measurable evidence from assumptions. "
        "The analysis objectives are to quantify delay differences, estimate ETA and "
        "late-order risk using placement-time fields, surface demand areas for further "
        "site research, and describe rider cohorts without turning clusters into "
        "individual judgments.",
    )
    _add_callout(
        document,
        "Decision boundary",
        "A model score is not a service promise; geographic concentration is not proof "
        "of a profitable store; and rider segments are not performance rankings.",
    )

    _page_break(document)
    document.add_heading("Dataset, quality and PS2 delay contract", level=1)
    _add_paragraph(
        document,
        f"The supplied dataset contains {report['dataset']['rows']:,} orders and "
        f"{report['dataset']['columns']} fields. Preparation retained all "
        f"{report['dataset']['retained_rows']:,} rows; duplicate order IDs after the "
        "first were not observed. Delivery time is `Time_taken(min)` in minutes. "
        "An order is late only when actual delivery time is strictly greater than "
        "30 minutes; exactly 30 minutes is on time. The 30-minute SLA is provisional "
        "because the guide provides no approved threshold.",
    )
    coord = report["dataset"]["valid_distance_observations"]
    _add_table(
        document,
        ["Data-quality item", "Observed evidence and handling"],
        [
            ("GPS / distance", f"{coord:,} model-eligible coordinate distances; {report['dataset']['invalid_or_unavailable_distance_observations']:,} unavailable/invalid distances remain missing to the model."),
            ("Rider rating", f"{quality['numeric_validation']['rider_rating']['out_of_range_count']} values outside the valid 1–5 range; missing ratings are retained as missing."),
            ("Rider age", f"{quality['numeric_validation']['rider_age']['suspicious_count']} values flagged for review; missing age values are not imputed as experience."),
            ("Missing fields", "Age, ratings, order time, traffic, weather, festival, multiple deliveries and city have missing tokens; these are normalized, not blanket-imputed in the source data."),
            ("Outliers", "Suspicious values are flagged for audit; valid delivery targets are retained rather than removed solely for being extreme."),
            ("Coordinate screen", "Zero pairs and points outside the broad India envelope are excluded from model-eligible distance; the envelope is a screening assumption, not ground truth."),
        ],
        font_size=8,
    )
    document.add_heading("Distance and prediction-time leakage", level=2)
    _add_paragraph(
        document,
        "Store-to-customer straight-line distance is calculated by the Haversine "
        "method using Earth radius 6,371 km. It is not road distance. Predictors use "
        "validated distance when coordinates are available, order date/time features, "
        "city and order type. Pickup time is known after placement; actual delivery time "
        "is the regression target; late status is target-derived. Rider assignment and "
        "vehicle features are timing-uncertain. Weather, traffic and festival availability "
        "at placement is unverified. These fields are excluded from the saved model "
        "feature allowlist.",
    )
    _add_paragraph(
        document,
        "The PS2 contract is documented in `reports/PS2_prediction_contract.md`. "
        "The live demo accepts order-placement information only. Distance requires "
        "the explicit operational assumption that both coordinate pairs are available "
        "at scoring time.",
    )

    _page_break(document)
    document.add_heading("Delay patterns and regression", level=1)
    _add_paragraph(
        document,
        "The observed mean delivery time is 26.29 minutes and the median is 26 minutes. "
        "Late rates differ across recorded traffic and distance groups. For Jam traffic, "
        f"{ps3['traffic'][3]['late_rate']:.1%} of {ps3['traffic'][3]['order_count']:,} "
        "orders were late, versus "
        f"{ps3['traffic'][0]['late_rate']:.1%} of {ps3['traffic'][0]['order_count']:,} "
        "Low-traffic orders. In the 12+ km band, "
        f"{ps3['distance_bands'][-1]['late_rate']:.1%} of "
        f"{ps3['distance_bands'][-1]['order_count']:,} deliveries were late. "
        "These are retrospective associations with potential confounding, not causal effects.",
    )
    _add_chart(
        document,
        _bar_chart(
            "Observed late-order rate by traffic category",
            [
                (str(row["analysis_traffic"]), row["late_rate"] * 100)
                for row in ps3["traffic"]
                if row["analysis_traffic"] != "(Missing)"
            ],
            "%",
        ),
    )
    document.add_heading("PS4 regression comparison", level=2)
    model_rows = [
        (
            row["model"],
            f"{row['cv_mae_minutes']:.3f}",
            f"{row['test_metrics']['mae_minutes']:.3f}",
            f"{row['test_metrics']['rmse_minutes']:.3f}",
            f"{row['test_metrics']['r2']:.3f}",
        )
        for row in reg["model_comparison"]
    ]
    _add_table(document, ["Model", "CV MAE", "Test MAE", "Test RMSE", "Test R²"], model_rows, font_size=8)
    _add_paragraph(
        document,
        f"Gradient Boosting was selected by lowest training-only cross-validated MAE "
        f"({next(row['cv_mae_minutes'] for row in reg['model_comparison'] if row['model'] == reg['selected_model']):.3f} min). It achieved held-out MAE "
        f"{reg['selected_test_metrics']['mae_minutes']:.3f} min, RMSE "
        f"{reg['selected_test_metrics']['rmse_minutes']:.3f} min and R² "
        f"{reg['selected_test_metrics']['r2']:.3f}. Its test MAE improved "
        f"{reg['mae_improvement_vs_baseline_percent']:.1f}% over the mean baseline. "
        "XGBoost test MAE is nearly tied, with only a 0.0015-minute difference; this "
        "does not establish meaningful superiority.",
    )
    _add_chart(
        document,
        _bar_chart(
            "Held-out regression MAE (minutes)",
            [(row["model"], row["test_metrics"]["mae_minutes"]) for row in reg["model_comparison"]],
            " min",
        ),
    )

    _page_break(document)
    document.add_heading("Late-order risk and threshold decision", level=1)
    _add_paragraph(
        document,
        f"PS5 predicts whether an order exceeds the provisional SLA at order placement. "
        f"Gradient Boosting was selected; its threshold remains {clf['selected_threshold']:.2f}. "
        "Model comparison and threshold selection use training-only validation. The "
        "held-out test set was not used for choosing the threshold.",
    )
    _add_table(
        document,
        ["Model", "CV avg precision", "Validation precision", "Validation recall", "F2", "ROC-AUC"],
        [
            (
                row["model"],
                f"{row['cv_average_precision']:.3f}",
                f"{row['validation_metrics']['precision']:.3f}",
                f"{row['validation_metrics']['recall']:.3f}",
                f"{row['validation_metrics']['f_beta']:.3f}",
                f"{row['validation_metrics']['roc_auc']:.3f}",
            )
            for row in clf["model_comparison"]
        ],
        font_size=7.8,
    )
    _add_callout(
        document,
        "Why F2 and 0.15",
        f"The selected threshold maximized validation F2 ({clf['selected_validation_metrics']['f_beta']:.3f}; "
        f"recall {clf['selected_validation_metrics']['recall']:.1%}, precision "
        f"{clf['selected_validation_metrics']['precision']:.1%}). F2 emphasizes recall "
        "to reduce missed late-order signals, while the trade-off is more false-positive "
        "intervention flags.",
        PALE_TEAL,
    )
    _add_paragraph(
        document,
        f"At the unchanged threshold, the held-out test flagged {test['flagged_orders']:,} "
        f"of {test['order_count']:,} orders ({test['flagged_share']:.1%}); "
        f"{test['false_positives']:,} were false positives and "
        f"{test['false_negatives']:,} were false negatives. Test precision was "
        f"{test['precision']:.1%}, recall {test['recall']:.1%}, F1 {test['f1']:.3f} "
        f"and ROC-AUC {test['roc_auc']:.3f}. These test metrics describe the fixed "
        "operating point; they did not choose it.",
    )
    _add_paragraph(
        document,
        "The 0.15 point is not proven cost-optimal. No measured false-negative/false-positive "
        "cost ratio or operational alert capacity was supplied. A high flag share and low "
        "precision require an operations-defined review capacity before any intervention. "
        "The dashboard slider is a validation-only what-if control; it does not change the "
        "saved model threshold.",
    )
    document.add_heading("Explainability", level=2)
    _add_paragraph(
        document,
        "Saved held-out permutation importance shows predictive reliance, not causal "
        "influence. Peak-hour and validated distance are among the strongest features "
        "in both tasks. Correlated and derived features can share importance.",
    )

    _page_break(document)
    document.add_heading("Geographic zones and rider groups", level=1)
    document.add_heading("PS6 demand zones", level=2)
    _add_paragraph(
        document,
        f"PS6 used {geo['customer_coordinate_observations']:,} valid customer-coordinate "
        f"observations; {geo['customer_coordinate_records_excluded']:,} records with "
        "invalid/zero/out-of-envelope customer coordinates were excluded from geographic "
        "analysis. K=2–10 were compared using weighted inertia and a reproducible "
        f"silhouette sample. The selected K={geo['selected_cluster_count']} has silhouette "
        f"{geo['selected_silhouette_score']:.3f}; the inertia elbow suggests K="
        f"{geo['inertia_elbow_suggestion']}. K=10 is the candidate-range boundary, so "
        "finer partitions may score better and this is not a globally optimal site plan.",
    )
    zone = geo["candidate_zone_ranking"][0]
    _add_table(
        document,
        ["Evidence-ranked zone", "Orders", "Demand share", "Late rate", "Mean time", "Nearest restaurant"],
        [[
            f"Zone {zone['cluster_id']}",
            f"{zone['order_count']:,}",
            f"{zone['demand_share']:.1%}",
            f"{zone['late_rate']:.1%}",
            f"{zone['mean_delivery_minutes']:.2f} min",
            f"{zone['nearest_restaurant_km']:.1f} km",
        ]],
        font_size=8,
    )
    _add_paragraph(
        document,
        "The ranking is a relative investigation aid, not site suitability or financial "
        "return. Restaurant coordinates are only a rough coverage proxy; real site work "
        "needs verified property availability, rent, inventory, road access, staffing, "
        "capacity and operational feasibility.",
    )
    document.add_heading("PS7 rider segments", level=2)
    segment_rows = []
    for segment in rider["segments"]:
        segment_rows.append(
            (
                f"Segment {segment['segment_id']}",
                f"{segment['rider_count']:,}",
                f"{segment['late_rate']:.1%}",
                f"{segment['mean_delivery_minutes']:.2f}",
                f"{segment['mean_deliveries_per_rider']:.1f}",
            )
        )
    _add_table(
        document,
        ["Segment (not rank)", "Riders", "Late rate", "Mean time (min)", "Mean deliveries/rider"],
        segment_rows,
        font_size=8,
    )
    _add_paragraph(
        document,
        f"Of {rider['total_riders']:,} riders, {rider['segmented_riders']:,} met "
        f"the 30-delivery eligibility screen and {rider['insufficient_sample_riders']:,} "
        "were flagged insufficient. K=3 "
        "was selected at the inertia elbow; silhouette 0.212 indicates substantial "
        "profile overlap. Rider ID is a group key only; age is not tenure.",
    )

    _page_break(document)
    document.add_heading("Three prioritised management recommendations", level=1)
    for index, recommendation in enumerate(solution["recommendations"], start=1):
        document.add_heading(
            f"{index}. {recommendation['title']}  |  {recommendation['priority']} priority",
            level=2,
        )
        _add_paragraph(document, recommendation["evidence"], bold_lead="Evidence:")
        _add_paragraph(document, recommendation["action"], bold_lead="Action:")
        _add_paragraph(document, recommendation["validation"], bold_lead="Validation:")
    document.add_heading("Limitations and implementation", level=2)
    _add_bullets(
        document,
        [
            "The historical source is food delivery, not verified quick-commerce operations; transferability must be tested.",
            "The provisional SLA and coordinate availability at scoring time require operational confirmation.",
            "No intervention impact, alert capacity, financial benefits or return on investment was measured.",
            "Start risk scoring in shadow mode; keep human decisions and audit model/data versions.",
            "Verify location, property and rider-workload records before site or roster actions.",
        ],
    )
    path = OUTPUT / "Business_Report.docx"
    document.save(path)
    return path


def build_business_solution() -> Path:
    report, quality = _quality_values()
    reg = report["regression"]
    clf = report["classification"]
    test = clf["selected_test_metrics"]
    ps3 = report["ps3_findings"]
    geo = report["geographic_analysis"]
    rider = _read_json("artifacts/metrics/rider_segmentation_results.json")
    output_report = _read_json("reports/management_insights.json")
    solution = _make_document(
        "Business Solution Document",
        "Quick-Commerce Delivery Intelligence  |  Technical and operational solution",
    )

    solution.add_heading("1. Executive summary", level=1)
    _add_paragraph(
        solution,
        "The proposed delivery-intelligence workflow combines data-quality controls, "
        "retrospective delay analysis, ETA regression, late-risk classification, "
        "demand-zone clustering and rider cohort descriptions. It is designed to help "
        "operations teams decide what to investigate—not automate rider judgments or "
        "promise delivery outcomes. The submitted dataset is food-delivery data used as "
        "a quick-commerce proxy.",
    )
    _add_table(
        solution,
        ["Workstream", "Evidence-led result"],
        [
            ("Delay lens", "Mean 26.29 min; 29.8% late under a provisional 30-minute cutoff."),
            ("ETA", f"Gradient Boosting; held-out MAE {reg['selected_test_metrics']['mae_minutes']:.2f} min."),
            ("Late risk", f"Gradient Boosting, fixed threshold 0.15; test recall {test['recall']:.1%}."),
            ("Demand zones", f"K={geo['selected_cluster_count']}, silhouette {geo['selected_silhouette_score']:.3f}; elbow K={geo['inertia_elbow_suggestion']}."),
            ("Rider groups", "Three descriptive groups among 720 riders meeting the existing 30-delivery screen."),
        ],
    )
    _add_callout(
        solution,
        "Use boundary",
        "Predictions are estimates. The SLA is provisional. No model result establishes "
        "causality, site profitability, or an individual rider's ability.",
    )

    _page_break(solution)
    solution.add_heading("2. Business problem and objectives", level=1)
    _add_paragraph(
        solution,
        "The challenge describes unpredictable deliveries, customer-facing delay risk, "
        "dark-store investment decisions and rider capacity planning. The dataset can "
        "describe historical delivery outcomes and provide offline predictive evidence. "
        "It does not include verified store operations, property economics, road-network "
        "times, intervention histories or rider tenure. The solution therefore separates "
        "what is observed from proposed operational tests.",
    )
    _add_table(
        solution,
        ["Objective", "Output", "Decision boundary"],
        [
            ("Understand delay patterns", "PS3 descriptive summaries and filters.", "Associations are not causes; no rider tenure field."),
            ("Estimate ETA", "Saved PS4 pipeline returns minutes.", "Offline accuracy does not guarantee individual ETA."),
            ("Flag potential late orders", "Saved PS5 pipeline returns risk and status.", "Threshold is not cost-optimal without costs/capacity."),
            ("Locate demand concentrations", "PS6 K-Means zones and candidate rank.", "Not a dark-store feasibility or ROI result."),
            ("Describe rider cohorts", "PS7 standardized K-Means groups.", "Not an individual ranking or employment decision."),
        ],
    )
    solution.add_heading("Stakeholder use", level=2)
    _add_paragraph(
        solution,
        "Operations analysts can use dashboards to inspect historical patterns and "
        "validate candidate process changes. Dispatch teams may evaluate a human-reviewed "
        "risk triage only after shadow scoring and alert-capacity definition. Site planners "
        "should treat the zone ranking as a prompt for field validation. Rider support "
        "teams should use segment profiles only for group-level, voluntary process inquiry.",
    )

    _page_break(solution)
    solution.add_heading("3. Dataset, quality and PS2 methodology", level=1)
    _add_paragraph(
        solution,
        f"The source contains {report['dataset']['rows']:,} rows and "
        f"{report['dataset']['columns']} columns. The preparation pipeline retained "
        f"{report['dataset']['retained_rows']:,} rows; duplicate order IDs after the "
        "first were zero. It normalizes missing tokens and validates numeric, temporal "
        "and geographic fields without universal row deletion or outcome imputation.",
    )
    _add_table(
        solution,
        ["Quality dimension", "Observed evidence", "Treatment"],
        [
            ("Delivery target", "45,593 valid target rows; times range 10–54 min.", "Parsed target; retained valid observations."),
            ("Coordinates", f"{report['dataset']['valid_distance_observations']:,} model-eligible distances; {report['dataset']['invalid_or_unavailable_distance_observations']:,} missing/invalid.", "Invalid, zero-pair and out-of-envelope distances are missing to models, not genuine locations."),
            ("Rating", "53 values outside the valid 1–5 range; 1,908 missing.", "Invalid values flagged; valid observations used where sufficient."),
            ("Rider age", "38 values flagged outside the configured plausible range; 1,854 missing.", "Audit only; never treated as experience or tenure."),
            ("Timestamp / categories", "Order times, traffic, weather, city and other fields include missing tokens.", "Normalize missingness; exclude unverified availability from prediction features."),
        ],
        font_size=7.8,
    )
    solution.add_heading("Target, SLA and late label", level=2)
    _add_paragraph(
        solution,
        "`Time_taken(min)` is delivery duration in minutes and the PS4 regression target. "
        "The PS5 label is 1 only when actual delivery time is strictly greater than the "
        "configured 30 minutes; exactly 30 is on time. The guide asks the team to state an "
        "SLA but does not publish an approved value; 30 minutes is explicitly provisional.",
    )
    solution.add_heading("Distance and leakage policy", level=2)
    _add_paragraph(
        solution,
        "Store-to-customer distance is the Haversine great-circle distance (Earth radius "
        "6,371 km), not road distance. Coordinates outside the broad India screening "
        "envelope or zero pairs do not enter model distance. Both model pipelines use the "
        "same nine-feature allowlist: distance, order hour/day, weekday, weekend/peak flags, "
        "distance band, city and order type. Pickup time is post-order; actual duration is "
        "the target; late status is target-derived. Rider/vehicle assignment timing and "
        "weather/traffic/festival availability at placement are unverified. These fields "
        "are excluded. The standalone contract is `reports/PS2_prediction_contract.md`.",
    )

    _page_break(solution)
    solution.add_heading("4. Exploratory delay findings", level=1)
    _add_paragraph(
        solution,
        f"Across {report['dataset']['rows']:,} valid delivery outcomes, the mean is "
        f"{ps3['mean_delivery_minutes']:.2f} minutes and median is "
        f"{ps3['median_delivery_minutes']:.0f} minutes. The observed late share is "
        f"{ps3['late_rate']:.1%} under the provisional threshold. Group comparisons "
        "measure association only; they do not adjust for all confounders or establish "
        "an intervention effect.",
    )
    traffic_rows = [
        (
            row["analysis_traffic"],
            f"{row['order_count']:,}",
            f"{row['mean_delivery_minutes']:.2f}",
            f"{row['late_rate']:.1%}",
        )
        for row in ps3["traffic"]
        if row["analysis_traffic"] != "(Missing)"
    ]
    _add_table(solution, ["Traffic", "Orders", "Mean delivery (min)", "Late rate"], traffic_rows)
    _add_chart(
        solution,
        _bar_chart(
            "Observed late-order rate by distance band",
            [
                (row["analysis_distance_band"], row["late_rate"] * 100)
                for row in ps3["distance_bands"]
            ],
            "%",
        ),
    )
    long_band = ps3["distance_bands"][-1]
    _add_paragraph(
        solution,
        f"The 12+ km group had a {long_band['late_rate']:.1%} late rate "
        f"({long_band['order_count']:,} orders), compared with "
        f"{ps3['distance_bands'][0]['late_rate']:.1%} for 0–3 km. Longer distance, "
        "traffic category, time of day and location differ together in the historical data; "
        "the comparisons are hypotheses for like-for-like operational review, not causal claims.",
    )
    _add_paragraph(
        solution,
        "Rider experience is a requested analytical comparison in the guide, but no "
        "verified experience/tenure column exists. The project does not use age as a proxy.",
    )

    _page_break(solution)
    solution.add_heading("5. Delivery-time regression (PS4)", level=1)
    _add_paragraph(
        solution,
        "The regression pipeline fits median imputation, scaling/encoding and estimator "
        "within training folds. A reproducible 80/20 split is used; candidate selection "
        "uses training-only cross-validation MAE. A latest-date slice of the training "
        "partition provides a separate temporal check. The held-out test does not tune "
        "features, hyperparameters or the selected model.",
    )
    _add_table(
        solution,
        ["Model", "Training CV MAE", "Test MAE", "Test RMSE", "Test R²"],
        [
            (
                row["model"],
                f"{row['cv_mae_minutes']:.3f}",
                f"{row['test_metrics']['mae_minutes']:.3f}",
                f"{row['test_metrics']['rmse_minutes']:.3f}",
                f"{row['test_metrics']['r2']:.3f}",
            )
            for row in reg["model_comparison"]
        ],
        font_size=8,
    )
    _add_chart(
        solution,
        _bar_chart(
            "Held-out MAE comparison (minutes)",
            [(row["model"], row["test_metrics"]["mae_minutes"]) for row in reg["model_comparison"]],
            " min",
        ),
    )
    _add_paragraph(
        solution,
        f"Gradient Boosting was chosen because it had the lowest training-only CV MAE "
        f"({next(row['cv_mae_minutes'] for row in reg['model_comparison'] if row['model'] == reg['selected_model']):.3f} min). Its held-out test MAE/RMSE/R² were "
        f"{reg['selected_test_metrics']['mae_minutes']:.3f}/"
        f"{reg['selected_test_metrics']['rmse_minutes']:.3f}/"
        f"{reg['selected_test_metrics']['r2']:.3f}. The mean baseline test MAE was "
        f"{reg['baseline_test_metrics']['mae_minutes']:.3f} min. XGBoost test MAE differs "
        "by about 0.0015 minutes, which is not evidence of meaningful operational superiority.",
    )
    _add_paragraph(
        solution,
        "Saved held-out permutation importance ranks peak-hour, validated distance and "
        "city among the stronger predictors. This measures model reliance, not causal "
        "impact. Error remains substantial; the ETA is an estimate, not a promise.",
    )

    _page_break(solution)
    solution.add_heading("6. Late-order classification (PS5)", level=1)
    _add_paragraph(
        solution,
        "The classifier shares PS4's placement-time feature builder and predicts the "
        "configured late label. Class prevalence is about 29.8%; class-weight candidates "
        "are compared and synthetic resampling is not used. Candidate selection and "
        "threshold selection are separated from the final held-out test.",
    )
    _add_table(
        solution,
        ["Model", "CV average precision", "Validation precision", "Validation recall", "F2", "ROC-AUC"],
        [
            (
                row["model"],
                f"{row['cv_average_precision']:.3f}",
                f"{row['validation_metrics']['precision']:.3f}",
                f"{row['validation_metrics']['recall']:.3f}",
                f"{row['validation_metrics']['f_beta']:.3f}",
                f"{row['validation_metrics']['roc_auc']:.3f}",
            )
            for row in clf["model_comparison"]
        ],
        font_size=7.8,
    )
    _add_callout(
        solution,
        "Threshold rationale — unchanged 0.15",
        f"0.15 maximized F2 on training-only validation (F2 "
        f"{clf['selected_validation_metrics']['f_beta']:.3f}; recall "
        f"{clf['selected_validation_metrics']['recall']:.1%}; precision "
        f"{clf['selected_validation_metrics']['precision']:.1%}). F2 weights recall "
        "more heavily to prioritize identifying potentially late orders. Lower thresholds "
        "can capture more late orders but create unnecessary intervention flags.",
        PALE_TEAL,
    )
    _add_table(
        solution,
        ["Fixed test evidence", "Value"],
        [
            ("Orders flagged", f"{test['flagged_orders']:,}/{test['order_count']:,} ({test['flagged_share']:.1%})"),
            ("Precision / recall / F1 / ROC-AUC", f"{test['precision']:.1%} / {test['recall']:.1%} / {test['f1']:.3f} / {test['roc_auc']:.3f}"),
            ("False positives / false negatives", f"{test['false_positives']:,} / {test['false_negatives']:,}"),
            ("Confusion matrix", f"TN {test['true_negatives']:,}; FP {test['false_positives']:,}; FN {test['false_negatives']:,}; TP {test['true_positives']:,}"),
        ],
    )
    _add_paragraph(
        solution,
        "This threshold is not proven cost-optimal because no false-negative/false-positive "
        "intervention cost ratio or operational alert capacity was measured. These test "
        "metrics are an evaluation of a fixed threshold; held-out labels were not used to "
        "select or tune it. A threshold what-if in the dashboard changes validation results "
        "only; the saved threshold and model artifact remain unchanged.",
    )
    _add_paragraph(
        solution,
        "Operational use should begin with shadow scoring. A human-review workflow should "
        "be considered only after confirming the SLA, input availability and review capacity. "
        "Do not automatically alter customer promises, dispatch decisions or rider treatment.",
    )

    _page_break(solution)
    solution.add_heading("7. Demand-zone discovery (PS6)", level=1)
    _add_paragraph(
        solution,
        f"Customer coordinates provide {geo['customer_coordinate_observations']:,} valid "
        f"observations; {geo['customer_coordinate_records_excluded']:,} invalid/zero or "
        "out-of-region customer coordinates were excluded. Restaurant locations are "
        "retained as a separate rough coverage layer. K-Means uses customer location and "
        "order-frequency weights; delivery target, late label and restaurant points do not "
        "fit the clustering model.",
    )
    _add_table(
        solution,
        ["K", "WCSS (projected km²)", "Silhouette", "Orders in smallest zone"],
        [
            (
                row["cluster_count"],
                f"{row['inertia_km_squared'] / 1_000_000:.1f}M",
                f"{row['silhouette_score']:.3f}",
                f"{row['smallest_cluster_orders']:,}",
            )
            for row in geo["cluster_count_evaluation"]
        ],
        font_size=7.5,
    )
    _add_chart(
        solution,
        _bar_chart(
            "Silhouette across evaluated geographic K",
            [(str(row["cluster_count"]), row["silhouette_score"]) for row in geo["cluster_count_evaluation"]],
            "",
        ),
    )
    _add_paragraph(
        solution,
        f"Selected K={geo['selected_cluster_count']} has silhouette "
        f"{geo['selected_silhouette_score']:.3f}; the inertia elbow suggests "
        f"K={geo['inertia_elbow_suggestion']}. Since the silhouette maximum occurs at "
        "the upper evaluated boundary K=10, it does not establish a globally optimal or "
        "stable site partition. The relative candidate score is an investigation aid only.",
    )
    zone = geo["candidate_zone_ranking"][0]
    _add_callout(
        solution,
        f"Leading evidence candidate: zone {zone['cluster_id']}",
        f"{zone['order_count']:,} orders ({zone['demand_share']:.1%} share), "
        f"{zone['late_rate']:.1%} late, {zone['mean_delivery_minutes']:.2f} min mean "
        f"delivery, nearest supplied restaurant {zone['nearest_restaurant_km']:.1f} km. "
        "This is not a site recommendation; property/rent/inventory/road-access and "
        "operational feasibility data are absent.",
        PALE_TEAL,
    )

    _page_break(solution)
    solution.add_heading("8. Rider segments and group support hypotheses (PS7)", level=1)
    _add_paragraph(
        solution,
        f"Rider-level aggregation includes {rider['data_audit']['distinct_riders_with_completed_deliveries']:,} "
        f"riders. {rider['data_audit']['riders_meeting_minimum_deliveries']:,} met the "
        "30-completed-delivery minimum; 600 were retained but flagged insufficient. "
        "Six standardized features represent volume, delivery time, late rate, distance, "
        "valid rating and multiple-delivery load. Rider ID is not a model feature. "
        "Rider age is not treated as verified experience.",
    )
    _add_table(
        solution,
        ["Group (not rank)", "Riders", "Late rate", "Mean time", "Mean deliveries", "Observed profile vs eligible average"],
        [
            (
                f"Segment {segment['segment_id']}",
                f"{segment['rider_count']:,}",
                f"{segment['late_rate']:.1%}",
                f"{segment['mean_delivery_minutes']:.2f} min",
                f"{segment['mean_deliveries_per_rider']:.1f}",
                segment["operational_description"],
            )
            for segment in rider["segments"]
        ],
        font_size=7,
    )
    for segment in rider["segments"]:
        segment_id = int(segment["segment_id"])
        actions = SEGMENT_SUPPORT_ACTIONS[segment_id]
        solution.add_heading(f"Segment {segment_id} — group-level hypotheses", level=2)
        _add_paragraph(
            solution,
            f"Measured: {segment['rider_count']} riders, {segment['late_rate']:.1%} "
            f"late rate, {segment['mean_delivery_minutes']:.2f} min mean delivery, "
            f"{segment['mean_deliveries_per_rider']:.1f} mean deliveries per rider. "
            f"Other observed profile: {segment['operational_description']}.",
        )
        _add_bullets(
            solution,
            [
                f"Potential operational support: {actions['support']}",
                f"Suggested training / process review: {actions['training']}",
                f"Roster / workload consideration: {actions['roster']}",
                f"Monitoring KPI: {actions['kpi']}",
                f"Validation / limitation: {actions['validation']}",
            ],
        )
    _add_callout(
        solution,
        "Safeguard",
        "These are group-level hypotheses, not causal findings or individual ratings. "
        "Do not penalize, rank, dismiss or restrict opportunities based on segment membership. "
        "The modest silhouette (0.212) and 30-order screen require visible uncertainty and "
        "validation across route/time mixes.",
        PALE_TEAL,
    )

    _page_break(solution)
    solution.add_heading("9. Exactly three prioritised management recommendations", level=1)
    if len(output_report["recommendations"]) != 3:
        raise ValueError("The saved management evidence must contain exactly three actions.")
    for number, recommendation in enumerate(output_report["recommendations"], start=1):
        solution.add_heading(
            f"{number}. {recommendation['title']} — {recommendation['priority']} priority",
            level=2,
        )
        _add_paragraph(solution, recommendation["evidence"], bold_lead="Evidence:")
        _add_paragraph(solution, recommendation["problem"], bold_lead="Problem:")
        _add_paragraph(solution, recommendation["action"], bold_lead="Proposed action:")
        _add_paragraph(
            solution,
            recommendation["expected_benefit"],
            bold_lead="Potential benefit (not measured):",
        )
        _add_paragraph(
            solution,
            "Implementation: " + " ".join(recommendation["implementation"]),
        )
        _add_paragraph(
            solution,
            "Risks: " + " ".join(recommendation["risks"]),
        )
        _add_paragraph(
            solution,
            "Monitoring: " + recommendation["monitoring_kpi"],
        )
        _add_paragraph(
            solution,
            "Validation: " + recommendation["validation"],
        )

    _page_break(solution)
    solution.add_heading("10. Limitations, assumptions and implementation", level=1)
    _add_bullets(
        solution,
        [
            "The source is food-delivery data, not verified quick-commerce/dark-store operations. Historical metrics may not transfer.",
            "Thirty minutes is a provisional SLA; organizer and operations confirmation is required before business use.",
            "Validated coordinate availability at scoring time is an assumption; Haversine distance is straight-line, not road travel.",
            "The classifier threshold reflects validation F2, not measured costs or staffing capacity; test flags indicate substantial review volume.",
            "The PS6 silhouette maximum is at the tested upper K boundary; zone rankings do not establish optimal locations.",
            "The PS7 silhouette is modest, segment estimates are sample-dependent, and rider tenure/experience is unavailable.",
            "No intervention has been tested. No savings, ROI, causal effects or guaranteed delivery improvement are reported.",
        ],
    )
    solution.add_heading("Implementation sequence", level=2)
    _add_table(
        solution,
        ["Stage", "Control / evidence required"],
        [
            ("Confirm policy", "Confirm the SLA, late definition and decision-time availability of every input."),
            ("Shadow validation", "Run saved ETA/risk models without operational actions; audit data drift, calibration, missingness and flagged share."),
            ("Human-reviewed pilot", "Only after review capacity and a pre-registered comparison plan are approved; do not auto-penalize riders or alter promises."),
            ("Spatial diligence", "Ground-truth coordinates; gather current property, rent, road-access, inventory, capacity and staffing information."),
            ("Rider support trial", "Voluntary, group-level support hypotheses; compare like-for-like routes/time windows and publish uncertainty."),
        ],
        font_size=8,
    )

    _page_break(solution)
    solution.add_heading("11. Technical architecture and reproducibility", level=1)
    _add_paragraph(
        solution,
        "The application follows a small Python/Streamlit architecture. `src/data/` "
        "loads, cleans, validates and engineers shared features. `src/analytics/` "
        "contains PS3, PS6, PS7 and PS8 analyses. `src/models/` contains PS4/PS5 "
        "training, saved inference and the order-placement wrapper. `src/visualization/` "
        "renders separate PS3–PS8 dashboards and the saved-model order demo. `pages/` "
        "registers the multi-page Streamlit interface. Tests exercise data policy, "
        "metrics, persistence, reproducibility, filters and live inference.",
    )
    _add_table(
        solution,
        ["Task", "Command"],
        [
            ("Install", "`python -m pip install -r requirements.txt`"),
            ("Prepare data", "`python -m src.data.pipeline`"),
            ("Generate verified PS8 report", "`python -m src.analytics.management_insights`"),
            ("Generate submission documents", "`python submission/build_documents.py`"),
            ("Run application", "`python -m streamlit run app.py`"),
            ("Run all tests", "`python -m unittest discover -s tests -v`"),
        ],
    )
    _add_paragraph(
        solution,
        "The submitted management report is generated from saved model comparisons, "
        "predictions and reports; its verification checks data provenance and recomputes "
        "held-out metrics without retraining. The document generator reads saved evidence "
        "and does not fit models. It requires `python-docx` and Pillow. No source dataset, "
        "model or threshold is modified while building documents.",
    )
    _add_paragraph(
        solution,
        "Exactly three management actions are reproduced from the verified PS8 report. "
        "The PS7 support matrix above consists of segment-specific, non-punitive test "
        "hypotheses and is not an additional set of PS8 management recommendations.",
    )
    path = OUTPUT / "Business_Solution_Document.docx"
    solution.save(path)
    return path


def main() -> None:
    report_path = build_business_report()
    solution_path = build_business_solution()
    print(f"Business report: {report_path}")
    print(f"Business solution: {solution_path}")


if __name__ == "__main__":
    main()
