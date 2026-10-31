# Requirement-to-Evidence Checklist

## Audit basis

The official guide at `../QUICK_COMMERCEDELIVERY_INTELLIGENCE.txt` was
reviewed as the source of truth, including Sections 4 (eight problem
statements), 7 (mandatory evidence), 13 (submission package), 14 (recommended
report structure), and 15 (judging rubric). The package section gives no
archive format, filename convention, or size limit.

## Problem statement evidence

| Problem statement | Status from repository evidence | Evidence and qualification |
|---|---|---|
| PS1 — data quality and preparation | Implemented | `src/data/loader.py`, `cleaning.py`, `validation.py`, `feature_engineering.py`, and `pipeline.py`; quality reports under `reports/`; `tests/test_data_pipeline.py`. Missingness, duplicates, invalid coordinates, ratings/ages, timestamps, and outliers are reported with treatment. |
| PS2 — target, SLA, distance, and prediction-time data | Implemented against listed requirements | `reports/PS2_prediction_contract.md` documents `Time_taken(min)`, strict `> 30` late logic, provisional SLA, Haversine distance, and post-order/post-delivery exclusions. The guide does not publish an approved SLA, so 30 minutes remains provisional. |
| PS3 — delivery-delay analysis | Implemented with a data limitation | `src/analytics/delay_analysis.py`, `src/visualization/delay_dashboard.py`, `reports/delay_analysis.md`, and `tests/test_delay_analysis.py` provide multiple comparisons and non-causal observations. The dataset has no verified rider tenure/experience field, so the experience comparison cannot be established; age is not treated as experience. |
| PS4 — ETA regression | Implemented | Saved fitted model and metrics under `artifacts/models/` and `artifacts/metrics/`; training/inference in `src/models/regression.py`; dashboard in `src/visualization/regression_dashboard.py`; `tests/test_regression.py`. |
| PS5 — late-order risk | Implemented; operational cost evidence unavailable | Saved classifier, training-only threshold selection, validation/test evidence, and importance under `artifacts/`; inference in `src/models/classification.py`; `tests/test_classification.py`. The 0.15 threshold is justified by validation F2/recall preference; organizer-specified false-negative/false-positive cost ratios are absent, so no cost-optimal threshold can be claimed. |
| PS6 — demand zones | Implemented with exploratory limitations | `src/analytics/demand_zones.py`, `src/visualization/demand_zone_dashboard.py`, and saved K-Means/diagnostics/assignments under `artifacts/`; `tests/test_demand_zones.py`. Silhouette maximum is at the evaluated K boundary, so it does not establish a globally optimal site partition. |
| PS7 — rider segmentation | Implemented with descriptive limitations | `src/analytics/rider_segmentation.py`, `src/visualization/rider_segmentation_dashboard.py`, `src/analytics/rider_support.py`, and saved model/assignments under `artifacts/`; `tests/test_rider_segmentation.py`. Segment-specific support, training/process-review, roster/workload, monitoring, and validation hypotheses are shown in the dashboard and Business Solution Document. No verified tenure/experience field exists; segments are not individual rankings or employment recommendations. |
| PS8 — explainability and management actions | Implemented | `src/analytics/management_insights.py`, `reports/management_report.md`, `reports/management_insights.json`, `src/visualization/management_dashboard.py`, and `tests/test_management_insights.py`. The report verifies saved outputs without retraining and contains exactly three actions. |

## Six required submission deliverables

| Deliverable | Evidence/status | Notes |
|---|---|---|
| Complete runnable source code/notebook | Present and verified | Source code, `requirements.txt`, README commands, 66 passing tests and Python compilation. No notebook is required when runnable source code is supplied. |
| Model comparison table | Present | `reports/management_report.md` compares regression candidates against the mean baseline and classification candidates using validation metrics; selected models/reasons and held-out metrics are reported. |
| Business Report (recommended 4–6 pages) | Present; page count verified | `submission/Business_Report.docx` is 6 pages, counted in Microsoft Word after pagination. It is generated from verified report evidence and retains exactly three prioritised recommendations. |
| Working Streamlit ETA and delay-risk demo | Present and verified | `pages/Order_Prediction_Demo.py` uses saved PS4/PS5 artifacts, returns ETA/risk, and retains threshold 0.15. AppTest and live home/demo routes passed. |
| Business Solution Document (minimum 5 pages) | Present; page count verified | `submission/Business_Solution_Document.docx` is 13 pages, counted in Microsoft Word after pagination. It includes the requested business, data, PS2–PS7 methods/results, exactly three recommendations, limitations, and reproducibility sections. |
| README with setup, dependencies, assumptions, and execution instructions | Present | `README.md` documents installation, pipeline/tests, application/model run commands, SLA/data assumptions, feature availability, limitations, and PS5's fixed 0.15 validation-F2 rationale. |

## Final gap-closure verification

| Identified gap | Action taken | Evidence/output | Verification status | Remaining limitation |
|---|---|---|---|---|
| Standalone Business Solution Document, minimum five pages | Created a separate evidence-based DOCX with the required sections, measured results, charts/tables, exactly three recommendations, and implementation limitations. | `submission/Business_Solution_Document.docx` | Verified: Microsoft Word reports 13 pages; document structure and section content inspected. | Findings remain retrospective and depend on the supplied food-delivery dataset and stated assumptions. |
| PS7 segment-specific support actions | Added non-punitive group-level support, training/process-review, roster/workload, KPI, and validation hypotheses for all three observed segments; wired the selected-segment view to dashboard support content. | `src/analytics/rider_support.py`, `src/visualization/rider_segmentation_dashboard.py`, `tests/test_rider_segmentation.py`, and the Business Solution Document | Verified: included segment-filter/support test passed in the full 66-test suite; all three groups' guidance is present in the generated document. | Actions are hypotheses to test, not proven interventions or causal explanations; no individual employment decisions are supported. |
| PS5 threshold rationale consistency | Documented why validation F2 prioritizes recall, the false-positive trade-off, fixed-threshold test flag count/share, and the lack of measured costs/capacity. Kept threshold at 0.15 and did not retrain or retune. | `src/visualization/classification_dashboard.py`, `src/analytics/management_insights.py`, `reports/management_report.md`, `README.md`, and both submission DOCX files | Verified: report regeneration and dashboard/report tests passed; saved threshold is 0.15, validation F2 is 0.745, and fixed test flags are 6,542/9,119 (71.7%), with 3,967 false positives and 146 false negatives. | Threshold is not established as cost-optimal because intervention costs and alert capacity are unmeasured; test labels were not used for threshold selection. |
| Business Report pagination and format | Generated a formatted DOCX from the existing evidence report without changing model results or the recommendation set. | `submission/Business_Report.docx` | Verified: Microsoft Word reports 6 pages; document contains exactly three prioritised recommendations. | Page count is specific to the generated DOCX and its current Word layout. |
| Requirement checklist currency | Updated this checklist with completed actions, output paths, verification evidence, and limitations. | `reports/requirement_evidence_checklist.md` | Updated after document generation, pagination, report regeneration, compilation, and full test execution. | Future changes to artifacts require rerunning these checks and refreshing this checklist. |

## Validation record

- Full test suite: **66 tests passed** using `python -m unittest discover -s tests -v`.
- Python compilation: **passed** using `python -m compileall -q src pages submission tests`.
- Management evidence/report regeneration: **passed**; the saved threshold remained 0.15 and the report retained exactly three recommendations.
- Microsoft Word pagination: Business Report **6 pages**; Business Solution Document **13 pages**.
- Existing model artifacts and saved evaluation results were used; no model retraining or threshold retuning was performed.

No archive has been created because the guide specifies no archive format or
file-naming convention. Review dataset/model redistribution permissions before
external submission.

## Release limitations to carry into submission

- The data is food-delivery data used as a quick-commerce proxy.
- The 30-minute SLA is provisional and must not be presented as organizer-approved.
- Model metrics are retrospective; the live demo outputs estimates, not guarantees.
- Coordinate availability at order placement is assumed by the models, not proven by the source data.
- PS6 restaurant coordinates are only a rough coverage proxy; candidate zones
  require property and operational feasibility checks.
- PS7 groups are descriptive and must not support individual employment,
  compensation, or disciplinary decisions.
- PS5's selected threshold reflects a validation F2 policy, not measured
  operational intervention costs.
