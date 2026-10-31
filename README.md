# Quick-Commerce Delivery Intelligence — PS1–PS8

The project contains the Phase 1 data-quality pipeline, PS3 descriptive
delivery-delay analysis, PS4 delivery-time regression, PS5 late-order
classification, exploratory PS6 customer-demand clustering, PS7 rider
performance segmentation, and PS8 evidence-integrated management insights.
The official challenge guide is available at
[`../QUICK_COMMERCEDELIVERY_INTELLIGENCE.txt`](../QUICK_COMMERCEDELIVERY_INTELLIGENCE.txt)
and was used for the requirements audit. No PDF copy is kept in this project.
The supplied `train.csv` is a food-delivery dataset used as a quick-commerce
proxy; it is not verified dark-store operations data.

## Project layout

```text
Quick_Commerce/
├── app.py                         # Streamlit data-readiness audit
├── pages/
│   ├── Delivery_Delay_Intelligence.py
│   ├── Regression_Model_Laboratory.py
│   ├── Late_Order_Risk_Intelligence.py
│   ├── Dark_Store_Zone_Intelligence.py
│   ├── Rider_Performance_Segmentation.py
│   ├── Order_Prediction_Demo.py
│   └── Management_Insights.py
├── config.py                      # Paths, schema, provisional SLA, feature policy
├── train.csv                      # Original workspace file, retained unchanged
├── data/
│   ├── raw/train.csv               # Byte-identical raw input copy
│   └── processed/cleaned_orders.csv
├── src/data/
│   ├── loader.py
│   ├── cleaning.py
│   ├── validation.py
│   ├── feature_engineering.py
│   └── pipeline.py
├── src/analytics/
│   ├── delay_analysis.py          # PS3 descriptive summaries and reports
│   ├── demand_zones.py            # PS6 geographic clustering and zone evidence
│   ├── rider_segmentation.py      # PS7 rider performance aggregation and K-Means
│   └── management_insights.py     # PS8 saved-evidence checks and final report
├── src/models/
│   ├── regression.py              # PS4 training, evaluation, and inference
│   ├── classification.py          # PS5 late-risk training and inference
│   └── order_prediction.py        # Validated PS4/PS5 inference
├── src/visualization/
│   ├── delay_dashboard.py         # Streamlit/Plotly delay dashboard
│   ├── regression_dashboard.py    # Streamlit/Plotly regression lab
│   ├── classification_dashboard.py # Streamlit/Plotly late-risk analysis
│   ├── demand_zone_dashboard.py   # Streamlit/Plotly zone dashboard
│   ├── rider_segmentation_dashboard.py # Streamlit/Plotly rider segments
│   ├── order_prediction_dashboard.py # Saved-model order prediction demo
│   └── management_dashboard.py    # PS8 model, evidence, and management view
├── reports/
│   ├── data_quality_report.json
│   ├── data_quality_report.md
│   ├── delay_analysis.json
│   ├── delay_analysis.md
│   ├── PS2_prediction_contract.md
│   ├── requirement_evidence_checklist.md
│   ├── management_insights.json
│   └── management_report.md
├── artifacts/
│   ├── models/
│   │   ├── delivery_time_regressor.joblib
│   │   ├── late_order_classifier.joblib
│   │   ├── demand_zone_kmeans.joblib
│   │   └── rider_performance_kmeans.joblib
│   └── metrics/
│       ├── regression_results.json
│       ├── regression_test_predictions.csv
│       ├── regression_feature_importance.csv
│       ├── classification_results.json
│       ├── classification_test_predictions.csv
│       ├── classification_validation_predictions.csv
│       ├── classification_feature_importance.csv
│       ├── demand_zone_results.json
│       ├── demand_zone_assignments.csv
│       ├── demand_zone_restaurants.csv
│       ├── rider_segmentation_results.json
│       ├── rider_segment_assignments.csv
│       └── rider_order_assignments.csv
└── tests/
    ├── test_data_pipeline.py
    ├── test_delay_analysis.py
    ├── test_regression.py
    ├── test_classification.py
    ├── test_demand_zones.py
    ├── test_rider_segmentation.py
    ├── test_order_prediction.py
    └── test_management_insights.py
```

## Run the audit

From this directory, install runtime dependencies and run:

```powershell
python --version  # Python 3.10 or later
python -m pip install -r requirements.txt
python -m src.data.pipeline
python -m unittest discover -s tests -v
```

The pipeline writes a processed CSV and both JSON and Markdown reports. To use
another CSV with the same required schema, pass `--input PATH`.

## PS2: delivery-time and prediction contract

The explicit PS2 deliverable is
[`reports/PS2_prediction_contract.md`](reports/PS2_prediction_contract.md).
It defines the delivery-time target, strict late-order rule, Haversine
store-to-customer distance, and prediction-time feature exclusions. The
30-minute cutoff is **provisional**, not organizer-approved; the guide asks
teams to state their SLA but does not publish an approved value. Confirm the
operational SLA with the organizer before use.

The interactive **New Order Prediction Demo** uses the existing saved PS4 ETA
and PS5 late-risk models without retraining. It accepts placement-time inputs
only, retains the saved 0.15 classifier threshold, and returns estimates rather
than guarantees. From the project directory, run:

```powershell
python -m streamlit run app.py
```

Coordinates are optional but must be provided as a complete, validated store
and customer pair; otherwise distance is imputed by the saved pipeline and the
demo displays a warning. Pickup time, actual delivery outcomes, and rider
details are not requested.

## PS3: diagnose delay patterns

The PS3 module consumes `data/processed/cleaned_orders.csv` and reuses the
Phase 1 timestamp parsing, coordinate validation, weather/traffic
normalization, delivery-time extraction, and provisional SLA. It does not
impute values or remove outliers. Orders with a valid delivery time remain in
general comparisons even when their order time, rating, or coordinates are
missing. Each comparison uses its own valid observations, and the dashboard
shows counts of unavailable values.

Run the descriptive report and dashboard from this directory:

```powershell
python -m src.analytics.delay_analysis
python -m streamlit run app.py
python -m unittest discover -s tests -v
```

Streamlit shows the original data-readiness audit and separate delivery-delay,
regression, late-risk, dark-store-zone, rider-segmentation, and management-insights
pages. The delay page supports date, order-hour, weather,
traffic, order-type, city, and valid-distance filters. The full-network
benchmark is computed from all valid delivery outcomes and remains fixed while
filters change.

PS3 reports mean/median/standard deviation and delivery-time percentiles;
hour/day/weather/traffic/order-type/rating comparisons; distance-band
summaries; rider summaries requiring at least 30 valid orders; and
sample-qualified observed patterns. Those patterns are associations, not
causal claims. Rider tenure is not present in the supplied columns and is not
inferred. The 30-minute late-order cutoff remains provisional and is configured
centrally in `config.py`.

Distance summaries use Haversine distance only for coordinate pairs passing
the existing broad regional validation. Invalid or missing coordinates are
not converted into plausible distances. The straight-line distance is not
road distance. Date/hour summaries omit records whose corresponding timestamp
cannot be parsed; default dashboard filters retain such records in overall
KPIs, with the missing-data counts shown separately.

## PS4: delivery-time regression

The regression experiment consumes the Phase 1 processed dataset and uses the
configured `Time_taken(min)` target. It trains a mean baseline, Linear
Regression, Random Forest, Gradient Boosting, and XGBoost (when installed).
Run training from the project directory:

```powershell
python -m src.models.regression
python -m streamlit run app.py
```

The model lab appears as **Regression Model Laboratory** in Streamlit after
the saved artifacts are generated. It shows the held-out model comparison,
baseline improvement, actual-versus-predicted values, residuals, error by
actual delivery-time range, and held-out permutation importance.

Predictors are the validated distance, order time/calendar features, distance
band, destination city, and submitted order type. Weather, traffic, and
festival are excluded because this dataset does not establish that they are
known at order placement. Rider IDs, rider/vehicle attributes, pickup time,
IDs, the target, and target-derived labels are excluded. Using distance
assumes the store and customer coordinates are available when the order is
placed; the dataset itself does not verify that production condition. Invalid
distance is treated as missing by the training-fitted imputer.

The random 80/20 split is fixed at seed 42. A latest-date holdout from the
training partition provides a separate chronological check; the test set is
not used for tuning or model selection. Small randomized hyperparameter
searches use three-fold cross-validation on the earlier development portion
of training data. The final model is selected by lowest cross-validated MAE,
with held-out MAE, RMSE, R², and temporal metrics reported separately.
Preprocessing lives in each sklearn pipeline so imputers and encoders are fit
only on training folds and reused unchanged for inference.

The saved inference function is `predict_delivery_time` in
`src/models/regression.py`. It accepts a mapping of raw order-placement fields
(`Order_Date`, `Time_Orderd`, coordinates, `City`, and `Type_of_order`), rejects
identifiers and post-outcome fields, and returns predicted minutes plus any
imputation warnings. Missing or malformed optional values are handled by the
saved preprocessing pipeline.

## PS5: late-order classification

The classifier labels an order late only when its actual delivery time is
strictly greater than the configured provisional SLA in `config.py`. It reuses
the PS4 order-placement feature set and preprocessing, so rider assignment,
pickup time, the delivery target, and target-derived fields cannot enter the
classifier. Weather and traffic are shown only as retrospective outcome
context; their availability at placement is not established by this dataset.

Train models and run the risk page:

```powershell
python -m src.models.classification
python -m streamlit run app.py
```

The train/test split is stratified and reproducible. A second stratified split
of training data is used for model comparison, probability calibration, and
operating-threshold selection. The held-out test set is excluded from tuning
and threshold selection and is evaluated only after the chosen model and
threshold are fixed. Candidate models are tuned by training-only average
precision; the operating point prioritizes recall using F2, with precision
and intervention volume as tie-breaks. No false-negative or false-positive
cost values were supplied, so none are assumed. The configurable cost helper
accepts explicitly supplied costs if the team later receives defensible
operational estimates.

The saved threshold of 0.15 maximized F2 on training-only validation
(F2 0.745; recall 94.6%; precision 40.3%). F2 gives recall more weight to
surface potentially late orders early, while the trade-off is unnecessary
false-positive intervention flags. At this unchanged threshold, the held-out
test flagged 6,542 of 9,119 orders (71.7%), with 3,967 false positives and
146 false negatives. The operating point is not cost-optimal evidence:
intervention costs and alert capacity are unknown, and test labels were not
used to select the threshold.

The observed late-order share is approximately 29.8%, so this dataset is not
treated as severely imbalanced. Class-weight options are compared during
tuning; synthetic resampling is not used. Validation Brier scores and
reliability bins are compared with sigmoid calibration, and calibration is
retained only if it improves validation Brier score. The dashboard separates
validation threshold exploration from fixed held-out test metrics.

The saved inference function is `predict_late_order` in
`src/models/classification.py`. It accepts the same supported raw
order-placement fields as regression and returns a risk score, locked
threshold, predicted status, model version, and input warnings. A score is
qualified by the saved calibration evaluation; it is not a guarantee of
production calibration.

## PS6: dark-store demand-zone exploration

PS6 clusters validated customer coordinates to identify concentrated
customer-order demand. Delivery-time outcomes, late labels, distance, rider
IDs, and restaurant coordinates do not fit K-Means. Exact repeated customer
coordinate pairs contribute order-frequency sample weights; source rows are
not duplicated. A local east/north kilometer projection is used for this
dataset, and the broad India coordinate envelope is a screening assumption.

Run clustering and open the dedicated page:

```powershell
python -m src.analytics.demand_zones
python -m streamlit run app.py
```

The default experiment evaluates k=2 through k=10, reports weighted inertia
and a fixed-seed silhouette sample, and selects the highest-silhouette
candidate. The inertia elbow is reported separately as a coarse-grained
diagnostic and can disagree with that choice. If the best silhouette is at
the largest evaluated k, the dashboard warns that finer partitions may score
higher. Review the map, centroids, cluster sizes, and diagnostics before
using this exploratory partition; the evaluated range does not establish a
globally optimal k. An analyst can run an explicitly selected evaluated k
with `python -m src.analytics.demand_zones --clusters 3`.

The dashboard provides a selectable zone map with customer-order density,
demand-weighted centroids, and a distinct layer for supplied restaurant
locations. Zone selection updates demand, delivery, late-rate, and distance
comparisons against network benchmarks. Elbow, silhouette, cluster-size,
and candidate-ranking views are available. The relative candidate score uses
documented percentile ranks and weights (demand 35%, observed late rate 25%,
average delivery time 20%, geographic concentration 10%, and distance to a
supplied restaurant 10%). It is an investigation aid, not a forecast of
savings, profit, or causal effect. Restaurant locations are only a rough
coverage proxy.

The source has no customer identifier, so plotted points aggregate exact
coordinate pairs; order volume is not a unique-customer count. Actual site
selection additionally requires property availability, rent, inventory,
road access, staffing, capacity, and operational feasibility. The model,
projection settings, centroids, cluster-count diagnostics, row assignments,
restaurant-location layer, and JSON report are saved under `artifacts/`.
The saved analysis selected K=10 with a sampled silhouette score of about
0.765; its inertia-elbow suggestion is K=3. Since K=10 is the largest
evaluated value, the silhouette maximum is at the candidate-range boundary
and is not evidence of a globally optimal or stable site partition.

## PS7: rider performance segmentation

Run the analysis and open **Rider Performance Segmentation**:

```powershell
python -m src.analytics.rider_segmentation
python -m streamlit run app.py
```

PS7 reuses the PS3 prepared data, valid delivery targets, Haversine distance,
validated ratings, and configured SLA. Completed deliveries are grouped by
the rider identifier; rider ID is never a clustering feature. Riders with
fewer than 30 valid completed deliveries are retained in the output and
dashboard as insufficient-sample records, but are excluded from K-Means.
Thirty orders is the existing PS3 rider-reporting threshold, not a guarantee
that a rider's performance estimate is stable. Rider late-rate Wilson
intervals and delivery-volume filters make sample size visible.

The standardized features are log delivery count, mean delivery time,
late-delivery rate, mean valid delivery distance, average valid rider rating,
and mean valid multiple-delivery count. Rating and distance averages require
at least five valid observations; ratings must be within 1–5. Missing feature
values are median-imputed by a pipeline fit only on eligible riders. Missing
indicators are not clustering features, so the model does not directly
segment riders by coordinate/rating coverage; availability remains visible
in the audit and rider detail table. Rider age is excluded because it is not
verified experience or tenure. Vehicle condition and the entirely missing
vehicle-type field are not used.

The default experiment evaluates K=2 through K=8, with fixed seed 42,
inertia, and a reproducible silhouette sample. The selection rule uses the
inertia elbow when its silhouette is within 0.01 of the best candidate and
all segments have at least 30 riders; otherwise it selects the best
silhouette candidate meeting that size screen when available, falling back to
the best silhouette in the evaluated range if no candidate meets it. This
small-segment and 0.01 tolerance policy is an explicit analysis choice;
diagnostics, segment sizes, and actual profiles are shown for review. A
selected segment is a descriptive grouping, not a rank, causal finding,
individual performance verdict, or employment recommendation. The configured
30-minute SLA remains provisional.

Segment metrics report the actual delivery-level mean and median delivery
time, late rate, distance, and volume, with rider-level volume distributions,
available ratings, and sample-sufficiency counts. The page supports segment
and completed-delivery-volume filters; both update rider counts, order-level
metrics, comparison charts, and rider details. Saved artifacts include the
preprocessor, fitted K-Means, diagnostics, rider assignments (including
unsegmented low-volume riders), per-order segment assignments, and JSON
report under `artifacts/`. The page also shows segment-specific operational
support, process/training review, roster/workload considerations, and KPIs as
group-level hypotheses. These are non-punitive and require validation; they
do not establish causal rider characteristics, rank individuals, or imply
experience/tenure.

## PS8: model evidence and management insights

Generate the final management report from saved model outputs, then run the
Streamlit application:

```powershell
python -m src.analytics.management_insights
python -m streamlit run app.py
```

The command verifies that regression and classifier predictions correspond to
their saved fitted models, recomputes held-out metrics, checks selected model
and threshold metadata, checks dataset provenance and SLA consistency, and
writes `reports/management_insights.json` and `reports/management_report.md`.
It does not retrain either model or retune the classifier threshold. The
**Management Insights** page presents saved model comparisons, explainability,
PS3–PS7 evidence, limitations, and exactly three management actions. Model
and rider-segment selectors update their metrics; the classifier threshold
remains locked and its final test results are not used to select a threshold.

| Problem statement | Completion status |
|---|---|
| PS1 | Implemented: data audit and validated preprocessing. |
| PS2 | Documented against the official guide; target, stated provisional SLA, Haversine distance, leakage exclusions, and live demo. |
| PS3 | Implemented: descriptive delivery-delay analysis. |
| PS4 | Implemented: ETA regression, comparisons, and saved diagnostics. |
| PS5 | Implemented: late-order classifier with validation-only threshold selection. |
| PS6 | Implemented with exploratory limitations: geographic zones and candidate ranking. |
| PS7 | Implemented with descriptive limitations: rider-level segmentation. |
| PS8 | Implemented: verified model synthesis, explainability, three actions, and dashboard. |

PS4 selected Gradient Boosting because it had the lowest training-only
cross-validation MAE (6.351 minutes). Its held-out test MAE/RMSE/R² were
6.406 minutes / 7.984 minutes / 0.273; the mean baseline test MAE was
7.579 minutes. XGBoost test MAE was 6.404 minutes, a difference of only
0.0015 minutes. MAE is average absolute error, RMSE gives larger misses more
weight, and R² describes test-set variation relative to a mean baseline.

PS5 selected Gradient Boosting at the 0.15 threshold using training-only
validation F2. Its held-out test precision/recall/F1/ROC-AUC were
39.4%/94.6%/0.556/0.784. It flagged 6,542 of 9,119 test orders; 3,967 were
false positives and 146 were false negatives. High recall does not establish
operational efficiency, and the classifier does not prevent delays. No
alert-capacity or error-cost assumptions are encoded.

PS8 explainability uses existing held-out permutation importance, not
fabricated SHAP values: regression importance is measured as MAE increase,
and classification importance as average-precision decrease. Peak-hour,
validated distance, and city show the largest saved predictive reliance.
These are associations, not causes. Pickup time is unavailable at order
placement and excluded; weather and traffic are retrospective context because
their availability at placement is unverified.

The saved PS6 analysis chose K=10 at the top of its evaluated K=2–10 range,
while the inertia elbow suggests K=3. Restaurant locations are only a proxy
for coverage; actual property, rent, inventory, road access, staffing, and
operating-cost information is absent. PS7's groups are descriptive, have a
modest silhouette score, and must not be used to rank individual riders.

### Three prioritised management actions

1. **High — Pilot human-reviewed late-order risk triage.** Begin with shadow scoring and only consider a controlled human-review pilot after confirming the SLA, predictor availability, and review capacity. Monitor alert share, precision, recall, false-positive/negative counts, and review response; validate with a controlled comparison.
2. **High — Investigate peak-period, distance, and workload processes.** Review aggregate dispatch/load conditions for high-traffic and longer-distance cohorts, treating PS7 patterns only as non-punitive hypotheses. Monitor time and late-rate by comparable cohorts; test changes with a pre-defined control or phased rollout.
3. **Medium — Validate geographic evidence before site commitments.** Verify coordinates, demand stability, and the service-area proxy before field research; gather actual property and operational feasibility data before any investment decision. Monitor coordinate validity, zone stability, and independently confirmed coverage.

No action has a guaranteed effect or quantified savings. See the generated
management report for measured evidence, implementation requirements, risks,
KPIs, and validation approach.

## Project tests

Run all unit tests for Phase 1 and PS3–PS8, including the saved-model order
prediction demo:

```powershell
python -m unittest discover -s tests -v
```

Training regenerates model artifacts under `artifacts/`; the raw dataset and
processed dataset are read-only inputs to the experiments.

## What each module does

| Module | Input → output | Main logic | How to test |
|---|---|---|---|
| `loader.py` | CSV path → source DataFrame | Reads fields as strings so missing tokens stay visible; checks required columns. | Pipeline test and missing-file/schema errors |
| `cleaning.py` | Source DataFrame → normalized DataFrame and operation counts | Normalizes missing tokens/categories and parses numeric values; does not impute or drop rows. | Missing-value and target-extraction unit tests |
| `feature_engineering.py` | Cleaned DataFrame → features and audit flags | Parses timestamps, computes Haversine distances, masks model distance to the regional envelope, and selects features through an allowlist. | Distance, timestamp, SLA, invalid-coordinate, and leakage tests |
| `validation.py` | Source, cleaned, and enriched frames → report data and report files | Counts missing/duplicate/suspicious values, flags outliers, and renders JSON/Markdown. | Pipeline consistency test; inspect `reports/` |
| `pipeline.py` | Input path → processed CSV and reports | Runs the modules in order and returns output paths/counts. | `python -m src.data.pipeline` and full unit suite |
| `app.py` | Generated JSON report → Streamlit home page | Displays the audit viewer; Streamlit discovers analysis pages from `pages/`. | Streamlit AppTest smoke test |
| `src/analytics/delay_analysis.py` | Phase 1 processed frame → grouped metrics and reports | Prepares consistent analysis fields; computes distributions, comparisons, benchmarks, and observations. | `tests/test_delay_analysis.py` |
| `src/visualization/delay_dashboard.py` | Prepared PS3 frame → interactive dashboard | Renders filters, KPIs, benchmarks, and Plotly charts. | Streamlit page smoke test |
| `src/models/regression.py` | Phase 1 processed frame → saved estimator, comparison metrics, predictions, and inference API | Fits fold-contained preprocessing, compares regression candidates, and selects by training-only CV MAE. | `tests/test_regression.py` |
| `src/visualization/regression_dashboard.py` | Saved PS4 metrics and held-out predictions → model-lab page | Displays regression comparison and test diagnostics. | Streamlit page smoke test |
| `src/models/classification.py` | Processed orders → SLA labels, selected classifier, evaluation artifacts, and risk inference | Tunes on training folds, selects its threshold on a separate validation split, and evaluates on test once. | `tests/test_classification.py` |
| `src/visualization/classification_dashboard.py` | Saved PS5 report and validation/test predictions → risk dashboard | Shows model comparison, test evaluation, validation-only threshold control, calibration, and retrospective risk breakdowns. | Streamlit page smoke test |
| `src/analytics/demand_zones.py` | Validated processed orders → K-Means zones, diagnostics, evidence, and saved artifacts | Demand-weighted geographic clustering, cluster outcomes, restaurant coverage proxy, and explicit limitations. | `tests/test_demand_zones.py` |
| `src/visualization/demand_zone_dashboard.py` | Saved PS6 artifacts → interactive zone map and comparison dashboard | Zone selection updates order-level KPIs, the map, and delivery-performance comparisons. | `tests/test_demand_zones.py` and Streamlit AppTest |
| `src/analytics/rider_segmentation.py` | PS3 prepared delivery data → rider summaries, fitted preprocessing, K-Means segments, and artifacts | Excludes riders below the configured completed-delivery minimum; compares K with inertia and silhouette. | `tests/test_rider_segmentation.py` |
| `src/visualization/rider_segmentation_dashboard.py` | Saved PS7 artifacts → interactive rider-segment dashboard | Segment and delivery-volume filters update metrics, comparisons, and rider details. | `tests/test_rider_segmentation.py` and Streamlit AppTest |
| `src/analytics/management_insights.py` | Saved PS1–PS7 reports, fitted models, predictions, and importance → verified PS8 JSON/Markdown report | Recomputes held-out metrics, checks model/data/SLA consistency, and creates exactly three evidence-linked actions without retraining. | `tests/test_management_insights.py` |
| `src/visualization/management_dashboard.py` | PS8 report → final interactive management page | Compares saved models, explains feature reliance, surfaces PS3–PS7 evidence, and presents the three actions. | `tests/test_management_insights.py` and Streamlit AppTest |

`tests/test_data_pipeline.py` is the focused test suite for these modules. It
uses a tiny in-memory fixture for edge cases and a temporary directory for
pipeline outputs; it does not modify the supplied dataset.

## Decisions the team should be able to explain

| Decision | Why this was chosen | Assumption or alternative |
|---|---|---|
| Normalize missing tokens, but do not impute or drop rows. | The missingness is observable, but a universal fill value or row deletion could bias later analysis. | Choose imputation only inside a training pipeline, fitted on its training split; confirm with measured validation results. |
| Calculate Haversine distance, but keep only broad-region-valid distances as model features. | Zero and geographically implausible coordinates can otherwise produce misleadingly large distances. The raw calculation remains in the audit output so the issue is visible. | India bounds and the 100 km review threshold are screening heuristics. Official store/city boundaries would be better if supplied. |
| Flag IQR and percentile outliers without removing them. | Statistical extremeness alone does not establish a data error. | Review source records and compare model performance before considering robust transformations or exclusions. |
| Use an explicit order-placement feature allowlist. | It prevents accidental inclusion of pickup time, target-derived flags, or unverified rider-assignment fields. | Weather, traffic, festival and coordinates still require confirmation that they are available when the prediction is made. |
| Treat 30 minutes as a provisional SLA. | It provides a configurable initial late-order label, but the challenge does not set an official threshold. | Replace it centrally if the organizer confirms a different SLA. |

The original CSV is loaded read-only. The pipeline does not delete rows,
impute values, or drop statistical outliers. Missing tokens become nullable
values; numeric fields are parsed; suspicious ranges and outliers are marked
for review. Every operation records affected, retained, and excluded record
counts in the report. The SHA-256 digest records which raw file was audited.

Run the read-only report viewer after generating the report:

```powershell
python -m streamlit run app.py
```

## Data and prediction policy

- `Time_taken(min)` is parsed as the delivery-time target, in minutes.
- The late-order label is true only when delivery time is **greater than** the
  configured 30-minute SLA. This threshold is provisional; the challenge brief
  does not specify an official SLA.
- The report and viewer may show how many supplied outcomes exceed 30 minutes.
  This is a descriptive label count, not a model prediction or operational KPI.
- `Time_Order_picked` is excluded because it is observed after placement.
- Order ID, outcome-derived values, and rider/vehicle assignment-dependent
  attributes are excluded from the order-placement feature set.
- Weather, traffic, festival, and coordinates are conditional on being
  genuinely available at prediction time.
- No rider tenure is fabricated; the dataset has no rider-tenure field.
- Peak-hour windows and the broad India coordinate envelope are explicit,
  reviewable heuristics in `config.py`, not organizer-confirmed business rules.
- Haversine distance is calculated for globally in-range coordinates so
  implausible long pairs can be seen in the audit. The model feature
  `delivery_distance_km` remains missing unless both points fall within the
  broad India envelope. Distances over 100 km are review flags, not deletion
  rules.

Missing predictor values are retained rather than imputed. Imputation choices
belong in a later training pipeline and must be fit only on training data.
