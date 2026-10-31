# Quick-Commerce Delivery Intelligence — Final Management Report

## Project status

| Problem statement | Status | Evidence / qualification |
|---|---|---|
| PS1 | Implemented | Data audit, quality reports, and validated preprocessing are present. |
| PS2 | Documented against official guide | The standalone contract and saved-model prediction demo cover the target, provisional SLA/late definition, Haversine distance, and prediction-time leakage exclusions. |
| PS3 | Implemented with data limitation | Saved delay analysis and dashboard; the dataset has no verified rider experience/tenure field, so that comparison is unavailable. |
| PS4 | Implemented | Saved regression model comparisons, test predictions, importance, and dashboard. |
| PS5 | Implemented with cost limitation | Saved classifier, validation-only threshold selection, held-out predictions, and dashboard; no measured intervention-cost ratio is available to claim a cost-optimal threshold. |
| PS6 | Implemented with exploratory limitations | Saved customer-coordinate clustering, diagnostics, candidate ranking, and dashboard. |
| PS7 | Implemented with descriptive limitations | Saved rider aggregation, K-Means segmentation, measured profiles, assignments, and dashboard; segment-specific support/roster/training actions are not supplied. |
| PS8 | Implemented | This report, explainability evidence, exactly three management actions, and the final dashboard. |

The official challenge guide was reviewed from the workspace-root TXT file. PS2's target, SLA/late definition, Haversine distance, and leakage exclusions are documented; the 30-minute SLA remains provisional.

## Dataset and SLA

- 45,593 food-delivery records; 45,593 retained by preprocessing.
- 0 duplicate order IDs after the first row.
- 41,522 valid distance observations; 4,071 invalid or unavailable.
- Observed late-order share: 29.8% (13,606 orders).
- SLA: provisional. A delivery is late only when time is strictly greater than 30 minutes.
- The source is a food-delivery dataset used as a proxy, not verified quick-commerce or dark-store operations data.

## Regression model selection

Selected **Gradient Boosting** using the saved training-only cross-validation MAE criterion. No model was retrained for this report.

| Model | CV MAE (min) | Test MAE (min) | Test RMSE (min) | Test R² |
|---|---:|---:|---:|---:|
| Mean Baseline | 7.585 | 7.579 | 9.364 | -0.000 |
| Linear Regression | 6.558 | 6.631 | 8.260 | 0.222 |
| Random Forest | 6.784 | 6.803 | 8.532 | 0.170 |
| Gradient Boosting | 6.351 | 6.406 | 7.984 | 0.273 |
| XGBoost | 6.358 | 6.404 | 7.983 | 0.273 |

The mean baseline test MAE was 7.579 minutes. Selected-model MAE improved by 1.173 minutes (15.5%). Gradient Boosting and XGBoost test MAEs differ by only 0.0015 minutes; this small difference does not establish a meaningful deployment advantage.

MAE is average absolute delivery-time error; RMSE is more sensitive to larger errors; R² is the share of test-target variation accounted for relative to a mean baseline. The selected tree ensemble is less directly interpretable than linear regression and still leaves substantial error. These are offline estimates, not a guarantee for an individual order. Gradient Boosting is a nonlinear ensemble with more implementation/inference complexity than linear regression; XGBoost's test MAE is effectively tied, while the saved sklearn Gradient Boosting model avoids requiring XGBoost for the selected estimator. No serving-latency or deployment-cost benchmark was measured.

## Late-order classification

Selected **Gradient Boosting** at the fixed **0.15** threshold. Threshold selection used training-only validation and maximized F2; no held-out test label was used to choose or change it.

| Model | CV average precision | Validation precision | Validation recall | Validation F2 | Validation ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.550 | 0.405 | 0.906 | 0.726 | 0.769 |
| Random Forest | 0.535 | 0.389 | 0.943 | 0.734 | 0.773 |
| Gradient Boosting | 0.571 | 0.403 | 0.946 | 0.745 | 0.793 |
| XGBoost | 0.575 | 0.405 | 0.938 | 0.743 | 0.793 |

Model comparisons above use validation results. Only the selected model's held-out test evaluation is reported; test labels were not used for model or threshold selection.

On 9,119 test orders: precision 39.4%, recall 94.6%, F1 0.556, F2 0.739, ROC-AUC 0.784.
Confusion matrix (rows actual on-time/late; columns predicted on-time/late): [[2,431, 3,967], [146, 2,575]]. The model flagged 6,542 orders (71.7%); false positives were 3,967, and false negatives were 146.

Threshold rationale: 0.15 maximized F2 on the training-only validation split (F2 0.745, recall 94.6%, precision 40.3%). F2 weights recall more heavily than precision because missed late orders matter for early identification; the trade-off is additional false-positive intervention flags. At the unchanged threshold, the held-out test flagged 6,542/9,119 orders (71.7%); 3,967 were false positives and 146 were false negatives. This is not a cost-optimal threshold: intervention costs and alert capacity are not measured, and held-out test labels were not used to select it. High recall means most observed late orders were flagged, not that alerts are operationally efficient. No alert costs or capacity were invented. On validation, the 0.15 threshold had precision 40.3%, recall 94.6%, and F2 0.745; at 0.50, precision was 60.2%, recall 46.0%, and F2 0.483. The classifier identifies retrospective risk and does not prevent delays. Precision is the share of flagged orders that were actually late; recall is the share of actual late orders flagged. F1 is their harmonic mean, and F2 weights recall more heavily. ROC-AUC measures ranking discrimination across thresholds; it does not select an operating point or establish probability calibration. The confusion matrix gives the true/false positive and negative counts at the fixed threshold.

## Explainability

Permutation importance is calculated on the saved held-out evaluation data. Regression values are MAE increase in minutes; classification values are average precision decrease. They measure predictive reliance, not causal influence.

| Regression feature | MAE increase (min) |
|---|---:|
| is_peak_hour | 0.6651 |
| delivery_distance_km | 0.5418 |
| City | 0.2935 |
| order_hour | 0.1971 |
| distance_band | 0.0233 |

| Classification feature | Average-precision decrease |
|---|---:|
| is_peak_hour | 0.1037 |
| delivery_distance_km | 0.0902 |
| City | 0.0495 |
| order_hour | 0.0210 |
| order_day | 0.0037 |

Peak-hour and distance features rank highly in both saved analyses. Pickup time is unavailable at order placement and excluded. Weather/traffic are retrospective context only; distance is usable only if validated coordinates are available when scored. Correlated/derived features can split or obscure permutation importance.

## PS3–PS7 evidence

- PS3: average/median delivery time is 26.29/26.0 minutes. Traffic, weather, time, and distance differences are observational associations.
- PS6: selected K=10, silhouette 0.765; the inertia elbow suggests K=3. PS6 selected K=10, which is the upper boundary of the evaluated range (maximum K=10); the silhouette result does not establish a globally optimal or stable site partition.
- PS7: 720 of 1,320 riders met the 30-delivery screen; 600 were flagged for insufficient observations. Segments are descriptive, not individual rankings.
- Restaurant coordinates are only a coverage proxy; actual property, rent, inventory, road-access, staffing, and operational-cost data are absent.

## Exactly three prioritised management actions

### 1. Pilot human-reviewed late-order risk triage — High priority

**Evidence:** On 9,119 held-out orders, the selected classifier at its training-validation threshold of 0.15 achieved 94.6% recall and 39.4% precision. It flagged 6,542 orders, including 3,967 false positives. The observed late-order share was 29.8%.

**Operational problem:** The retrospective classifier catches many late orders, but its low precision means a large share of alerts would be unnecessary.

**Action:** Evaluate the locked order-placement classifier in shadow mode, then consider a controlled human-review pilot only after operations defines alert capacity and the official SLA. Keep decisions with staff; do not automatically change delivery promises, dispatch, or rider treatment.

**Expected benefit:** Could help direct attention toward orders with higher estimated late risk. The dataset does not measure intervention effectiveness, staffing capacity, or savings, so no numerical benefit is claimed.

**Implementation requirements:**
- Confirm the operational SLA and that each predictor is available at the proposed scoring time.
- Use the saved model and locked 0.15 threshold for the initial offline/shadow evaluation; do not tune against the held-out test labels.
- Have operations define a review workflow and alert capacity before any live pilot.
- Log scores, review actions, subsequent outcomes, and model/data version for audit.

**Risks and limitations:**
- At the selected test operating point, 3,967 of 6,542 flags were false positives; observed test precision was 39.4%.
- The evaluation is retrospective on a food-delivery proxy dataset; calibration and feature availability may change in a live service.
- The threshold optimizes validation F2 rather than verified operational costs or staffing capacity.

**Monitoring KPI:** Flagged-order share, precision, recall, false-positive and false-negative counts, review response time, and late rate among reviewed versus comparable unreviewed orders.

**Validation:** Start with shadow scoring. If operationally feasible, run a pre-registered controlled pilot with a suitable comparison group and confidence intervals; evaluate outcome changes and alert workload without using test labels to retune.

### 2. Investigate peak-period, distance, and workload processes — High priority

**Evidence:** PS3 observed a 50.6% late rate for Jam traffic (n=14,143) versus 7.9% for Low traffic (n=15,477); orders in the 12+ km band had a 45.1% late rate (n=14,881). PS4 held-out permutation importance was 0.665 minutes MAE increase for peak-hour and 0.542 for distance. PS7's segment late rates ranged from 25.2% to 34.8%; these groups are descriptive and formed partly from late-rate and volume features.

**Operational problem:** Observed delivery outcomes vary across traffic, distance, and time groups, but the dataset cannot establish which process or resource constraints cause the differences.

**Action:** Conduct a non-punitive operational review of peak-period and longer-distance orders, recording dispatch/load conditions and available rider workload. Use PS7 segment profiles only to frame aggregate support questions, never to rank or discipline individuals.

**Expected benefit:** May identify process constraints worth testing. No causal effect, improvement size, or rider-level explanation is established by these retrospective associations.

**Implementation requirements:**
- Confirm traffic/weather data definitions and timestamps; they were retrospective context, not model inputs.
- Validate the configured peak-hour windows and compare like-for-like locations, dates, and distance bands.
- Review dispatch timing, order load, and operating conditions at aggregate cohort level before selecting a specific intervention.
- Keep sufficient-observation flags visible and do not infer rider tenure from age.

**Risks and limitations:**
- Traffic and weather categories may be missing, subjective, confounded, or recorded after the decision point.
- PS7 uses a 30-delivery screening floor, not a guarantee of stable rider estimates; its silhouette score is modest.
- The PS3 subgroup differences are not causal estimates and may not transfer to another service or time period.

**Monitoring KPI:** Delivery-time median and p90, late-order rate, distance-adjusted outcomes, and workload/assignment measures by pre-defined time and operating-condition cohort.

**Validation:** Predefine comparable pilot and control cohorts, measure a baseline period, then use a randomized or defensible phased rollout where feasible. Report sample sizes and uncertainty and check that any aggregate differences persist.

### 3. Validate geographic evidence before site commitments — Medium priority

**Evidence:** The highest PS6 relative candidate score is zone 4 (score 77.0), with 3,993 orders (9.5% of clustered demand), 31.9% late, and a nearest supplied restaurant distance of 37.4 km. PS6 selected K=10, which is the upper boundary of the evaluated range (maximum K=10); the silhouette result does not establish a globally optimal or stable site partition.

**Operational problem:** The exploratory zones are derived from food-delivery customer coordinates; restaurants are only a service-coverage proxy, and the score is not a site-feasibility measure.

**Action:** Treat the top-ranked zone as a validation candidate for further study, not a dark-store recommendation. Reconcile source coordinates and time coverage, then assess actual candidate properties and operating feasibility before any investment decision.

**Expected benefit:** Could reduce the chance of acting on a geographic artifact and focus field research on measured demand. No site benefit, savings, or commercial return can be estimated from the available data.

**Implementation requirements:**
- Verify customer and existing-location coordinates against trusted operational records and current service boundaries.
- Recompute demand using recent time windows and sensitivity checks across plausible cluster counts.
- Collect candidate property availability, rent, inventory, road access, staffing, capacity, and operational-cost data.
- Keep the weighted zone score labeled as an exploratory ranking, separate from a feasibility or investment score.

**Risks and limitations:**
- The selected PS6 K is at the upper end of the evaluated range, so finer partitions may score better.
- The source is a food-delivery dataset, not verified dark-store operations; restaurant locations do not prove dark-store coverage.
- No property, rent, road-access, inventory, or operating-cost data is available.

**Monitoring KPI:** Coordinate validation rate, demand-share stability by time window and K, candidate-zone order coverage, and independently verified service-area coverage.

**Validation:** Reconcile a sample of locations with ground-truth records, test spatial and temporal stability on a separate period, and complete field/property feasibility checks before proposing a site.

## Remaining limitations

- The official guide was reviewed from QUICK_COMMERCEDELIVERY_INTELLIGENCE.txt in the workspace root.
- The 30-minute SLA remains provisional because the guide does not publish an approved threshold.
- The 30-minute SLA is provisional and must be confirmed before operational use.
- The food-delivery dataset is only a quick-commerce proxy and may not represent the intended operation or current conditions.
- All model metrics are retrospective; performance, calibration, and feature availability may change in deployment.
- The classification threshold is based on validation F2, not verified operating capacity or measured false-positive/false-negative costs.
- PS6 zones and restaurant locations do not establish dark-store suitability; the selected K is at the evaluated boundary.
- PS7 segments are descriptive groups and must not be used as individual rankings or employment judgments; the proposed support actions are hypotheses requiring validation.
- No intervention has been tested; the report makes no quantified savings, ROI, delivery-improvement, or causal claims.

## Reproducibility

The report is generated from saved PS1, PS3, PS4, PS5, PS6, and PS7 reports, saved predictions, importance tables, and fitted regression/classification models. The generation step checks dataset provenance, SLA consistency, model/report identity, model predictions, and recomputed test metrics. It does not retrain models.
