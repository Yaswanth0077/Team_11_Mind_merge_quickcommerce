# Quick-Commerce Delivery Intelligence — Data Quality Report

## 1. Dataset overview
- Source: `E:\Manthan_calibo\Quick_Commerce\data\raw\train.csv`
- SHA-256: `d293b6fc7157aef30fb34301216a3432ec895d407ce2ea9edddf942b416cdde1`
- Shape: 45,593 rows × 20 columns
- Loaded DataFrame memory use: 14,109,430 bytes

| Column | Inferred CSV dtype | Cleaned dtype | Unique values |
|---|---|---|---:|
| `ID` | str | string | 45,593 |
| `Delivery_person_ID` | str | string | 1,320 |
| `Delivery_person_Age` | str | Int64 | 23 |
| `Delivery_person_Ratings` | str | Float64 | 29 |
| `Restaurant_latitude` | float64 | Float64 | 658 |
| `Restaurant_longitude` | float64 | Float64 | 519 |
| `Delivery_location_latitude` | float64 | Float64 | 4,373 |
| `Delivery_location_longitude` | float64 | Float64 | 4,373 |
| `Order_Date` | str | string | 44 |
| `Time_Orderd` | str | string | 177 |
| `Time_Order_picked` | str | string | 193 |
| `Weatherconditions` | str | string | 7 |
| `Road_traffic_density` | str | string | 5 |
| `Vehicle_condition` | int64 | Int64 | 4 |
| `Type_of_order` | str | string | 4 |
| `Type_of_vehicle` | str | string | 4 |
| `multiple_deliveries` | str | Int64 | 5 |
| `Festival` | str | string | 3 |
| `City` | str | string | 4 |
| `Time_taken(min)` | str | Int64 | 45 |

## 2. Missing-value investigation

| Column | Null | Empty | Whitespace | Missing tokens | Placeholders | Normalized missing | % |
|---|---:|---:|---:|---:|---:|---:|---:|
| `ID` | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| `Delivery_person_ID` | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| `Delivery_person_Age` | 0 | 0 | 0 | 1854 | 0 | 1854 | 4.07 |
| `Delivery_person_Ratings` | 0 | 0 | 0 | 1908 | 0 | 1908 | 4.18 |
| `Restaurant_latitude` | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| `Restaurant_longitude` | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| `Delivery_location_latitude` | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| `Delivery_location_longitude` | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| `Order_Date` | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| `Time_Orderd` | 0 | 0 | 0 | 1731 | 0 | 1731 | 3.80 |
| `Time_Order_picked` | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| `Weatherconditions` | 0 | 0 | 0 | 0 | 616 | 616 | 1.35 |
| `Road_traffic_density` | 0 | 0 | 0 | 601 | 0 | 601 | 1.32 |
| `Vehicle_condition` | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| `Type_of_order` | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| `Type_of_vehicle` | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |
| `multiple_deliveries` | 0 | 0 | 0 | 993 | 0 | 993 | 2.18 |
| `Festival` | 0 | 0 | 0 | 228 | 0 | 228 | 0.50 |
| `City` | 0 | 0 | 0 | 1200 | 0 | 1200 | 2.63 |
| `Time_taken(min)` | 0 | 0 | 0 | 0 | 0 | 0 | 0.00 |

## 3. Duplicate analysis

- Exact duplicate rows: 0
- Duplicate order-ID rows after first occurrence: 0
- Rows involved in duplicate order IDs: 0

## 4. Numeric validation

| Field | Valid values | Minimum | Maximum | Values flagged |
|---|---:|---:|---:|---:|
| Rider age | 43,739 | 15.0 | 50.0 | 38 |
| Rider rating | 43,685 | 1.0 | 6.0 | 53 |
| Multiple deliveries | 44,600 | 0.0 | 3.0 | 0 |
| Vehicle condition | 45,593 | 0.0 | 3.0 | 0 |
| Delivery time (minutes) | 45,593 | 10.0 | 54.0 | 0 |

## 5. Geographic validation

- Zero-coordinate pairs: 3,640.
- Zero store pairs: 3,640; zero customer pairs: 0.
- Pairs within global latitude/longitude ranges: 45,593 of 45,593.
- Pairs within the broad India review envelope: 41,522; flagged outside it: 4,071.

| Coordinate field | Zero values | Negative values | Outside global range |
|---|---:|---:|---:|
| `store_latitude` | 3,640 | 431 | 0 |
| `store_longitude` | 3,640 | 162 | 0 |
| `customer_latitude` | 0 | 0 | 0 |
| `customer_longitude` | 0 | 0 | 0 |

- Haversine distance computed for 45,593 globally in-range pairs. 431 exceeded the 100 km review threshold.
- Model-eligible distance range: 1.4650674052309467–20.969489380087342 km; median 9.193021332844554 km. Distances for pairs outside the regional envelope remain available separately for audit, not model input.

## 6. Timestamp validation

| Field | Expected format | Missing | Malformed nonmissing | Valid | Parsed range |
|---|---|---:|---:|---:|---|
| `Order_Date` | `%d-%m-%Y` | 0 | 0 | 45,593 | 2022-02-11 00:00:00 to 2022-04-06 00:00:00 |
| `Time_Orderd` | `%H:%M:%S` | 1,731 | 0 | 43,862 | 00:00:00 to 23:55:00 |
| `Time_Order_picked` | `%H:%M:%S` | 0 | 0 | 45,593 | 00:00:00 to 23:55:00 |

## 7. Outlier analysis

| Measure | IQR bounds | IQR flags | 1st/99th percentiles | Percentile flags |
|---|---|---:|---|---:|
| Delivery time (minutes) | -0.50 to 51.50 | 270 | 10.00, 49.00 | 436 |
| Model-eligible distance (km) | -8.88 to 27.22 | 0 | 1.49, 20.83 | 817 |

Percentile columns show p01 and p99. IQR outside Q1-1.5*IQR/Q3+1.5*IQR and values strictly below p01 or above p99 are flagged; all records are retained.
All flagged observations are retained.

## 8. Cleaning decisions

| Operation | Reason | Affected records | Retained | Excluded |
|---|---|---:|---:|---:|
| Normalize missing values | Represent actual nulls, blank strings, known string tokens, and the weather 'conditions NaN' placeholder consistently. | 4,225 | 45,593 | 0 |
| Trim categorical text and normalize weather labels | Remove inconsistent surrounding whitespace and the repeated 'conditions ' prefix without changing category meaning. | 45,593 | 45,593 | 0 |
| Convert numeric fields | Parse rider, coordinate, vehicle, delivery-count, and delivery-time fields as nullable numeric types; record unparseable nonmissing values. | 45,593 | 45,593 | 0 |
| Imputation and row exclusion | No imputation or row deletion is justified in this foundation phase; preserve missing and suspicious records for review and model-specific handling. | 0 | 45,593 | 0 |
| Flag rider_age_suspicious | Preserve the source value while making records requiring validation explicit. | 38 | 45,593 | 0 |
| Flag rating_out_of_range | Preserve the source value while making records requiring validation explicit. | 53 | 45,593 | 0 |
| Flag multiple_deliveries_invalid | Preserve the source value while making records requiring validation explicit. | 0 | 45,593 | 0 |
| Flag vehicle_condition_invalid | Preserve the source value while making records requiring validation explicit. | 0 | 45,593 | 0 |
| Flag delivery_time_invalid | Preserve the source value while making records requiring validation explicit. | 0 | 45,593 | 0 |
| Flag distances over 100 km | Review unusually long coordinate pairs; the threshold is a screening heuristic and no records are removed. | 431 | 45,593 | 0 |
| Flag IQR delivery_time outliers | Identify statistically unusual observations for review; do not automatically discard them. | 270 | 45,593 | 0 |
| Flag 1st/99th percentile delivery_time outliers | Identify statistically unusual observations for review; do not automatically discard them. | 436 | 45,593 | 0 |
| Flag IQR distance outliers | Identify statistically unusual observations for review; do not automatically discard them. | 0 | 45,593 | 0 |
| Flag 1st/99th percentile distance outliers | Identify statistically unusual observations for review; do not automatically discard them. | 817 | 45,593 | 0 |

## 9. Records retained and excluded

- Input: 45,593
- Retained: 45,593
- Excluded: 0
- Imputed values: 0

## Delay lens and leakage policy

- Target: `Time_taken(min)` in minutes.
- SLA: **30 minutes (provisional)**.
- Late means delivery time **strictly greater** than the SLA.
- In this dataset, 13,606 observed deliveries exceed this provisional threshold; this is a label count, not a model result.
- `Time_Order_picked`, the delivery outcome, identifiers, and assignment-dependent rider/vehicle fields are excluded from order-placement model inputs.
- Weather, traffic, festival, and coordinates require verified availability at prediction time.
- Distance, date, and time source columns are transformed; quality flags are audit-only.

| Field | Availability policy |
|---|---|
| `delivery_distance_km` | Eligible if store/customer coordinates are known at placement and pass geographic validation. |
| `order_hour` | Eligible if the order time is known at placement; missing/malformed values remain missing. |
| `order_day` | Eligible: derived from the order date. |
| `day_of_week` | Eligible: derived from the order date; Monday=0 through Sunday=6. |
| `is_weekend` | Eligible: derived from the order date. |
| `is_peak_hour` | Heuristic windows 11:00-13:59 and 17:00-21:59; confirm with operations. |
| `distance_band` | Eligible: derived from validated store-to-customer distance. |
| `weather_normalized` | Conditionally eligible: use only if weather is available at order placement. |
| `traffic_normalized` | Conditionally eligible: use only if traffic is available at order placement. |
| `Festival` | Conditionally eligible: use only if the schedule is known at order placement. |
| `City` | Eligible if known from the order destination at placement. |
| `Type_of_order` | Eligible if known from the submitted order at placement. |
| `ID` | Excluded: order identifier, not a predictive input. |
| `Delivery_person_ID` | Excluded: rider identifier; rider assignment timing is unknown. |
| `Delivery_person_Age` | Excluded: rider assignment timing is unknown. |
| `Delivery_person_Ratings` | Excluded: rider assignment timing is unknown. |
| `Vehicle_condition` | Excluded: vehicle assignment timing is unknown. |
| `Type_of_vehicle` | Excluded: vehicle assignment timing is unknown. |
| `multiple_deliveries` | Excluded: assignment/load timing is unknown. |
| `Time_Order_picked` | Excluded: known only after order placement. |
| `Time_taken(min)` | Excluded from inputs: delivery outcome target. |
| `is_late_order` | Excluded from inputs: label derived from the delivery outcome. |

## 10. Remaining limitations

- The supplied file is a food-delivery dataset used as a quick-commerce proxy, not verified grocery dark-store operations data.
- The 30-minute SLA is provisional and requires organizer/business confirmation.
- Rider ages below 18 are flagged for review; 18 is a screening assumption and no maximum age is imposed.
- Rider tenure/experience is not present and is not fabricated.
- Rider/vehicle assignment timing is unknown; rider-specific attributes and multiple-delivery load are excluded from order-placement features.
- Weather, traffic, and festival fields are eligible only if known when a prediction is made.
- The broad India coordinate envelope is a review heuristic; flagged records are retained and require source confirmation.
- No imputation or duplicate removal has been performed.
