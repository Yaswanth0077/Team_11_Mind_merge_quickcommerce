# PS2 — Delivery-Time and Prediction Contract

## Status and scope

This deliverable documents the official PS2 requirements in
`../QUICK_COMMERCEDELIVERY_INTELLIGENCE.txt` and the implemented PS4/PS5
prediction contract. The guide requires a transparent delivery-time/target
definition, an explicitly stated late-order SLA, the distance method, and
post-order/post-delivery leakage exclusions; these are covered below.

The supplied source is food-delivery data used as a quick-commerce proxy. It
does not establish that its fields, service process, or SLA represent the
intended production operation.

## Delivery time and target

Delivery time is the dataset field `Time_taken(min)`, measured in minutes. It
is the PS4 regression target. PS5 derives the binary late-order outcome from
that observed target; the target and derived outcome are labels, never
prediction inputs.

## Late delivery and SLA

The current configured SLA is **30 minutes, provisional**. The guide does not
specify an organizer-approved threshold, and the project does not claim
approval. An order is labelled late only when
`Time_taken(min) > 30`; a value equal to 30 minutes is on time. Missing or
invalid delivery-time targets do not receive a late/on-time label.

The SLA constant is shared in `config.py` and is used by feature engineering,
PS5, the PS2 prediction demo, and the analytical reports. Confirm the
organizer's official SLA before using the label operationally.

## Store-to-customer distance

Distance is the great-circle distance between the restaurant/store and
customer latitude/longitude coordinates, calculated by the Haversine method
in `src/data/feature_engineering.py`. Latitude and longitude are converted
from degrees to radians. With angular differences `Δφ` and `Δλ`:

```text
a = sin²(Δφ / 2) + cos(φ₁) cos(φ₂) sin²(Δλ / 2)
d = 2 R atan2(√a, √(1 − a))
```

The configured Earth radius is 6,371 km. This is straight-line distance, not
road distance. The model pipeline masks missing, zero-pair, and coordinates
outside the configured broad India screening envelope rather than treating
them as real locations. The envelope is a data-quality screen, not proof that
a coordinate is operationally correct.

Distance is used only when both locations are supplied and valid at the
prediction decision time. That availability assumption is not verified by
the historical dataset. The prediction demo can also run without coordinates;
in that case the saved model's fitted preprocessing pipeline imputes the
missing distance feature and the interface warns the user.

## Prediction-time features and leakage exclusions

The shared PS4 regression and PS5 classification pipelines use the same
allowlisted inputs: validated store-to-customer distance, order hour, order
calendar fields, weekend/peak-hour indicators, distance band, destination
city, and order type. Raw prediction-time fields accepted by the inference
functions are order date/time, the two coordinate pairs, city, and order type.

The following are excluded because they are unavailable at order placement,
their availability is not established, or they contain outcome information:

| Excluded field(s) | Reason |
|---|---|
| `Time_Order_picked` (pickup time) | Observed after order placement. |
| `Time_taken(min)` (actual delivery time) | Regression target and post-delivery outcome. |
| `is_late_order` / `is_late` | Derived from the delivery-time target. |
| Rider ID, age, and rating | Rider assignment or rating timing is unknown; age is not verified experience. |
| Vehicle condition/type and multiple deliveries | Assignment/load timing is unknown. |
| Weather, traffic, and festival fields | The dataset does not establish that they are available at prediction time. |
| Order ID | Identifier, not a predictive input. |

The explicit feature allowlist and saved sklearn preprocessing pipelines are
shared by regression and classification. Pickup time is absent from both
feature definitions and their saved importance reports. The demo validates
its input keys against this allowlist; it never asks for actual delivery time,
pickup time, rider details, or the late label. It uses the existing fitted
PS4 and PS5 artifacts and does not retrain either model or change the saved
PS5 operating threshold of 0.15.

## Implemented evidence

- Shared feature and availability policy: `config.py`.
- Coordinate validation and Haversine calculation: `src/data/feature_engineering.py`.
- Regression and classification allowlist/inference: `src/models/regression.py` and `src/models/classification.py`.
- Validated paired inference: `src/models/order_prediction.py`.
- Interactive form: Streamlit page **New Order Prediction Demo**.
- Focused input and saved-artifact tests: `tests/test_order_prediction.py`.

The ETA, risk score, and status are retrospective model estimates, not
guarantees of future delivery performance.
