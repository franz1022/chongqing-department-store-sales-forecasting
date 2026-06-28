# Feature Availability and Forecast-Time Assumptions

## 1. Purpose

A forecasting feature should only be used when its value is available at the time the prediction is generated.

Using a value that becomes available only after the forecast week would create feature-availability leakage and could make the reported performance unrealistically optimistic.

The primary project scenario is rolling one-week-ahead forecasting.

At the end of week `t`, the model predicts week `t+1`.

## 2. Feature Availability Classification

### A. Known in Advance

These variables are normally available before the forecast week:

* `year`
* `month`
* `week_of_year`
* `store_id`
* `department`
* `store_type`
* `store_size`
* `region`
* `season`
* `is_holiday`
* `holiday_name`

Calendar information, store attributes and planned holidays are known before the forecast is generated.

### B. Historical Sales Features

These variables use only sales observed before the forecast week:

* `lag_1`
* `lag_4`
* `lag_8`
* `rolling_mean_4`
* `rolling_mean_8`
* `rolling_std_4`

All rolling features are calculated after applying `shift(1)`, so the current week's target is excluded.

These variables are valid under the rolling one-week-ahead scenario.

### C. Promotion Features

The following features are treated as known in advance only when markdown activity represents a planned promotion:

* `markdown_1`
* `markdown_2`
* `markdown_3`
* `markdown_4`
* `markdown_5`
* `total_markdown`

This assumption must be stated explicitly because actual realised promotion expenditure may not be fully known before the forecast week.

### D. Potentially Unavailable Current-Week Features

The following variables may not be available as final observed values when the forecast is generated:

* `temperature`
* `fuel_price`
* `cpi`
* `unemployment`

Potential issues include:

* Actual temperature becomes known during or after the forecast week
* CPI and unemployment values may be published with a delay
* Fuel-price observations may not be known for the complete future week

Using their actual forecast-week values may therefore introduce feature-availability leakage.

## 3. Evaluation Scenarios

Two modelling scenarios should be reported.

### Full-Information Benchmark

Uses the current complete feature set, including contemporaneous economic and environmental variables.

This scenario measures predictive potential but may rely on information that is not fully available before the forecast week.

### Operational Conservative Scenario

Uses only:

* Calendar variables
* Store and department attributes
* Holiday information
* Historical sales lag and rolling features
* Planned promotion features, under an explicit advance-availability assumption

The following contemporaneous variables are excluded:

* Temperature
* Fuel price
* CPI
* Unemployment

This scenario provides a more realistic estimate of performance at forecast time.

## 4. Interpretation Rule

If the full-information and operational models have similar performance, the model is not strongly dependent on uncertain future information.

If performance decreases substantially after removing contemporaneous external features, the original result may rely heavily on forecast-time unavailable information.

## 5. Current Methodological Status

The project has already addressed:

* Business-key validation
* Safe many-to-one merging
* Cleaning reconciliation
* Target-lag leakage
* Rolling-feature leakage
* Evaluation-level consistency
* Rolling one-week-ahead ARIMA evaluation

Feature-availability sensitivity analysis is the next methodological validation step.
