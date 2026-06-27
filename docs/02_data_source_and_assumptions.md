# Data Source and Project Assumptions

## 1. Dataset Positioning

This project uses a structured retail dataset for forecasting methodology development and portfolio demonstration.

The dataset should not be presented as verified internal data from a real Chongqing department-store company. Based on its structure and fields, it appears to be a synthetic, educational, or adapted retail dataset rather than proprietary business data.

The Chongqing department-store context is therefore used as a business scenario for model design, evaluation, and communication.

## 2. Data Structure

The project uses three source tables:

- Sales data
- Store-level feature data
- Store metadata

After validation and merging, the modelling dataset contains:

- 50 stores
- 20 departments per store
- 156 weekly dates
- 156,000 store-department-week observations
- Date range from 2022-01-01 to 2024-12-21

The panel is balanced:

- Every store contains the same number of departments
- Every store-department series contains the same number of weekly observations
- The business key is `store_id + department + date`

## 3. Indicators That the Dataset Is Not Verified Chongqing Business Data

The source fields include characteristics commonly found in educational retail forecasting datasets, including:

- US-style holiday labels
- Fuel-price indicators
- CPI
- Unemployment
- Multiple markdown variables
- Store type and store size classifications

These fields do not, by themselves, verify that the dataset represents a real Chongqing retailer.

Therefore, the project does not claim:

- That the data came from a real Chongqing department store
- That the stores represent actual Chongqing locations
- That the holiday definitions match local Chongqing retail calendars
- That the economic indicators are official local indicators
- That the reported sales values are actual company revenue

## 4. Business Scenario

The project assumes a hypothetical department-store group that wants to forecast weekly sales for planning purposes.

The forecasting outputs may support:

- Company-level weekly sales planning
- Seasonal sales monitoring
- Promotion planning
- Staffing and inventory discussions
- Store and department performance analysis

The current strongest results apply to company-level weekly sales after aggregating store-department predictions.

They should not be interpreted as equally accurate forecasts for every individual store and department.

## 5. Unit and Field Assumptions

### Weekly Sales

`weekly_sales` is treated as a continuous sales-value target.

The currency and accounting definition are not verified from the source data. Results should therefore be interpreted as relative forecasting performance rather than audited financial reporting.

### Temperature

The dataset contains a `temperature` field, but its measurement unit is not independently verified in this project.

The project does not assume Celsius or Fahrenheit in business conclusions.

### Fuel Price, CPI, and Unemployment

These variables are treated as external contextual indicators.

Because actual forecast-week values may not be available when a prediction is made, the operational model excludes:

- `temperature`
- `fuel_price`
- `cpi`
- `unemployment`

A separate feature-availability sensitivity analysis compares the full-information and operational scenarios.

### Markdown Variables

The following fields are treated as planned promotion information:

- `markdown_1`
- `markdown_2`
- `markdown_3`
- `markdown_4`
- `markdown_5`
- `total_markdown`

This treatment assumes that promotion plans are known before the forecast week.

If the markdown fields represent realised rather than planned activity, they would require additional lagging or removal.

### Holiday Variables

Holiday fields are treated as calendar features known in advance.

However, the holiday names appear to reflect a non-Chongqing calendar. They are retained for forecasting methodology testing but should be replaced with local holiday definitions in a real deployment.

## 6. Forecasting Assumptions

The main operational scenario is rolling one-week-ahead forecasting.

At the end of week `t`, the model predicts week `t+1`.

Historical sales features include:

- `lag_1`
- `lag_4`
- `lag_8`
- `rolling_mean_4`
- `rolling_mean_8`
- `rolling_std_4`

All rolling features are calculated after `shift(1)`, so the target week is excluded from its own feature construction.

## 7. Evaluation Design

The project uses:

- Store-department-week evaluation for granular performance
- Company-week evaluation for aggregate planning performance
- MAE
- RMSE
- MAPE
- WAPE
- Forecast bias
- Six expanding-window rolling-origin backtest folds

Each backtest fold contains 14 test weeks.

The operational XGBoost model is selected based on cross-fold stability rather than a single favourable test period.

## 8. Current Model Selection

The current primary model is:

**Operational Conservative XGBoost**

Across six rolling-origin folds, it achieved:

- Mean company-level WAPE of approximately 1.91%
- Median company-level WAPE of approximately 1.85%
- Maximum fold WAPE of approximately 2.58%
- Five wins across six backtest folds
- Mean absolute company-level forecast bias of approximately 0.66%

These results describe the current dataset and backtest design only. They do not guarantee the same performance on real company data.

## 9. Limitations

The main limitations are:

- The source is not verified as real Chongqing business data
- The history covers about three years
- Holiday definitions may not match the intended market
- Some field units and business definitions are not independently verified
- Promotion variables are assumed to be known in advance
- The model has not been tested on live future data
- Company-level accuracy is much stronger than store-department-level accuracy
- Hyperparameter tuning is limited
- No inventory, stockout, competitor, or local event data are included

## 10. Real-World Deployment Requirements

Before deployment in a real retailer, the following would be required:

- Replace the current source with verified internal data
- Confirm all field definitions and units
- Use local holidays and events
- Validate promotion-data availability
- Add inventory and stockout information
- Add product and category hierarchy
- Define forecast refresh frequency
- Monitor forecast drift and bias
- Retrain models on a scheduled basis
- Establish business thresholds for acceptable error

## 11. Disclosure Statement

This repository is a portfolio and methodology project.

It demonstrates:

- Data validation
- Leakage-safe feature engineering
- Forecast-time feature auditing
- Baseline and machine-learning comparison
- Rolling-origin backtesting
- Business interpretation

It does not claim access to confidential internal data from a real Chongqing retailer.
