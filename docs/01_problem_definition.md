# Project Problem Definition

## 1. Business Objective

This project aims to forecast weekly summer sales for a department-store business to support:

- Company-level sales planning
- Summer promotion scheduling
- Overall inventory preparation
- Store staffing and resource allocation

The primary forecasting period is June to August 2024.

## 2. Data Granularity

Each row in the modelling dataset represents:

> One Store × One Department × One Week

The target variable is `Weekly_Sales`.

For example, one row may represent the weekly sales of Department 5 in Store 12 during a specific week.

## 3. Forecasting Scenario

The primary scenario is a rolling one-week-ahead forecast.

At the end of week `t`, actual sales data up to week `t` are assumed to be available. The model then predicts sales for week `t+1`.

After the actual result for week `t+1` becomes available, the forecasting process is updated to predict week `t+2`.

This differs from predicting the entire 14-week summer period from one fixed forecast origin.

## 4. Forecast Levels

The project evaluates forecasts at two levels.

### Company-Level Weekly Forecast

Store-department forecasts are aggregated into total weekly company sales.

This level supports:

- Total sales planning
- Company-wide staffing decisions
- Overall inventory and promotion planning

### Store-Department-Level Forecast

Each Store × Department combination is evaluated separately.

This level is more relevant to:

- Detailed replenishment
- Store-level operations
- Department-level inventory decisions

## 5. Evaluation Metrics

### Company-Level Metrics

- MAE
- RMSE
- MAPE
- WAPE
- Forecast Bias

### Store-Department-Level Metrics

- WAPE
- RMSE
- Forecast Bias
- Error by store
- Error by department

## 6. Current V1 Results

The current V1 workflow reports approximately:

- Random Forest aggregate MAPE: 1.48%
- XGBoost aggregate MAPE: 1.52%
- Rolling 4-week baseline aggregate MAPE: 3.87%

However, aggregate accuracy may hide offsetting errors between stores and departments.

Therefore, the current model is more suitable for company-level planning than precise store-department replenishment.

## 7. V1 Evaluation Limitation

The machine-learning features use lagged actual sales from previous test weeks. This represents a rolling one-week-ahead forecasting scenario.

The existing ARIMA evaluation predicts multiple future weeks from one fixed origin.

Therefore, the machine-learning and ARIMA models are not currently evaluated under exactly the same forecasting conditions.

## 8. V2 Project Goal

The V2 project will:

1. Define a consistent rolling forecasting scenario
2. Prevent temporal and target leakage
3. Compare models under the same information conditions
4. Report both aggregate and granular performance
5. Improve reproducibility and code structure
6. Clearly explain the business use and limitations of the model