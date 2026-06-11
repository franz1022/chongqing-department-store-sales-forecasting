# Business Recommendations

## Project Context

This project reproduces a summer retail sales forecasting workflow for a department-store-like business scenario. The analysis focuses on weekly sales patterns across stores and departments, with external features such as holidays, markdowns, store attributes, and economic indicators.

## Key Findings

1. **Summer sales increased by approximately 7.64% from 2022 to 2024.**
   This suggests stronger summer demand in the latest year and indicates that summer operation planning should receive more attention.

2. **July generated the highest total summer sales.**
   This implies that mid-summer is the most important period for promotion scheduling, inventory preparation, and staffing allocation.

3. **Holiday weeks had approximately 25.40% higher average weekly sales than non-holiday weeks.**
   Holiday periods should be treated as key sales windows, especially for high-demand departments and larger stores.

4. **Store Type A contributed the largest share of total summer sales.**
   Larger or stronger-performing store types should receive priority in inventory allocation and promotional resource planning.

5. **The East region contributed the highest total summer sales among all regions.**
   Regional demand differences should be considered when planning marketing campaigns and stock distribution.

6. **Top-performing departments were:
   - Department 20: total summer sales = 134,999,920
   - Department 19: total summer sales = 128,745,877
   - Department 18: total summer sales = 126,388,335
   - Department 17: total summer sales = 122,751,645
   - Department 16: total summer sales = 122,060,269

## Model Performance

- The best aggregate-level forecasting model was **Random Forest Aggregate**.
- Best model MAE: **737,077**.
- Best model MAPE: **1.48%**.
- Rolling 4-week average baseline MAPE: **3.87%**.
- Forecast error reduction over the rolling baseline: **61.74%**.

The result shows that machine learning models with engineered features can improve aggregate-level weekly sales forecasting accuracy compared with naive baselines and traditional ARIMA-style models.

## XGBoost Feature Importance Insights

The top XGBoost feature importance results suggest that the following variables were important sales drivers:

- `cat__store_type_A`
- `cat__holiday_name_Non-Holiday`
- `num__rolling_mean_8`
- `num__is_holiday`
- `cat__holiday_name_Christmas`
- `num__store_size`
- `cat__holiday_name_Black Friday`
- `num__month`
- `cat__store_type_B`
- `cat__season_Fall`

These features indicate that store type, holiday effects, historical rolling sales, store size, month, markdown intensity, and department-level differences were important drivers of weekly sales.

## Business Recommendations

### 1. Strengthen promotion planning around peak summer weeks
Since July produced the highest summer sales, marketing campaigns and discount events should be scheduled around this period to maximize demand capture.

### 2. Prepare inventory earlier for high-performing departments
Top-performing departments should receive earlier replenishment planning before expected peak weeks to reduce stock-out risk.

### 3. Allocate more staff during holiday and high-demand weeks
Holiday weeks and predicted high-sales weeks should receive additional cashier, sales assistant, and floor-management staffing.

### 4. Use store type and region to guide resource allocation
Store Type A and the East region showed stronger sales contribution, so they should receive priority in inventory and promotion budget allocation.

### 5. Use forecasting results as an operational early-warning tool
Weekly sales forecasts can help management identify upcoming peak demand periods and prepare promotion schedules, inventory allocation, and staffing plans in advance.

## Interview Talking Point

This project demonstrates an end-to-end retail forecasting workflow: data cleaning, multi-table merging, summer-specific EDA, baseline forecasting, ARIMA/SARIMA modeling, machine learning regression, feature importance analysis, and business recommendation generation.
