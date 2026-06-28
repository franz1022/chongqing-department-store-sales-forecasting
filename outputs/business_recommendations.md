# Business Recommendations

## Project Context

This portfolio project uses a structured educational, synthetic, or adapted retail dataset in a hypothetical department-store business scenario. It does not claim to use confidential internal data from a real Chongqing retailer.

The operational forecasting scenario predicts one week ahead using calendar information, store and department attributes, planned markdown variables, and leakage-safe historical sales features.

## Key Business Findings

1. **Summer sales increased by approximately 7.64% from 2022 to 2024.**
   This indicates stronger summer demand in the latest year within the current dataset.

2. **July generated the highest total summer sales.**
   Promotion, inventory, and staffing plans should place additional attention on this peak summer period.

3. **Holiday weeks had approximately 25.40% higher average weekly sales than non-holiday weeks.**
   Holiday periods should be treated as high-demand planning windows, while the holiday definitions should be localised before real deployment.

4. **Store Type A contributed the largest share of total summer sales.**
   Higher-contribution store types may require greater inventory, staffing, and promotion capacity.

5. **The East region contributed the highest total summer sales among the available regional labels.**
   Regional demand differences should be considered in resource allocation, subject to validation on real local data.

6. **Top-performing departments were:**
   - Department 20: total summer sales = 134,999,920
   - Department 19: total summer sales = 128,745,877
   - Department 18: total summer sales = 126,388,335
   - Department 17: total summer sales = 122,751,645
   - Department 16: total summer sales = 122,060,269

## Final Model Performance

Final model selection is based on six expanding-window rolling-origin backtest folds rather than one favourable summer test period.

- Selected model: **Operational XGBoost**.
- Mean company-level MAPE: **1.82%**.
- Mean company-level WAPE: **1.91%**.
- Median company-level WAPE: **1.85%**.
- WAPE standard deviation: **0.42**.
- Best-fold WAPE: **1.39%**.
- Worst-fold WAPE: **2.58%**.
- Fold wins: **5 of 6**.
- Mean absolute company-level forecast bias: **0.66%**.
- Winning folds: **1, 2, 3, 4, 6**.
- Rolling four-week baseline mean WAPE: **10.56%**.
- Mean WAPE reduction relative to the rolling baseline: **81.94%**.

Operational XGBoost is preferred because it achieved the lowest mean WAPE, won five of six folds, and showed the strongest cross-season stability. Random Forest remains a useful alternative because it performed best in one summer-focused fold.

## XGBoost Model Interpretation

The existing XGBoost feature-importance output provides an initial view of model contribution. It should not be interpreted as causal evidence.

- `cat__store_type_A`
- `num__rolling_mean_8`
- `num__is_holiday`
- `cat__holiday_name_Non-Holiday`
- `cat__store_type_B`
- `num__store_size`
- `cat__holiday_name_Black Friday`
- `num__month`
- `cat__season_Fall`
- `num__rolling_mean_4`

The feature-importance results suggest that store type, holiday information, historical rolling sales, store size, month, and seasonal variables contributed to prediction.

## Operational Recommendations

### 1. Use the forecast as a weekly planning signal
Generate one-week-ahead forecasts on a fixed weekly schedule and review predicted peaks before inventory and staffing decisions are finalised.

### 2. Prioritise peak summer and holiday planning
Because July and holiday periods showed stronger sales, management should prepare promotions, staffing, and inventory earlier for those periods.

### 3. Apply differentiated resource allocation
Store Type A, the East region, and high-performing departments should receive targeted planning attention rather than uniform allocation.

### 4. Monitor WAPE and forecast bias together
Low WAPE measures forecast accuracy, while forecast bias helps detect persistent over- or under-planning. Both should be monitored after deployment.

### 5. Keep a baseline challenger
Continue comparing XGBoost with Random Forest and the rolling four-week baseline. A simpler model should replace the primary model if live performance deteriorates.

### 6. Do not use company-level accuracy as a granular claim
Company-level WAPE is much lower than store-department-level WAPE. Aggregate accuracy should not be presented as equally strong inventory-item or department accuracy.

## Limitations

- The dataset is not verified as real internal Chongqing retail data.
- Holiday labels and field units require local validation.
- Markdown variables are assumed to represent planned promotions known before the forecast week.
- Inventory, stockouts, competitor actions, and local events are not included.
- The model has not yet been tested on live future data.

## Interview Talking Point

This project demonstrates an end-to-end retail forecasting workflow covering business-key validation, safe multi-table merging, leakage-safe lag features, forecast-time feature auditing, baseline and statistical model comparison, and six-fold rolling-origin backtesting. Operational XGBoost achieved a mean company-level WAPE of 1.91%, won 5 of 6 folds, and reduced average WAPE by 81.94% relative to the rolling four-week baseline.