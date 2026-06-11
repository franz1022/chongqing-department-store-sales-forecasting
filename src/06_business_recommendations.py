from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0. 路径设置
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. 读取前面步骤生成的结果
# ============================================================

yearly_sales_path = OUTPUT_DIR / "yearly_summer_sales.csv"
monthly_sales_path = OUTPUT_DIR / "monthly_summer_sales.csv"
holiday_sales_path = OUTPUT_DIR / "holiday_sales_summary.csv"
store_type_sales_path = OUTPUT_DIR / "store_type_sales_summary.csv"
region_sales_path = OUTPUT_DIR / "region_sales_summary.csv"
department_sales_path = OUTPUT_DIR / "department_sales_summary.csv"
final_model_path = OUTPUT_DIR / "final_comparable_model_comparison.csv"
feature_importance_path = OUTPUT_DIR / "xgboost_feature_importance.csv"

yearly_sales = pd.read_csv(yearly_sales_path)
monthly_sales = pd.read_csv(monthly_sales_path)
holiday_sales = pd.read_csv(holiday_sales_path)
store_type_sales = pd.read_csv(store_type_sales_path)
region_sales = pd.read_csv(region_sales_path)
department_sales = pd.read_csv(department_sales_path)
final_model = pd.read_csv(final_model_path)
feature_importance = pd.read_csv(feature_importance_path)


# ============================================================
# 2. 计算关键业务指标
# ============================================================

# 2.1 年度夏季销售变化
yearly_sales = yearly_sales.sort_values("year")
first_year = int(yearly_sales.iloc[0]["year"])
last_year = int(yearly_sales.iloc[-1]["year"])

first_year_sales = yearly_sales.iloc[0]["total_summer_sales"]
last_year_sales = yearly_sales.iloc[-1]["total_summer_sales"]

summer_growth = (last_year_sales - first_year_sales) / first_year_sales * 100


# 2.2 最高销售月份
monthly_sales = monthly_sales.sort_values("total_sales", ascending=False)
best_month_name = monthly_sales.iloc[0]["month_name"]
best_month_sales = monthly_sales.iloc[0]["total_sales"]


# 2.3 节假日提升
holiday_lift = None

if "holiday_type" in holiday_sales.columns:
    holiday_pivot = holiday_sales.set_index("holiday_type")["avg_weekly_sales"]

    if "Holiday" in holiday_pivot.index and "Non-Holiday" in holiday_pivot.index:
        holiday_sales_avg = holiday_pivot["Holiday"]
        non_holiday_sales_avg = holiday_pivot["Non-Holiday"]
        holiday_lift = (holiday_sales_avg - non_holiday_sales_avg) / non_holiday_sales_avg * 100


# 2.4 门店类型贡献
store_type_sales = store_type_sales.sort_values("total_sales", ascending=False)
top_store_type = store_type_sales.iloc[0]["store_type"]
top_store_type_sales = store_type_sales.iloc[0]["total_sales"]


# 2.5 区域贡献
region_sales = region_sales.sort_values("total_sales", ascending=False)
top_region = region_sales.iloc[0]["region"]
top_region_sales = region_sales.iloc[0]["total_sales"]


# 2.6 Top departments
department_sales = department_sales.sort_values("total_sales", ascending=False)
top_departments = department_sales.head(5)


# 2.7 最佳模型
final_model = final_model.sort_values("mape")
best_model = final_model.iloc[0]["model"]
best_model_mape = final_model.iloc[0]["mape"]
best_model_mae = final_model.iloc[0]["mae"]

# 找 rolling baseline
rolling_baseline = final_model[
    final_model["model"].str.contains("Rolling 4-Week", case=False, na=False)
]

if not rolling_baseline.empty:
    rolling_mape = rolling_baseline.iloc[0]["mape"]
    model_improvement = (rolling_mape - best_model_mape) / rolling_mape * 100
else:
    rolling_mape = None
    model_improvement = None


# 2.8 XGBoost top features
top_features = feature_importance.head(10)


# ============================================================
# 3. 画最终模型比较图
# ============================================================

plot_model = final_model.copy()

# 为了图表清楚，只画可比较模型的 MAPE
plt.figure(figsize=(12, 6))
plt.bar(plot_model["model"], plot_model["mape"])
plt.title("Final Comparable Model Comparison by MAPE")
plt.xlabel("Model")
plt.ylabel("MAPE (%)")
plt.xticks(rotation=35, ha="right")
plt.tight_layout()

model_fig_path = FIGURE_DIR / "14_final_comparable_model_comparison.png"
plt.savefig(model_fig_path, dpi=150)
plt.close()


# ============================================================
# 4. 生成业务建议 Markdown
# ============================================================

recommendation_path = OUTPUT_DIR / "business_recommendations.md"

lines = []

lines.append("# Business Recommendations")
lines.append("")
lines.append("## Project Context")
lines.append("")
lines.append(
    "This project reproduces a summer retail sales forecasting workflow for a department-store-like business scenario. "
    "The analysis focuses on weekly sales patterns across stores and departments, with external features such as holidays, markdowns, store attributes, and economic indicators."
)
lines.append("")

lines.append("## Key Findings")
lines.append("")

lines.append(f"1. **Summer sales increased by approximately {summer_growth:.2f}% from {first_year} to {last_year}.**")
lines.append(
    "   This suggests stronger summer demand in the latest year and indicates that summer operation planning should receive more attention."
)
lines.append("")

lines.append(f"2. **{best_month_name} generated the highest total summer sales.**")
lines.append(
    "   This implies that mid-summer is the most important period for promotion scheduling, inventory preparation, and staffing allocation."
)
lines.append("")

if holiday_lift is not None:
    lines.append(f"3. **Holiday weeks had approximately {holiday_lift:.2f}% higher average weekly sales than non-holiday weeks.**")
    lines.append(
        "   Holiday periods should be treated as key sales windows, especially for high-demand departments and larger stores."
    )
    lines.append("")
else:
    lines.append("3. **Holiday impact could not be fully quantified from the available summary table.**")
    lines.append("")

lines.append(f"4. **Store Type {top_store_type} contributed the largest share of total summer sales.**")
lines.append(
    "   Larger or stronger-performing store types should receive priority in inventory allocation and promotional resource planning."
)
lines.append("")

lines.append(f"5. **The {top_region} region contributed the highest total summer sales among all regions.**")
lines.append(
    "   Regional demand differences should be considered when planning marketing campaigns and stock distribution."
)
lines.append("")

lines.append("6. **Top-performing departments were:")
for _, row in top_departments.iterrows():
    lines.append(f"   - Department {int(row['department'])}: total summer sales = {row['total_sales']:,.0f}")
lines.append("")

lines.append("## Model Performance")
lines.append("")

lines.append(f"- The best aggregate-level forecasting model was **{best_model}**.")
lines.append(f"- Best model MAE: **{best_model_mae:,.0f}**.")
lines.append(f"- Best model MAPE: **{best_model_mape:.2f}%**.")

if rolling_mape is not None:
    lines.append(f"- Rolling 4-week average baseline MAPE: **{rolling_mape:.2f}%**.")
    lines.append(f"- Forecast error reduction over the rolling baseline: **{model_improvement:.2f}%**.")

lines.append("")
lines.append(
    "The result shows that machine learning models with engineered features can improve aggregate-level weekly sales forecasting accuracy compared with naive baselines and traditional ARIMA-style models."
)
lines.append("")

lines.append("## XGBoost Feature Importance Insights")
lines.append("")
lines.append("The top XGBoost feature importance results suggest that the following variables were important sales drivers:")
lines.append("")

for _, row in top_features.iterrows():
    lines.append(f"- `{row['feature']}`")

lines.append("")
lines.append(
    "These features indicate that store type, holiday effects, historical rolling sales, store size, month, markdown intensity, and department-level differences were important drivers of weekly sales."
)
lines.append("")

lines.append("## Business Recommendations")
lines.append("")

lines.append("### 1. Strengthen promotion planning around peak summer weeks")
lines.append(
    f"Since {best_month_name} produced the highest summer sales, marketing campaigns and discount events should be scheduled around this period to maximize demand capture."
)
lines.append("")

lines.append("### 2. Prepare inventory earlier for high-performing departments")
lines.append(
    "Top-performing departments should receive earlier replenishment planning before expected peak weeks to reduce stock-out risk."
)
lines.append("")

lines.append("### 3. Allocate more staff during holiday and high-demand weeks")
lines.append(
    "Holiday weeks and predicted high-sales weeks should receive additional cashier, sales assistant, and floor-management staffing."
)
lines.append("")

lines.append("### 4. Use store type and region to guide resource allocation")
lines.append(
    f"Store Type {top_store_type} and the {top_region} region showed stronger sales contribution, so they should receive priority in inventory and promotion budget allocation."
)
lines.append("")

lines.append("### 5. Use forecasting results as an operational early-warning tool")
lines.append(
    "Weekly sales forecasts can help management identify upcoming peak demand periods and prepare promotion schedules, inventory allocation, and staffing plans in advance."
)
lines.append("")

lines.append("## Interview Talking Point")
lines.append("")
lines.append(
    "This project demonstrates an end-to-end retail forecasting workflow: data cleaning, multi-table merging, summer-specific EDA, baseline forecasting, ARIMA/SARIMA modeling, machine learning regression, feature importance analysis, and business recommendation generation."
)
lines.append("")

recommendation_path.write_text("\n".join(lines), encoding="utf-8")


# ============================================================
# 5. 保存关键业务指标表
# ============================================================

summary = pd.DataFrame([
    {"metric": "first_year", "value": first_year},
    {"metric": "last_year", "value": last_year},
    {"metric": "summer_sales_growth_pct", "value": round(summer_growth, 2)},
    {"metric": "best_summer_month", "value": best_month_name},
    {"metric": "top_store_type", "value": top_store_type},
    {"metric": "top_region", "value": top_region},
    {"metric": "best_model", "value": best_model},
    {"metric": "best_model_mape", "value": round(best_model_mape, 4)},
    {"metric": "rolling_baseline_mape", "value": round(rolling_mape, 4) if rolling_mape is not None else None},
    {"metric": "model_improvement_over_baseline_pct", "value": round(model_improvement, 2) if model_improvement is not None else None},
])

summary_path = OUTPUT_DIR / "business_summary_metrics.csv"
summary.to_csv(summary_path, index=False)


# ============================================================
# 6. 输出结果
# ============================================================

print("\n========== Business Summary ==========")
print(summary)

print("\nSaved business recommendations to:")
print(recommendation_path)

print("\nSaved business summary metrics to:")
print(summary_path)

print("\nSaved final model comparison figure to:")
print(model_fig_path)

print("\nBusiness recommendation generation completed successfully.")