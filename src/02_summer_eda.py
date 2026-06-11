from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# 0. 路径设置
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PROCESSED_DIR / "retail_sales_cleaned_merged.csv"


# ============================================================
# 1. 读取清洗后的数据
# ============================================================

df = pd.read_csv(INPUT_PATH)
df["date"] = pd.to_datetime(df["date"])

print("\n========== Loaded Cleaned Data ==========")
print("Shape:", df.shape)
print("Date range:", df["date"].min(), "to", df["date"].max())
print(df.head())


# ============================================================
# 2. 生成促销相关字段
# ============================================================

markdown_cols = [col for col in df.columns if col.startswith("markdown")]

df["total_markdown"] = df[markdown_cols].sum(axis=1)
df["has_markdown"] = (df["total_markdown"] > 0).astype(int)


# ============================================================
# 3. 筛选夏季数据：6、7、8 月
# ============================================================

summer_df = df[df["month"].isin([6, 7, 8])].copy()

print("\n========== Summer Data ==========")
print("Shape:", summer_df.shape)
print("Date range:", summer_df["date"].min(), "to", summer_df["date"].max())
print("Years:", sorted(summer_df["year"].unique()))
print("Months:", sorted(summer_df["month"].unique()))
print("Stores:", summer_df["store_id"].nunique())
print("Departments:", summer_df["department"].nunique())


# ============================================================
# 4. 保存夏季数据
# ============================================================

summer_output_path = PROCESSED_DIR / "summer_sales.csv"
summer_df.to_csv(summer_output_path, index=False)

print(f"\nSummer sales data saved to: {summer_output_path}")


# ============================================================
# 5. 每年夏季总销售额
# ============================================================

yearly_sales = (
    summer_df
    .groupby("year", as_index=False)["weekly_sales"]
    .sum()
    .rename(columns={"weekly_sales": "total_summer_sales"})
)

print("\n========== Yearly Summer Sales ==========")
print(yearly_sales)

plt.figure(figsize=(10, 5))
plt.plot(yearly_sales["year"], yearly_sales["total_summer_sales"], marker="o")
plt.title("Total Summer Sales by Year")
plt.xlabel("Year")
plt.ylabel("Total Summer Sales")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "01_total_summer_sales_by_year.png", dpi=150)
plt.close()


# ============================================================
# 6. 6、7、8 月销售对比
# ============================================================

monthly_sales = (
    summer_df
    .groupby("month", as_index=False)["weekly_sales"]
    .sum()
    .rename(columns={"weekly_sales": "total_sales"})
)

month_map = {
    6: "June",
    7: "July",
    8: "August"
}

monthly_sales["month_name"] = monthly_sales["month"].map(month_map)

print("\n========== Summer Sales by Month ==========")
print(monthly_sales)

plt.figure(figsize=(8, 5))
plt.bar(monthly_sales["month_name"], monthly_sales["total_sales"])
plt.title("Sales Comparison: June vs July vs August")
plt.xlabel("Month")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "02_sales_by_summer_month.png", dpi=150)
plt.close()


# ============================================================
# 7. 每周销售趋势
# ============================================================

weekly_trend = (
    summer_df
    .groupby("date", as_index=False)["weekly_sales"]
    .sum()
    .rename(columns={"weekly_sales": "total_weekly_sales"})
    .sort_values("date")
)

print("\n========== Weekly Summer Sales Trend ==========")
print(weekly_trend.head())

plt.figure(figsize=(12, 5))
plt.plot(weekly_trend["date"], weekly_trend["total_weekly_sales"])
plt.title("Weekly Summer Sales Trend")
plt.xlabel("Date")
plt.ylabel("Total Weekly Sales")
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(FIGURE_DIR / "03_weekly_summer_sales_trend.png", dpi=150)
plt.close()


# ============================================================
# 8. 节假日 vs 非节假日
# ============================================================

holiday_sales = (
    summer_df
    .groupby("is_holiday", as_index=False)["weekly_sales"]
    .mean()
    .rename(columns={"weekly_sales": "avg_weekly_sales"})
)

holiday_sales["holiday_type"] = holiday_sales["is_holiday"].map({
    0: "Non-Holiday",
    1: "Holiday",
    False: "Non-Holiday",
    True: "Holiday"
})

print("\n========== Holiday vs Non-Holiday ==========")
print(holiday_sales)

plt.figure(figsize=(8, 5))
plt.bar(holiday_sales["holiday_type"], holiday_sales["avg_weekly_sales"])
plt.title("Average Weekly Sales: Holiday vs Non-Holiday")
plt.xlabel("Holiday Type")
plt.ylabel("Average Weekly Sales")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "04_holiday_vs_non_holiday_sales.png", dpi=150)
plt.close()


# ============================================================
# 9. 促销 markdown vs 非促销
# ============================================================

markdown_sales = (
    summer_df
    .groupby("has_markdown", as_index=False)["weekly_sales"]
    .mean()
    .rename(columns={"weekly_sales": "avg_weekly_sales"})
)

markdown_sales["markdown_type"] = markdown_sales["has_markdown"].map({
    0: "No Markdown",
    1: "With Markdown"
})

print("\n========== Markdown vs No Markdown ==========")
print(markdown_sales)

plt.figure(figsize=(8, 5))
plt.bar(markdown_sales["markdown_type"], markdown_sales["avg_weekly_sales"])
plt.title("Average Weekly Sales: Markdown vs No Markdown")
plt.xlabel("Markdown Type")
plt.ylabel("Average Weekly Sales")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "05_markdown_vs_no_markdown_sales.png", dpi=150)
plt.close()


# ============================================================
# 10. 门店类型销售表现
# ============================================================

store_type_sales = (
    summer_df
    .groupby("store_type", as_index=False)["weekly_sales"]
    .sum()
    .rename(columns={"weekly_sales": "total_sales"})
    .sort_values("total_sales", ascending=False)
)

print("\n========== Sales by Store Type ==========")
print(store_type_sales)

plt.figure(figsize=(8, 5))
plt.bar(store_type_sales["store_type"], store_type_sales["total_sales"])
plt.title("Summer Sales by Store Type")
plt.xlabel("Store Type")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "06_sales_by_store_type.png", dpi=150)
plt.close()


# ============================================================
# 11. 区域销售表现
# ============================================================

region_sales = (
    summer_df
    .groupby("region", as_index=False)["weekly_sales"]
    .sum()
    .rename(columns={"weekly_sales": "total_sales"})
    .sort_values("total_sales", ascending=False)
)

print("\n========== Sales by Region ==========")
print(region_sales)

plt.figure(figsize=(8, 5))
plt.bar(region_sales["region"], region_sales["total_sales"])
plt.title("Summer Sales by Region")
plt.xlabel("Region")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "07_sales_by_region.png", dpi=150)
plt.close()


# ============================================================
# 12. 部门销售排名
# ============================================================

department_sales = (
    summer_df
    .groupby("department", as_index=False)["weekly_sales"]
    .sum()
    .rename(columns={"weekly_sales": "total_sales"})
    .sort_values("total_sales", ascending=False)
)

top10_departments = department_sales.head(10)

print("\n========== Top 10 Departments by Summer Sales ==========")
print(top10_departments)

plt.figure(figsize=(10, 5))
plt.bar(top10_departments["department"].astype(str), top10_departments["total_sales"])
plt.title("Top 10 Departments by Summer Sales")
plt.xlabel("Department")
plt.ylabel("Total Sales")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "08_top10_departments_by_summer_sales.png", dpi=150)
plt.close()


# ============================================================
# 13. 温度与销售关系
# ============================================================

temperature_sales = (
    summer_df
    .groupby("temperature", as_index=False)["weekly_sales"]
    .mean()
    .rename(columns={"weekly_sales": "avg_weekly_sales"})
)

plt.figure(figsize=(8, 5))
plt.scatter(summer_df["temperature"], summer_df["weekly_sales"], alpha=0.2)
plt.title("Relationship Between Temperature and Weekly Sales")
plt.xlabel("Temperature")
plt.ylabel("Weekly Sales")
plt.tight_layout()
plt.savefig(FIGURE_DIR / "09_temperature_vs_sales.png", dpi=150)
plt.close()


# ============================================================
# 14. 保存汇总结果
# ============================================================

yearly_sales.to_csv(OUTPUT_DIR / "yearly_summer_sales.csv", index=False)
monthly_sales.to_csv(OUTPUT_DIR / "monthly_summer_sales.csv", index=False)
weekly_trend.to_csv(OUTPUT_DIR / "weekly_summer_sales_trend.csv", index=False)
holiday_sales.to_csv(OUTPUT_DIR / "holiday_sales_summary.csv", index=False)
markdown_sales.to_csv(OUTPUT_DIR / "markdown_sales_summary.csv", index=False)
store_type_sales.to_csv(OUTPUT_DIR / "store_type_sales_summary.csv", index=False)
region_sales.to_csv(OUTPUT_DIR / "region_sales_summary.csv", index=False)
department_sales.to_csv(OUTPUT_DIR / "department_sales_summary.csv", index=False)

print("\n========== Saved Summary Tables ==========")
print(OUTPUT_DIR / "yearly_summer_sales.csv")
print(OUTPUT_DIR / "monthly_summer_sales.csv")
print(OUTPUT_DIR / "weekly_summer_sales_trend.csv")
print(OUTPUT_DIR / "holiday_sales_summary.csv")
print(OUTPUT_DIR / "markdown_sales_summary.csv")
print(OUTPUT_DIR / "store_type_sales_summary.csv")
print(OUTPUT_DIR / "region_sales_summary.csv")
print(OUTPUT_DIR / "department_sales_summary.csv")

print("\n========== Saved Figures ==========")
for fig in sorted(FIGURE_DIR.glob("*.png")):
    print(fig)

print("\nSummer EDA completed successfully.")