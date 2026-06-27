from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error


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
# 1. 读取清洗后的完整数据
# 注意：baseline 用完整周度数据，不只用夏季数据
# 这样 rolling average 和 previous-year baseline 更合理
# ============================================================

df = pd.read_csv(INPUT_PATH)
df["date"] = pd.to_datetime(df["date"])

print("\n========== Loaded Data ==========")
print(df.shape)
print(df["date"].min(), "to", df["date"].max())


# ============================================================
# 2. 聚合成总周销售额时间序列
# 这里先做公司/百货整体层面的 sales forecasting
# 后面再做 store / department 层面的 ML 模型
# ============================================================

weekly_sales = (
    df
    .groupby("date", as_index=False)["weekly_sales"]
    .sum()
    .rename(columns={"weekly_sales": "total_weekly_sales"})
    .sort_values("date")
)

weekly_sales["year"] = weekly_sales["date"].dt.year
weekly_sales["month"] = weekly_sales["date"].dt.month
weekly_sales["week_of_year"] = weekly_sales["date"].dt.isocalendar().week.astype(int)
weekly_sales["is_summer"] = weekly_sales["month"].isin([6, 7, 8]).astype(int)

print("\n========== Weekly Sales ==========")
print(weekly_sales.head())
print(weekly_sales.tail())


# ============================================================
# 3. 构造 baseline predictions
# ============================================================

# Baseline 1: 上一周销售额
weekly_sales["pred_last_week"] = weekly_sales["total_weekly_sales"].shift(1)

# Baseline 2: 过去 4 周平均销售额
weekly_sales["pred_rolling_4w"] = (
    weekly_sales["total_weekly_sales"]
    .shift(1)
    .rolling(window=4)
    .mean()
)

# Baseline 3: 去年同一周销售额
# 由于数据是 weekly，约等于 shift(52)
weekly_sales["pred_prev_year_same_week"] = weekly_sales["total_weekly_sales"].shift(52)


# ============================================================
# 4. 选择测试集：2024 年夏季 6、7、8 月
# ============================================================

test_df = weekly_sales[
    (weekly_sales["year"] == 2024)
    & (weekly_sales["month"].isin([6, 7, 8]))
].copy()

print("\n========== Test Data: 2024 Summer ==========")
print(test_df[["date", "total_weekly_sales", "pred_last_week", "pred_rolling_4w", "pred_prev_year_same_week"]].head())
print("Test rows:", test_df.shape[0])


# ============================================================
# 5. 评估函数
# ============================================================

def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true != 0

    if not np.any(mask):
        return np.nan

    return (
        np.mean(
            np.abs(
                (y_true[mask] - y_pred[mask])
                / y_true[mask]
            )
        )
        * 100
    )


def wape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        return np.nan

    return (
        np.sum(np.abs(y_true - y_pred))
        / denominator
        * 100
    )


def forecast_bias(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    denominator = np.sum(np.abs(y_true))

    if denominator == 0:
        return np.nan

    return (
        np.sum(y_pred - y_true)
        / denominator
        * 100
    )


def evaluate_baseline(data, pred_col, model_name):
    valid = data.dropna(subset=[pred_col]).copy()

    actual = valid["total_weekly_sales"]
    predicted = valid[pred_col]

    mae = mean_absolute_error(actual, predicted)
    rmse = mean_squared_error(actual, predicted) ** 0.5
    mape_value = mape(actual, predicted)
    wape_value = wape(actual, predicted)
    bias_value = forecast_bias(actual, predicted)

    return {
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "mape": mape_value,
        "wape": wape_value,
        "forecast_bias": bias_value,
        "n_test_weeks": valid.shape[0],
        "evaluation_level": "company_week",
    }

results = []

results.append(
    evaluate_baseline(test_df, "pred_last_week", "Last Week Baseline")
)

results.append(
    evaluate_baseline(test_df, "pred_rolling_4w", "Rolling 4-Week Average Baseline")
)

results.append(
    evaluate_baseline(test_df, "pred_prev_year_same_week", "Previous Year Same Week Baseline")
)

results_df = pd.DataFrame(results)

print("\n========== Baseline Model Comparison ==========")
print(results_df)


# ============================================================
# 6. 保存结果
# ============================================================

results_path = OUTPUT_DIR / "baseline_model_comparison.csv"
results_df.to_csv(results_path, index=False)

forecast_path = OUTPUT_DIR / "baseline_forecast_2024_summer.csv"
test_df.to_csv(forecast_path, index=False)

print("\nSaved baseline results to:")
print(results_path)
print(forecast_path)


# ============================================================
# 7. 画图：Actual vs Baseline Predictions
# ============================================================

plt.figure(figsize=(12, 6))

plt.plot(
    test_df["date"],
    test_df["total_weekly_sales"],
    marker="o",
    label="Actual"
)

plt.plot(
    test_df["date"],
    test_df["pred_last_week"],
    marker="o",
    label="Last Week"
)

plt.plot(
    test_df["date"],
    test_df["pred_rolling_4w"],
    marker="o",
    label="Rolling 4-Week Average"
)

plt.plot(
    test_df["date"],
    test_df["pred_prev_year_same_week"],
    marker="o",
    label="Previous Year Same Week"
)

plt.title("2024 Summer Weekly Sales: Actual vs Baseline Forecasts")
plt.xlabel("Date")
plt.ylabel("Total Weekly Sales")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

figure_path = FIGURE_DIR / "10_baseline_forecast_2024_summer.png"
plt.savefig(figure_path, dpi=150)
plt.close()

print("\nSaved figure to:")
print(figure_path)

print("\nBaseline forecasting completed successfully.")
