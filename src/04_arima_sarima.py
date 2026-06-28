from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import mean_absolute_error, mean_squared_error
from statsmodels.tsa.statespace.sarimax import SARIMAX


# ============================================================
# 0. 基础设置
# ============================================================

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PROCESSED_DIR / "retail_sales_cleaned_merged.csv"
BASELINE_PATH = OUTPUT_DIR / "baseline_model_comparison.csv"


# ============================================================
# 1. 读取数据
# ============================================================

df = pd.read_csv(INPUT_PATH)
df["date"] = pd.to_datetime(df["date"])

print("\n========== Loaded Data ==========")
print(df.shape)
print(df["date"].min(), "to", df["date"].max())


# ============================================================
# 2. 聚合成整体 weekly sales time series
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

print("\n========== Weekly Sales ==========")
print(weekly_sales.head())
print(weekly_sales.tail())


# ============================================================
# 3. 划分训练集和测试集
# 测试集：2024 年夏季 6、7、8 月
# 训练集：测试集之前的所有数据
# ============================================================

test_start = pd.Timestamp("2024-06-01")
test_end = pd.Timestamp("2024-08-31")

train_df = weekly_sales[weekly_sales["date"] < test_start].copy()

test_df = weekly_sales[
    (weekly_sales["date"] >= test_start)
    & (weekly_sales["date"] <= test_end)
].copy()

print("\n========== Train / Test Split ==========")
print("Train date range:", train_df["date"].min(), "to", train_df["date"].max())
print("Test date range:", test_df["date"].min(), "to", test_df["date"].max())
print("Train rows:", train_df.shape[0])
print("Test rows:", test_df.shape[0])


# ============================================================
# 4. 评估函数
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


def evaluate_model(y_true, y_pred, model_name):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred) ** 0.5
    mape_value = mape(y_true, y_pred)
    wape_value = wape(y_true, y_pred)
    bias_value = forecast_bias(y_true, y_pred)

    return {
        "model": model_name,
        "mae": mae,
        "rmse": rmse,
        "mape": mape_value,
        "wape": wape_value,
        "forecast_bias": bias_value,
        "n_test_weeks": len(y_true),
        "evaluation_level": "company_week",
    }

results = []


# ============================================================
# 5. ARIMA 模型
# order=(1,1,1)
# 含义：
# p=1: 使用上一期信息
# d=1: 一阶差分处理趋势
# q=1: 使用上一期误差修正
# ============================================================

def rolling_one_step_forecast(
    fitted_result,
    actual_values,
    model_label,
):
    """
    Generate rolling one-week-ahead forecasts.

    Model parameters remain fixed after training, while the model state is
    updated with each newly observed actual weekly sales value.
    """
    predictions = []
    current_result = fitted_result
    total_steps = len(actual_values)

    for step_number, actual_value in enumerate(
        actual_values,
        start=1,
    ):
        next_forecast = current_result.forecast(steps=1)

        predicted_value = float(
            np.asarray(next_forecast, dtype=float)[0]
        )

        predictions.append(predicted_value)

        print(
            f"{model_label} rolling step "
            f"{step_number}/{total_steps} completed."
        )

        current_result = current_result.append(
            endog=np.asarray(
                [actual_value],
                dtype=float,
            ),
            refit=False,
        )

    return np.asarray(predictions, dtype=float)


train_values = train_df[
    "total_weekly_sales"
].to_numpy(dtype=float)

test_values = test_df[
    "total_weekly_sales"
].to_numpy(dtype=float)


# ============================================================
# 5. Rolling one-week-ahead ARIMA
# ============================================================

print(
    "\n========== Training Rolling ARIMA(1,1,1) =========="
)

arima_model = SARIMAX(
    train_values,
    order=(1, 1, 1),
    seasonal_order=(0, 0, 0, 0),
    enforce_stationarity=False,
    enforce_invertibility=False,
)

arima_fit = arima_model.fit(
    disp=False,
    maxiter=200,
)

arima_pred = rolling_one_step_forecast(
    fitted_result=arima_fit,
    actual_values=test_values,
    model_label="ARIMA",
)

test_df["pred_arima_111"] = arima_pred

results.append(
    evaluate_model(
        test_df["total_weekly_sales"],
        test_df["pred_arima_111"],
        "ARIMA(1,1,1) Rolling",
    )
)

print("Rolling ARIMA completed.")


# ============================================================
# 6. Rolling one-week-ahead SARIMA
# ============================================================

print(
    "\n========== Training Rolling "
    "SARIMA(1,1,1)(1,0,0,52) =========="
)

try:
    sarima_model = SARIMAX(
        train_values,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 0, 52),
        enforce_stationarity=False,
        enforce_invertibility=False,
    )

    sarima_fit = sarima_model.fit(
        disp=False,
        maxiter=200,
    )

    sarima_pred = rolling_one_step_forecast(
        fitted_result=sarima_fit,
        actual_values=test_values,
        model_label="SARIMA",
    )

    test_df["pred_sarima"] = sarima_pred

    results.append(
        evaluate_model(
            test_df["total_weekly_sales"],
            test_df["pred_sarima"],
            "SARIMA(1,1,1)(1,0,0,52) Rolling",
        )
    )

    print("Rolling SARIMA completed.")

except Exception as error:
    print("Rolling SARIMA failed. Error message:")
    print(error)
    test_df["pred_sarima"] = np.nan


# ============================================================
# 7. Save rolling ARIMA / SARIMA results
# ============================================================

arima_results_df = pd.DataFrame(results)

print("\n========== ARIMA / SARIMA Model Comparison ==========")
print(arima_results_df)

arima_results_path = OUTPUT_DIR / "arima_sarima_model_comparison.csv"
arima_forecast_path = OUTPUT_DIR / "arima_sarima_forecast_2024_summer.csv"

arima_results_df.to_csv(arima_results_path, index=False)
test_df.to_csv(arima_forecast_path, index=False)

print("\nSaved ARIMA outputs to:")
print(arima_results_path)
print(arima_forecast_path)


# ============================================================
# 8. 和 baseline 结果合并
# ============================================================

if BASELINE_PATH.exists():
    baseline_results = pd.read_csv(BASELINE_PATH)

    combined_results = pd.concat(
        [baseline_results, arima_results_df],
        ignore_index=True
    )

    combined_results = combined_results.sort_values("mape")

    combined_path = OUTPUT_DIR / "model_comparison_baseline_arima.csv"
    combined_results.to_csv(combined_path, index=False)

    print("\n========== Combined Model Comparison ==========")
    print(combined_results)

    print("\nSaved combined comparison to:")
    print(combined_path)


# ============================================================
# 9. 画图：Actual vs ARIMA / SARIMA
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
    test_df["pred_arima_111"],
    marker="o",
    label="ARIMA(1,1,1)"
)

if "pred_sarima" in test_df.columns and test_df["pred_sarima"].notna().any():
    plt.plot(
        test_df["date"],
        test_df["pred_sarima"],
        marker="o",
        label="SARIMA(1,1,1)(1,0,0,52)"
    )

plt.title("2024 Summer Weekly Sales: Actual vs ARIMA/SARIMA Forecasts")
plt.xlabel("Date")
plt.ylabel("Total Weekly Sales")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

figure_path = FIGURE_DIR / "11_arima_sarima_forecast_2024_summer.png"
plt.savefig(figure_path, dpi=150)
plt.close()

print("\nSaved figure to:")
print(figure_path)

print("\nARIMA / SARIMA forecasting completed successfully.")
