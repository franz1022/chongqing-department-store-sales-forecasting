from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


# ============================================================
# 0. 基础设置
# ============================================================

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]

PROCESSED_DIR = BASE_DIR / "data" / "processed"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
MODEL_DIR = BASE_DIR / "models"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PATH = PROCESSED_DIR / "retail_sales_cleaned_merged.csv"

BASELINE_ARIMA_PATH = OUTPUT_DIR / "model_comparison_baseline_arima.csv"


# ============================================================
# 1. 读取数据
# ============================================================

df = pd.read_csv(INPUT_PATH)
df["date"] = pd.to_datetime(df["date"])

print("\n========== Loaded Data ==========")
print(df.shape)
print(df["date"].min(), "to", df["date"].max())


# ============================================================
# 2. 构造基础特征
# ============================================================

# markdown 总强度
markdown_cols = [col for col in df.columns if col.startswith("markdown")]
df["total_markdown"] = df[markdown_cols].sum(axis=1)

# 是否有 markdown
# has_markdown is constant for all rows, so it is excluded from modelling.
# 确保排序正确
df = df.sort_values(["store_id", "department", "date"]).copy()


# ============================================================
# 3. 构造时间序列特征
# 重点：所有 rolling / lag 都必须 shift，避免 data leakage
# ============================================================

group_cols = ["store_id", "department"]

df["lag_1"] = df.groupby(group_cols)["weekly_sales"].shift(1)
df["lag_4"] = df.groupby(group_cols)["weekly_sales"].shift(4)
df["lag_8"] = df.groupby(group_cols)["weekly_sales"].shift(8)

# rolling features 必须在每个 store_id + department 内部分组计算
# shift(1) 是为了避免 data leakage：
# 预测本周时，只能使用本周之前的销售数据

df["rolling_mean_4"] = (
    df.groupby(group_cols)["weekly_sales"]
    .transform(lambda x: x.shift(1).rolling(window=4).mean())
)

df["rolling_mean_8"] = (
    df.groupby(group_cols)["weekly_sales"]
    .transform(lambda x: x.shift(1).rolling(window=8).mean())
)

df["rolling_std_4"] = (
    df.groupby(group_cols)["weekly_sales"]
    .transform(lambda x: x.shift(1).rolling(window=4).std())
)


# ============================================================
# 4. 删除因为 lag / rolling 产生的缺失行
# ============================================================

feature_na_cols = [
    "lag_1",
    "lag_4",
    "lag_8",
    "rolling_mean_4",
    "rolling_mean_8",
    "rolling_std_4",
]

before_drop = df.shape[0]
df_model = df.dropna(subset=feature_na_cols).copy()
after_drop = df_model.shape[0]

print("\n========== Feature Engineering ==========")
print("Rows before dropping lag NA:", before_drop)
print("Rows after dropping lag NA:", after_drop)


# ============================================================
# 5. 划分训练集和测试集
# 测试集：2024 年夏季 6、7、8 月
# 训练集：测试集之前所有数据
# ============================================================

test_start = pd.Timestamp("2024-06-01")
test_end = pd.Timestamp("2024-08-31")

train_df = df_model[df_model["date"] < test_start].copy()

test_df = df_model[
    (df_model["date"] >= test_start)
    & (df_model["date"] <= test_end)
].copy()

print("\n========== Train / Test Split ==========")
print("Train:", train_df["date"].min(), "to", train_df["date"].max(), train_df.shape)
print("Test:", test_df["date"].min(), "to", test_df["date"].max(), test_df.shape)


# ============================================================
# 6. 选择特征
# ============================================================

target_col = "weekly_sales"

numeric_features = [
    "year",
    "month",
    "week_of_year",
    "store_size",
    "temperature",
    "fuel_price",
    "cpi",
    "unemployment",
    "markdown_1",
    "markdown_2",
    "markdown_3",
    "markdown_4",
    "markdown_5",
    "total_markdown",
    "is_holiday",
    "lag_1",
    "lag_4",
    "lag_8",
    "rolling_mean_4",
    "rolling_mean_8",
    "rolling_std_4",
]

categorical_features = [
    "store_id",
    "department",
    "store_type",
    "region",
    "holiday_name",
    "season",
]

# 只保留实际存在的列，避免列名不一致报错
numeric_features = [col for col in numeric_features if col in df_model.columns]
categorical_features = [col for col in categorical_features if col in df_model.columns]

feature_cols = numeric_features + categorical_features

X_train = train_df[feature_cols]
y_train = train_df[target_col]

X_test = test_df[feature_cols]
y_test = test_df[target_col]

print("\n========== Features ==========")
print("Numeric features:", numeric_features)
print("Categorical features:", categorical_features)
print("X_train:", X_train.shape)
print("X_test:", X_test.shape)


# ============================================================
# 7. 预处理 Pipeline
# ============================================================

numeric_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
    ]
)

categorical_transformer = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ]
)

preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_transformer, numeric_features),
        ("cat", categorical_transformer, categorical_features),
    ]
)


# ============================================================
# 8. 评估函数
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
    """
    Weighted Absolute Percentage Error.

    Measures total absolute forecast error relative to total actual sales.
    Lower values are better.
    """
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
    """
    Signed forecast bias as a percentage of total actual sales.

    Positive: overall over-forecasting.
    Negative: overall under-forecasting.
    """
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
        "n_test_rows": len(y_true),
        "evaluation_level": "store_department_week",
    }


results = []
predictions = test_df[["date", "store_id", "department", "weekly_sales"]].copy()


# ============================================================
# 9. 模型 1：Linear Regression
# ============================================================

print("\n========== Training Linear Regression ==========")

linear_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("model", LinearRegression()),
    ]
)

linear_model.fit(X_train, y_train)

pred_linear = linear_model.predict(X_test)
predictions["pred_linear_regression"] = pred_linear

results.append(
    evaluate_model(y_test, pred_linear, "Linear Regression")
)

print("Linear Regression completed.")


# ============================================================
# 10. 模型 2：Random Forest
# 为了运行速度，参数设置得比较保守
# ============================================================

print("\n========== Training Random Forest ==========")

rf_model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        (
            "model",
            RandomForestRegressor(
                n_estimators=100,
                max_depth=12,
                min_samples_leaf=5,
                random_state=42,
                n_jobs=-1,
            ),
        ),
    ]
)

rf_model.fit(X_train, y_train)

pred_rf = rf_model.predict(X_test)
predictions["pred_random_forest"] = pred_rf

results.append(
    evaluate_model(y_test, pred_rf, "Random Forest Regressor")
)

print("Random Forest completed.")


# ============================================================
# 11. 模型 3：XGBoost
# ============================================================

xgb_model = None

if XGBOOST_AVAILABLE:
    print("\n========== Training XGBoost ==========")

    xgb_model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "model",
                XGBRegressor(
                    n_estimators=300,
                    max_depth=6,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    objective="reg:squarederror",
                    random_state=42,
                    n_jobs=-1,
                ),
            ),
        ]
    )

    xgb_model.fit(X_train, y_train)

    pred_xgb = xgb_model.predict(X_test)
    predictions["pred_xgboost"] = pred_xgb

    results.append(
        evaluate_model(y_test, pred_xgb, "XGBoost Regressor")
    )

    print("XGBoost completed.")

else:
    print("\nXGBoost is not installed. Skipping XGBoost model.")


# ============================================================
# 12. 保存模型评估结果
# ============================================================

ml_results_df = pd.DataFrame(results).sort_values("mape")

print("\n========== ML Model Comparison ==========")
print(ml_results_df)

ml_results_path = OUTPUT_DIR / "ml_model_comparison.csv"
ml_predictions_path = OUTPUT_DIR / "ml_forecast_2024_summer.csv"

ml_results_df.to_csv(ml_results_path, index=False)
predictions.to_csv(ml_predictions_path, index=False)

print("\nSaved ML outputs to:")
print(ml_results_path)
print(ml_predictions_path)


# ============================================================
# 13. Plot aggregate Actual vs ML predictions by date
# ============================================================

plot_df = predictions.groupby("date", as_index=False).agg(
    actual_total_sales=("weekly_sales", "sum"),
    pred_linear_total=("pred_linear_regression", "sum"),
    pred_rf_total=("pred_random_forest", "sum"),
)

if "pred_xgboost" in predictions.columns:
    plot_df["pred_xgb_total"] = (
        predictions.groupby("date")["pred_xgboost"].sum().values
    )

plt.figure(figsize=(12, 6))

plt.plot(
    plot_df["date"],
    plot_df["actual_total_sales"],
    marker="o",
    label="Actual"
)

plt.plot(
    plot_df["date"],
    plot_df["pred_linear_total"],
    marker="o",
    label="Linear Regression"
)

plt.plot(
    plot_df["date"],
    plot_df["pred_rf_total"],
    marker="o",
    label="Random Forest"
)

if "pred_xgb_total" in plot_df.columns:
    plt.plot(
        plot_df["date"],
        plot_df["pred_xgb_total"],
        marker="o",
        label="XGBoost"
    )

plt.title("2024 Summer Weekly Sales: Actual vs ML Forecasts")
plt.xlabel("Date")
plt.ylabel("Total Weekly Sales")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

figure_path = FIGURE_DIR / "12_ml_forecast_2024_summer.png"
plt.savefig(figure_path, dpi=150)
plt.close()

print("\nSaved ML forecast figure to:")
print(figure_path)


# ============================================================
# 14. XGBoost 特征重要性
# ============================================================

if xgb_model is not None:
    try:
        fitted_preprocessor = xgb_model.named_steps["preprocessor"]
        fitted_xgb = xgb_model.named_steps["model"]

        feature_names = fitted_preprocessor.get_feature_names_out()
        importances = fitted_xgb.feature_importances_

        feature_importance_df = pd.DataFrame({
            "feature": feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False)

        feature_importance_path = OUTPUT_DIR / "xgboost_feature_importance.csv"
        feature_importance_df.to_csv(feature_importance_path, index=False)

        top_features = feature_importance_df.head(15)

        plt.figure(figsize=(10, 6))
        plt.barh(top_features["feature"][::-1], top_features["importance"][::-1])
        plt.title("Top 15 XGBoost Feature Importances")
        plt.xlabel("Importance")
        plt.ylabel("Feature")
        plt.tight_layout()

        importance_figure_path = FIGURE_DIR / "13_xgboost_feature_importance.png"
        plt.savefig(importance_figure_path, dpi=150)
        plt.close()

        print("\nSaved XGBoost feature importance to:")
        print(feature_importance_path)
        print(importance_figure_path)

    except Exception as e:
        print("\nCould not save XGBoost feature importance.")
        print(e)

# ============================================================
# 15. 额外评估：把 ML 预测结果按 date 聚合后评估
# 这样才能和 ARIMA / baseline 的整体周销售额预测进行比较
# ============================================================

def evaluate_aggregate_model(data, actual_col, pred_col, model_name):
    actual = data[actual_col]
    predicted = data[pred_col]

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
        "n_test_weeks": data.shape[0],
        "evaluation_level": "company_week",
    }


aggregate_results = []

aggregate_results.append(
    evaluate_aggregate_model(
        plot_df,
        "actual_total_sales",
        "pred_linear_total",
        "Linear Regression Aggregate"
    )
)

aggregate_results.append(
    evaluate_aggregate_model(
        plot_df,
        "actual_total_sales",
        "pred_rf_total",
        "Random Forest Aggregate"
    )
)

if "pred_xgb_total" in plot_df.columns:
    aggregate_results.append(
        evaluate_aggregate_model(
            plot_df,
            "actual_total_sales",
            "pred_xgb_total",
            "XGBoost Aggregate"
        )
    )

aggregate_results_df = pd.DataFrame(aggregate_results).sort_values("mape")

print("\n========== Aggregate-Level ML Model Comparison ==========")
print(aggregate_results_df)

aggregate_results_path = OUTPUT_DIR / "ml_aggregate_model_comparison.csv"
aggregate_results_df.to_csv(aggregate_results_path, index=False)

print("\nSaved aggregate-level ML comparison to:")
print(aggregate_results_path)


# ============================================================
# 16. 生成真正可比较的最终模型表
# baseline / ARIMA / aggregate ML 都是整体每周销售额层面
# ============================================================

if BASELINE_ARIMA_PATH.exists():
    previous_results = pd.read_csv(BASELINE_ARIMA_PATH)

    final_comparable_results = pd.concat(
        [previous_results, aggregate_results_df],
        ignore_index=True,
        sort=False
    )

    final_comparable_results = final_comparable_results.sort_values("mape")

    final_comparable_path = OUTPUT_DIR / "final_comparable_model_comparison.csv"
    final_comparable_results.to_csv(final_comparable_path, index=False)

    print("\n========== Final Comparable Model Comparison ==========")
    print(final_comparable_results)

    print("\nSaved final comparable comparison to:")
    print(final_comparable_path)
print("\nML regression modeling completed successfully.")
