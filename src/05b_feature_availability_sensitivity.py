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
# 0. Settings
# ============================================================

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = (
    BASE_DIR
    / "data"
    / "processed"
    / "retail_sales_cleaned_merged.csv"
)

OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# 1. Load data
# ============================================================

df = pd.read_csv(INPUT_PATH)
df["date"] = pd.to_datetime(df["date"])

required_columns = {
    "store_id",
    "department",
    "date",
    "weekly_sales",
}

missing_required = required_columns.difference(df.columns)

if missing_required:
    raise ValueError(
        f"Missing required columns: {sorted(missing_required)}"
    )

df = df.sort_values(
    ["store_id", "department", "date"]
).reset_index(drop=True)

print("\n========== Loaded Data ==========")
print("Shape:", df.shape)
print(
    "Date range:",
    df["date"].min(),
    "to",
    df["date"].max(),
)


# ============================================================
# 2. Ensure calendar and derived variables exist
# ============================================================

df["year"] = df["date"].dt.year
df["month"] = df["date"].dt.month
df["week_of_year"] = (
    df["date"]
    .dt
    .isocalendar()
    .week
    .astype(int)
)

if "season" not in df.columns:
    season_map = {
        12: "Winter",
        1: "Winter",
        2: "Winter",
        3: "Spring",
        4: "Spring",
        5: "Spring",
        6: "Summer",
        7: "Summer",
        8: "Summer",
        9: "Autumn",
        10: "Autumn",
        11: "Autumn",
    }

    df["season"] = df["month"].map(season_map)

markdown_columns = [
    "markdown_1",
    "markdown_2",
    "markdown_3",
    "markdown_4",
    "markdown_5",
]

available_markdown_columns = [
    column
    for column in markdown_columns
    if column in df.columns
]

if (
    "total_markdown" not in df.columns
    and available_markdown_columns
):
    df["total_markdown"] = (
        df[available_markdown_columns]
        .fillna(0)
        .sum(axis=1)
    )


# ============================================================
# 3. Leakage-safe sales features
# ============================================================

group_columns = [
    "store_id",
    "department",
]

grouped_sales = df.groupby(
    group_columns,
    sort=False,
)["weekly_sales"]

df["lag_1"] = grouped_sales.shift(1)
df["lag_4"] = grouped_sales.shift(4)
df["lag_8"] = grouped_sales.shift(8)

df["rolling_mean_4"] = grouped_sales.transform(
    lambda series:
        series.shift(1)
        .rolling(
            window=4,
            min_periods=4,
        )
        .mean()
)

df["rolling_mean_8"] = grouped_sales.transform(
    lambda series:
        series.shift(1)
        .rolling(
            window=8,
            min_periods=8,
        )
        .mean()
)

df["rolling_std_4"] = grouped_sales.transform(
    lambda series:
        series.shift(1)
        .rolling(
            window=4,
            min_periods=4,
        )
        .std()
)

engineered_columns = [
    "lag_1",
    "lag_4",
    "lag_8",
    "rolling_mean_4",
    "rolling_mean_8",
    "rolling_std_4",
]

rows_before = len(df)

df_model = df.dropna(
    subset=engineered_columns
).copy()

rows_after = len(df_model)

print("\n========== Feature Engineering ==========")
print("Rows before:", rows_before)
print("Rows after:", rows_after)
print("Rows removed:", rows_before - rows_after)


# ============================================================
# 4. Train/test split
# ============================================================

test_start = pd.Timestamp("2024-06-01")
test_end = pd.Timestamp("2024-08-31")

train_df = df_model[
    df_model["date"] < test_start
].copy()

test_df = df_model[
    (df_model["date"] >= test_start)
    & (df_model["date"] <= test_end)
].copy()

if train_df.empty or test_df.empty:
    raise ValueError(
        "Train or test data is empty."
    )

print("\n========== Train / Test Split ==========")
print(
    "Train:",
    train_df["date"].min(),
    "to",
    train_df["date"].max(),
    train_df.shape,
)
print(
    "Test:",
    test_df["date"].min(),
    "to",
    test_df["date"].max(),
    test_df.shape,
)


# ============================================================
# 5. Feature scenarios
# ============================================================

target_column = "weekly_sales"

full_numeric_features = [
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

potentially_unavailable_features = {
    "temperature",
    "fuel_price",
    "cpi",
    "unemployment",
}

full_numeric_features = [
    feature
    for feature in full_numeric_features
    if feature in df_model.columns
]

categorical_features = [
    feature
    for feature in categorical_features
    if feature in df_model.columns
]

operational_numeric_features = [
    feature
    for feature in full_numeric_features
    if feature not in potentially_unavailable_features
]

feature_scenarios = {
    "full_information": {
        "numeric": full_numeric_features,
        "categorical": categorical_features,
    },
    "operational_conservative": {
        "numeric": operational_numeric_features,
        "categorical": categorical_features,
    },
}

print("\n========== Feature Scenarios ==========")

for scenario_name, feature_config in feature_scenarios.items():
    print(
        scenario_name,
        "numeric:",
        len(feature_config["numeric"]),
        "categorical:",
        len(feature_config["categorical"]),
        "total:",
        (
            len(feature_config["numeric"])
            + len(feature_config["categorical"])
        ),
    )

print(
    "Removed from operational scenario:",
    sorted(
        potentially_unavailable_features.intersection(
            full_numeric_features
        )
    ),
)


# ============================================================
# 6. Metrics
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


def evaluate_predictions(
    y_true,
    y_pred,
    model_name,
    scenario_name,
    evaluation_level,
    feature_count,
):
    result = {
        "scenario": scenario_name,
        "model": model_name,
        "mae": mean_absolute_error(
            y_true,
            y_pred,
        ),
        "rmse": (
            mean_squared_error(
                y_true,
                y_pred,
            )
            ** 0.5
        ),
        "mape": mape(
            y_true,
            y_pred,
        ),
        "wape": wape(
            y_true,
            y_pred,
        ),
        "forecast_bias": forecast_bias(
            y_true,
            y_pred,
        ),
        "evaluation_level": evaluation_level,
        "feature_count": feature_count,
    }

    if evaluation_level == "store_department_week":
        result["n_test_rows"] = len(y_true)
    else:
        result["n_test_weeks"] = len(y_true)

    return result


# ============================================================
# 7. Fresh preprocessing and models
# ============================================================

def build_preprocessor(
    numeric_features,
    categorical_features,
):
    numeric_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="median",
                ),
            ),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            (
                "imputer",
                SimpleImputer(
                    strategy="most_frequent",
                ),
            ),
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="ignore",
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            (
                "num",
                numeric_transformer,
                numeric_features,
            ),
            (
                "cat",
                categorical_transformer,
                categorical_features,
            ),
        ]
    )


def build_models(
    numeric_features,
    categorical_features,
):
    models = {}

    models["Linear Regression"] = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_features,
                    categorical_features,
                ),
            ),
            (
                "model",
                LinearRegression(),
            ),
        ]
    )

    models["Random Forest"] = Pipeline(
        steps=[
            (
                "preprocessor",
                build_preprocessor(
                    numeric_features,
                    categorical_features,
                ),
            ),
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

    if XGBOOST_AVAILABLE:
        models["XGBoost"] = Pipeline(
            steps=[
                (
                    "preprocessor",
                    build_preprocessor(
                        numeric_features,
                        categorical_features,
                    ),
                ),
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

    return models


# ============================================================
# 8. Train both scenarios
# ============================================================

granular_results = []
aggregate_results = []

for scenario_name, feature_config in feature_scenarios.items():
    numeric_features = feature_config["numeric"]
    scenario_categorical = feature_config["categorical"]

    feature_columns = (
        numeric_features
        + scenario_categorical
    )

    X_train = train_df[feature_columns]
    y_train = train_df[target_column]

    X_test = test_df[feature_columns]
    y_test = test_df[target_column]

    feature_count = len(feature_columns)

    print(
        f"\n========== Scenario: {scenario_name} =========="
    )
    print("Feature count:", feature_count)
    print("X_train:", X_train.shape)
    print("X_test:", X_test.shape)

    models = build_models(
        numeric_features,
        scenario_categorical,
    )

    for model_name, model_pipeline in models.items():
        print(
            f"\nTraining {model_name} "
            f"under {scenario_name}..."
        )

        model_pipeline.fit(
            X_train,
            y_train,
        )

        predictions = model_pipeline.predict(
            X_test
        )

        granular_results.append(
            evaluate_predictions(
                y_true=y_test,
                y_pred=predictions,
                model_name=model_name,
                scenario_name=scenario_name,
                evaluation_level=(
                    "store_department_week"
                ),
                feature_count=feature_count,
            )
        )

        prediction_frame = pd.DataFrame(
            {
                "date": test_df["date"].values,
                "actual": y_test.to_numpy(
                    dtype=float
                ),
                "predicted": np.asarray(
                    predictions,
                    dtype=float,
                ),
            }
        )

        aggregate_frame = (
            prediction_frame
            .groupby(
                "date",
                as_index=False,
            )
            .agg(
                actual_total=(
                    "actual",
                    "sum",
                ),
                predicted_total=(
                    "predicted",
                    "sum",
                ),
            )
        )

        aggregate_results.append(
            evaluate_predictions(
                y_true=aggregate_frame[
                    "actual_total"
                ],
                y_pred=aggregate_frame[
                    "predicted_total"
                ],
                model_name=model_name,
                scenario_name=scenario_name,
                evaluation_level="company_week",
                feature_count=feature_count,
            )
        )

        print(
            f"{model_name} completed "
            f"under {scenario_name}."
        )


# ============================================================
# 9. Save granular and aggregate results
# ============================================================

granular_results_df = (
    pd.DataFrame(granular_results)
    .sort_values(
        ["scenario", "wape"]
    )
    .reset_index(drop=True)
)

aggregate_results_df = (
    pd.DataFrame(aggregate_results)
    .sort_values(
        ["scenario", "wape"]
    )
    .reset_index(drop=True)
)

granular_path = (
    OUTPUT_DIR
    / "feature_availability_granular_comparison.csv"
)

aggregate_path = (
    OUTPUT_DIR
    / "feature_availability_aggregate_comparison.csv"
)

granular_results_df.to_csv(
    granular_path,
    index=False,
)

aggregate_results_df.to_csv(
    aggregate_path,
    index=False,
)

print(
    "\n========== Granular Comparison =========="
)
print(
    granular_results_df.to_string(
        index=False
    )
)

print(
    "\n========== Aggregate Comparison =========="
)
print(
    aggregate_results_df.to_string(
        index=False
    )
)


# ============================================================
# 10. Build sensitivity table
# ============================================================

full_results = (
    aggregate_results_df[
        aggregate_results_df["scenario"]
        == "full_information"
    ]
    .set_index("model")
)

operational_results = (
    aggregate_results_df[
        aggregate_results_df["scenario"]
        == "operational_conservative"
    ]
    .set_index("model")
)

common_models = (
    full_results.index
    .intersection(
        operational_results.index
    )
)

sensitivity_rows = []

for model_name in common_models:
    full_row = full_results.loc[model_name]
    operational_row = operational_results.loc[
        model_name
    ]

    full_wape = full_row["wape"]
    operational_wape = operational_row["wape"]

    sensitivity_rows.append(
        {
            "model": model_name,
            "full_mape": full_row["mape"],
            "operational_mape": (
                operational_row["mape"]
            ),
            "mape_change": (
                operational_row["mape"]
                - full_row["mape"]
            ),
            "full_wape": full_wape,
            "operational_wape": operational_wape,
            "wape_change": (
                operational_wape
                - full_wape
            ),
            "wape_relative_change_pct": (
                (
                    operational_wape
                    - full_wape
                )
                / full_wape
                * 100
                if full_wape != 0
                else np.nan
            ),
            "full_forecast_bias": (
                full_row["forecast_bias"]
            ),
            "operational_forecast_bias": (
                operational_row[
                    "forecast_bias"
                ]
            ),
            "full_feature_count": (
                full_row["feature_count"]
            ),
            "operational_feature_count": (
                operational_row[
                    "feature_count"
                ]
            ),
            "evaluation_level": "company_week",
        }
    )

sensitivity_df = (
    pd.DataFrame(sensitivity_rows)
    .sort_values("full_wape")
    .reset_index(drop=True)
)

sensitivity_path = (
    OUTPUT_DIR
    / "feature_availability_sensitivity.csv"
)

sensitivity_df.to_csv(
    sensitivity_path,
    index=False,
)

print(
    "\n========== Feature Availability Sensitivity =========="
)
print(
    sensitivity_df.to_string(
        index=False
    )
)


# ============================================================
# 11. Plot aggregate WAPE comparison
# ============================================================

plot_df = sensitivity_df.copy()

x_positions = np.arange(
    len(plot_df)
)

bar_width = 0.36

fig, ax = plt.subplots(
    figsize=(11, 6)
)

ax.bar(
    x_positions - bar_width / 2,
    plot_df["full_wape"],
    width=bar_width,
    label="Full Information",
)

ax.bar(
    x_positions + bar_width / 2,
    plot_df["operational_wape"],
    width=bar_width,
    label="Operational Conservative",
)

ax.set_xticks(
    x_positions
)

ax.set_xticklabels(
    plot_df["model"],
    rotation=15,
    ha="right",
)

ax.set_ylabel(
    "Company-Level WAPE (%)"
)

ax.set_title(
    "Feature Availability Sensitivity Analysis"
)

ax.legend()
ax.grid(
    axis="y",
    alpha=0.3,
)

fig.tight_layout()

figure_path = (
    FIGURE_DIR
    / "14_feature_availability_sensitivity.png"
)

fig.savefig(
    figure_path,
    dpi=150,
)

plt.close(fig)


# ============================================================
# 12. Completion summary
# ============================================================

print("\nSaved outputs to:")
print(granular_path)
print(aggregate_path)
print(sensitivity_path)
print(figure_path)

print(
    "\nFeature availability sensitivity "
    "analysis completed successfully."
)
