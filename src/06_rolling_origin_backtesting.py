from pathlib import Path
import gc
import warnings

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

try:
    from xgboost import XGBRegressor
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

warnings.filterwarnings("ignore")

BASE_DIR = Path(__file__).resolve().parents[1]
INPUT_PATH = BASE_DIR / "data" / "processed" / "retail_sales_cleaned_merged.csv"
OUTPUT_DIR = BASE_DIR / "outputs"
FIGURE_DIR = OUTPUT_DIR / "figures"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

INITIAL_TRAIN_WEEKS = 64
TEST_WEEKS_PER_FOLD = 14


def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    if not np.any(mask):
        return np.nan
    return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100


def wape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return np.nan
    return np.sum(np.abs(y_true - y_pred)) / denominator * 100


def forecast_bias(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    denominator = np.sum(np.abs(y_true))
    if denominator == 0:
        return np.nan
    return np.sum(y_pred - y_true) / denominator * 100


def evaluate_company_predictions(y_true, y_pred):
    return {
        "company_mae": mean_absolute_error(y_true, y_pred),
        "company_rmse": mean_squared_error(y_true, y_pred) ** 0.5,
        "company_mape": mape(y_true, y_pred),
        "company_wape": wape(y_true, y_pred),
        "company_forecast_bias": forecast_bias(y_true, y_pred),
    }


def build_preprocessor(numeric_features, categorical_features):
    numeric_transformer = Pipeline(
        steps=[("imputer", SimpleImputer(strategy="median"))]
    )
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    return ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )


def build_models(numeric_features, categorical_features):
    models = {
        "Operational Random Forest": Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
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
    }
    if XGBOOST_AVAILABLE:
        models["Operational XGBoost"] = Pipeline(
            steps=[
                ("preprocessor", build_preprocessor(numeric_features, categorical_features)),
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
                        verbosity=0,
                    ),
                ),
            ]
        )
    return models


def main():
    df = pd.read_csv(INPUT_PATH)
    df["date"] = pd.to_datetime(df["date"])

    required = {"store_id", "department", "date", "weekly_sales"}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.sort_values(["store_id", "department", "date"]).reset_index(drop=True)

    print("\n========== Loaded Data ==========")
    print("Shape:", df.shape)
    print("Date range:", df["date"].min(), "to", df["date"].max())

    df["year"] = df["date"].dt.year
    df["month"] = df["date"].dt.month
    df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)

    if "season" not in df.columns:
        season_map = {
            12: "Winter", 1: "Winter", 2: "Winter",
            3: "Spring", 4: "Spring", 5: "Spring",
            6: "Summer", 7: "Summer", 8: "Summer",
            9: "Autumn", 10: "Autumn", 11: "Autumn",
        }
        df["season"] = df["month"].map(season_map)

    markdown_columns = [f"markdown_{i}" for i in range(1, 6)]
    available_markdowns = [c for c in markdown_columns if c in df.columns]
    if "total_markdown" not in df.columns and available_markdowns:
        df["total_markdown"] = df[available_markdowns].fillna(0).sum(axis=1)

    grouped_sales = df.groupby(["store_id", "department"], sort=False)["weekly_sales"]
    df["lag_1"] = grouped_sales.shift(1)
    df["lag_4"] = grouped_sales.shift(4)
    df["lag_8"] = grouped_sales.shift(8)
    df["rolling_mean_4"] = grouped_sales.transform(
        lambda s: s.shift(1).rolling(4, min_periods=4).mean()
    )
    df["rolling_mean_8"] = grouped_sales.transform(
        lambda s: s.shift(1).rolling(8, min_periods=8).mean()
    )
    df["rolling_std_4"] = grouped_sales.transform(
        lambda s: s.shift(1).rolling(4, min_periods=4).std()
    )

    engineered = [
        "lag_1", "lag_4", "lag_8",
        "rolling_mean_4", "rolling_mean_8", "rolling_std_4",
    ]
    rows_before = len(df)
    df_model = df.dropna(subset=engineered).copy()

    print("\n========== Feature Engineering ==========")
    print("Rows before:", rows_before)
    print("Rows after:", len(df_model))
    print("Rows removed:", rows_before - len(df_model))

    numeric_features = [
        "year", "month", "week_of_year", "store_size",
        "markdown_1", "markdown_2", "markdown_3", "markdown_4", "markdown_5",
        "total_markdown", "is_holiday",
        "lag_1", "lag_4", "lag_8",
        "rolling_mean_4", "rolling_mean_8", "rolling_std_4",
    ]
    categorical_features = [
        "store_id", "department", "store_type", "region", "holiday_name", "season"
    ]
    numeric_features = [c for c in numeric_features if c in df_model.columns]
    categorical_features = [c for c in categorical_features if c in df_model.columns]
    feature_columns = numeric_features + categorical_features

    print("\n========== Operational Features ==========")
    print("Numeric features:", len(numeric_features))
    print("Categorical features:", len(categorical_features))
    print("Total features:", len(feature_columns))

    if len(feature_columns) != 23:
        raise ValueError(f"Expected 23 operational features, found {len(feature_columns)}")

    weekly_dates = pd.Index(sorted(df_model["date"].unique()))
    remaining_weeks = len(weekly_dates) - INITIAL_TRAIN_WEEKS
    if remaining_weeks <= 0:
        raise ValueError("Initial training period is too long.")
    if remaining_weeks % TEST_WEEKS_PER_FOLD != 0:
        raise ValueError(
            f"Remaining weeks ({remaining_weeks}) are not divisible by "
            f"{TEST_WEEKS_PER_FOLD}."
        )

    number_of_folds = remaining_weeks // TEST_WEEKS_PER_FOLD
    fold_plan_rows = []
    for fold_number in range(1, number_of_folds + 1):
        start_idx = INITIAL_TRAIN_WEEKS + (fold_number - 1) * TEST_WEEKS_PER_FOLD
        end_idx = start_idx + TEST_WEEKS_PER_FOLD
        test_dates = weekly_dates[start_idx:end_idx]
        fold_plan_rows.append(
            {
                "fold": fold_number,
                "train_start": weekly_dates[0],
                "train_end": weekly_dates[start_idx - 1],
                "test_start": test_dates[0],
                "test_end": test_dates[-1],
                "n_train_weeks": start_idx,
                "n_test_weeks": len(test_dates),
            }
        )

    fold_plan_df = pd.DataFrame(fold_plan_rows)
    print("\n========== Rolling-Origin Fold Plan ==========")
    print(fold_plan_df.to_string(index=False))

    company_weekly = (
        df.groupby("date", as_index=False)["weekly_sales"]
        .sum()
        .rename(columns={"weekly_sales": "actual_total_sales"})
        .sort_values("date")
        .reset_index(drop=True)
    )
    company_weekly["pred_rolling_4w"] = (
        company_weekly["actual_total_sales"]
        .shift(1)
        .rolling(4, min_periods=4)
        .mean()
    )

    fold_result_rows = []
    prediction_rows = []

    for fold_row in fold_plan_df.itertuples(index=False):
        fold_number = int(fold_row.fold)
        fold_start = pd.Timestamp(fold_row.test_start)
        fold_end = pd.Timestamp(fold_row.test_end)

        train_fold = df_model[df_model["date"] < fold_start].copy()
        test_fold = df_model[
            (df_model["date"] >= fold_start) & (df_model["date"] <= fold_end)
        ].copy()

        if test_fold["date"].nunique() != TEST_WEEKS_PER_FOLD:
            raise ValueError(f"Fold {fold_number} does not contain 14 test weeks.")

        print("\n============================================")
        print(f"Fold {fold_number}/{number_of_folds}")
        print("Train:", train_fold["date"].min(), "to", train_fold["date"].max(), train_fold.shape)
        print("Test:", test_fold["date"].min(), "to", test_fold["date"].max(), test_fold.shape)
        print("============================================")

        X_train = train_fold[feature_columns]
        y_train = train_fold["weekly_sales"]
        X_test = test_fold[feature_columns]
        y_test = test_fold["weekly_sales"]

        models = build_models(numeric_features, categorical_features)
        for model_name, model_pipeline in models.items():
            print(f"\nTraining {model_name}...")
            model_pipeline.fit(X_train, y_train)
            predictions = model_pipeline.predict(X_test)

            prediction_frame = pd.DataFrame(
                {
                    "date": test_fold["date"].to_numpy(),
                    "actual": y_test.to_numpy(dtype=float),
                    "predicted": np.asarray(predictions, dtype=float),
                }
            )
            aggregate_frame = (
                prediction_frame.groupby("date", as_index=False)
                .agg(
                    actual_total_sales=("actual", "sum"),
                    predicted_total_sales=("predicted", "sum"),
                )
            )
            company_metrics = evaluate_company_predictions(
                aggregate_frame["actual_total_sales"],
                aggregate_frame["predicted_total_sales"],
            )

            fold_result_rows.append(
                {
                    "fold": fold_number,
                    "model": model_name,
                    "train_start": train_fold["date"].min(),
                    "train_end": train_fold["date"].max(),
                    "test_start": fold_start,
                    "test_end": fold_end,
                    "n_train_weeks": train_fold["date"].nunique(),
                    "n_test_weeks": test_fold["date"].nunique(),
                    "n_train_rows": len(train_fold),
                    "n_test_rows": len(test_fold),
                    "feature_count": len(feature_columns),
                    "granular_mape": mape(y_test, predictions),
                    "granular_wape": wape(y_test, predictions),
                    "granular_forecast_bias": forecast_bias(y_test, predictions),
                    **company_metrics,
                    "evaluation_level": "company_week",
                    "forecast_horizon": "one_week_ahead",
                    "scenario": "operational_conservative",
                }
            )

            for row in aggregate_frame.itertuples(index=False):
                prediction_rows.append(
                    {
                        "fold": fold_number,
                        "model": model_name,
                        "date": row.date,
                        "actual_total_sales": row.actual_total_sales,
                        "predicted_total_sales": row.predicted_total_sales,
                    }
                )

            print(f"{model_name} completed. Company WAPE: {company_metrics['company_wape']:.4f}%")
            del model_pipeline, predictions
            gc.collect()

        baseline_fold = company_weekly[
            (company_weekly["date"] >= fold_start)
            & (company_weekly["date"] <= fold_end)
        ].dropna(subset=["pred_rolling_4w"])

        baseline_metrics = evaluate_company_predictions(
            baseline_fold["actual_total_sales"],
            baseline_fold["pred_rolling_4w"],
        )
        fold_result_rows.append(
            {
                "fold": fold_number,
                "model": "Rolling 4-Week Baseline",
                "train_start": train_fold["date"].min(),
                "train_end": train_fold["date"].max(),
                "test_start": fold_start,
                "test_end": fold_end,
                "n_train_weeks": train_fold["date"].nunique(),
                "n_test_weeks": len(baseline_fold),
                "n_train_rows": len(train_fold),
                "n_test_rows": np.nan,
                "feature_count": 0,
                "granular_mape": np.nan,
                "granular_wape": np.nan,
                "granular_forecast_bias": np.nan,
                **baseline_metrics,
                "evaluation_level": "company_week",
                "forecast_horizon": "one_week_ahead",
                "scenario": "baseline",
            }
        )

        for row in baseline_fold.itertuples(index=False):
            prediction_rows.append(
                {
                    "fold": fold_number,
                    "model": "Rolling 4-Week Baseline",
                    "date": row.date,
                    "actual_total_sales": row.actual_total_sales,
                    "predicted_total_sales": row.pred_rolling_4w,
                }
            )

        print(f"Rolling 4-Week Baseline completed. Company WAPE: {baseline_metrics['company_wape']:.4f}%")
        del models, train_fold, test_fold
        gc.collect()

    fold_results_df = (
        pd.DataFrame(fold_result_rows)
        .sort_values(["fold", "company_wape"])
        .reset_index(drop=True)
    )
    predictions_df = (
        pd.DataFrame(prediction_rows)
        .sort_values(["fold", "model", "date"])
        .reset_index(drop=True)
    )

    winner_indexes = fold_results_df.groupby("fold")["company_wape"].idxmin()
    fold_winners_df = (
        fold_results_df.loc[winner_indexes, ["fold", "model", "company_wape"]]
        .rename(
            columns={
                "model": "winning_model",
                "company_wape": "winning_company_wape",
            }
        )
        .sort_values("fold")
        .reset_index(drop=True)
    )

    summary_df = (
        fold_results_df.groupby("model", as_index=False)
        .agg(
            folds=("fold", "nunique"),
            mean_company_mape=("company_mape", "mean"),
            mean_company_wape=("company_wape", "mean"),
            median_company_wape=("company_wape", "median"),
            std_company_wape=("company_wape", "std"),
            min_company_wape=("company_wape", "min"),
            max_company_wape=("company_wape", "max"),
            mean_company_bias=("company_forecast_bias", "mean"),
            mean_absolute_company_bias=(
                "company_forecast_bias",
                lambda values: np.mean(np.abs(values)),
            ),
        )
    )

    win_counts = (
        fold_winners_df["winning_model"]
        .value_counts()
        .rename_axis("model")
        .reset_index(name="fold_wins")
    )
    summary_df = summary_df.merge(win_counts, on="model", how="left")
    summary_df["fold_wins"] = summary_df["fold_wins"].fillna(0).astype(int)
    summary_df = summary_df.sort_values(
        ["mean_company_wape", "std_company_wape"]
    ).reset_index(drop=True)
    summary_df.insert(0, "rank", np.arange(1, len(summary_df) + 1))

    print("\n========== Fold Results ==========")
    print(
        fold_results_df[
            [
                "fold", "model", "test_start", "test_end",
                "company_mape", "company_wape", "company_forecast_bias",
            ]
        ].to_string(index=False)
    )
    print("\n========== Fold Winners ==========")
    print(fold_winners_df.to_string(index=False))
    print("\n========== Rolling-Origin Summary ==========")
    print(summary_df.to_string(index=False))

    fold_plan_path = OUTPUT_DIR / "rolling_origin_fold_plan.csv"
    fold_results_path = OUTPUT_DIR / "rolling_origin_fold_results.csv"
    summary_path = OUTPUT_DIR / "rolling_origin_summary.csv"
    predictions_path = OUTPUT_DIR / "rolling_origin_company_predictions.csv"
    winners_path = OUTPUT_DIR / "rolling_origin_fold_winners.csv"

    fold_plan_df.to_csv(fold_plan_path, index=False)
    fold_results_df.to_csv(fold_results_path, index=False)
    summary_df.to_csv(summary_path, index=False)
    predictions_df.to_csv(predictions_path, index=False)
    fold_winners_df.to_csv(winners_path, index=False)

    fig, ax = plt.subplots(figsize=(12, 6))
    for model_name, model_frame in fold_results_df.groupby("model"):
        model_frame = model_frame.sort_values("fold")
        ax.plot(
            model_frame["fold"],
            model_frame["company_wape"],
            marker="o",
            label=model_name,
        )
    ax.set_xlabel("Backtest Fold")
    ax.set_ylabel("Company-Level WAPE (%)")
    ax.set_title("Rolling-Origin Backtesting: WAPE by Fold")
    ax.set_xticks(sorted(fold_results_df["fold"].unique()))
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()

    figure_path = FIGURE_DIR / "15_rolling_origin_wape_by_fold.png"
    fig.savefig(figure_path, dpi=150)
    plt.close(fig)

    print("\nSaved outputs to:")
    print(fold_plan_path)
    print(fold_results_path)
    print(summary_path)
    print(predictions_path)
    print(winners_path)
    print(figure_path)
    print("\nRolling-origin backtesting completed successfully.")


if __name__ == "__main__":
    main()
