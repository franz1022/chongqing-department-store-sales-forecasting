from pathlib import Path

import pandas as pd


# ============================================================
# 1. 项目路径与数据读取
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"


def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """读取三张原始数据表，并将日期解析为 datetime。"""

    sales = pd.read_csv(
        RAW_DATA_DIR / "sales.csv",
        parse_dates=["date"],
    )

    features = pd.read_csv(
        RAW_DATA_DIR / "features.csv",
        parse_dates=["date"],
    )

    stores = pd.read_csv(
        RAW_DATA_DIR / "stores.csv",
    )

    return sales, features, stores


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# 2. 主程序
# ============================================================

def main() -> None:
    sales, features, stores = load_data()

    # --------------------------------------------------------
    # A. 检查数据规模和面板结构
    # --------------------------------------------------------

    print_section("A. DATA CARDINALITY AND PANEL STRUCTURE")

    number_of_stores = sales["store_id"].nunique()
    number_of_departments = sales["department"].nunique()
    number_of_dates = sales["date"].nunique()
    number_of_series = (
        sales[["store_id", "department"]]
        .drop_duplicates()
        .shape[0]
    )

    theoretical_rows = (
        number_of_stores
        * number_of_departments
        * number_of_dates
    )

    print(f"Unique stores:                 {number_of_stores:,}")
    print(f"Unique departments:            {number_of_departments:,}")
    print(f"Unique dates:                  {number_of_dates:,}")
    print(f"Store-department series:       {number_of_series:,}")
    print(f"Actual sales rows:             {len(sales):,}")
    print(f"Theoretical full-panel rows:   {theoretical_rows:,}")

    observations_per_series = (
        sales.groupby(["store_id", "department"])
        .size()
    )

    departments_per_store = (
        sales.groupby("store_id")["department"]
        .nunique()
    )

    print(
        "Observations per store-department series: "
        f"min={observations_per_series.min()}, "
        f"max={observations_per_series.max()}"
    )

    print(
        "Departments per store: "
        f"min={departments_per_store.min()}, "
        f"max={departments_per_store.max()}"
    )

    # --------------------------------------------------------
    # B. 检查 Sales 与 Features 的 key 是否完整匹配
    # --------------------------------------------------------

    print_section("B. SALES TO FEATURES KEY COVERAGE")

    sales_store_date = (
        sales[["store_id", "date"]]
        .drop_duplicates()
    )

    feature_store_date = (
        features[["store_id", "date"]]
        .drop_duplicates()
    )

    missing_feature_keys = (
        sales_store_date
        .merge(
            feature_store_date,
            on=["store_id", "date"],
            how="left",
            indicator=True,
            validate="one_to_one",
        )
        .query("_merge == 'left_only'")
    )

    unused_feature_keys = (
        feature_store_date
        .merge(
            sales_store_date,
            on=["store_id", "date"],
            how="left",
            indicator=True,
            validate="one_to_one",
        )
        .query("_merge == 'left_only'")
    )

    print(
        "Sales store-date keys without Features: "
        f"{len(missing_feature_keys):,}"
    )

    print(
        "Features store-date keys without Sales: "
        f"{len(unused_feature_keys):,}"
    )

    # --------------------------------------------------------
    # C. 检查 Sales 与 Stores 的门店覆盖
    # --------------------------------------------------------

    print_section("C. SALES TO STORES KEY COVERAGE")

    sales_store_ids = set(sales["store_id"].unique())
    store_master_ids = set(stores["store_id"].unique())

    missing_store_ids = sales_store_ids - store_master_ids
    unused_store_ids = store_master_ids - sales_store_ids

    print(
        "Sales store IDs missing from Stores: "
        f"{len(missing_store_ids):,}"
    )

    print(
        "Stores IDs not used by Sales: "
        f"{len(unused_store_ids):,}"
    )

    if missing_store_ids:
        print(f"Missing store IDs: {sorted(missing_store_ids)}")

    # --------------------------------------------------------
    # D. 检查 Sales 内部的 holiday 是否一致
    # --------------------------------------------------------

    print_section("D. HOLIDAY CONSISTENCY")

    sales_holiday_variation = (
        sales.groupby(["store_id", "date"])["is_holiday"]
        .nunique()
    )

    inconsistent_sales_holiday_keys = (
        sales_holiday_variation[
            sales_holiday_variation > 1
        ]
    )

    print(
        "Store-date keys with inconsistent holiday flags "
        f"inside Sales: {len(inconsistent_sales_holiday_keys):,}"
    )

    sales_holiday = (
        sales.groupby(
            ["store_id", "date"],
            as_index=False,
        )["is_holiday"]
        .first()
    )

    holiday_comparison = sales_holiday.merge(
        features[["store_id", "date", "is_holiday"]],
        on=["store_id", "date"],
        how="inner",
        validate="one_to_one",
        suffixes=("_sales", "_features"),
    )

    holiday_mismatches = holiday_comparison[
        holiday_comparison["is_holiday_sales"]
        != holiday_comparison["is_holiday_features"]
    ]

    print(
        "Holiday flag mismatches between Sales and Features: "
        f"{len(holiday_mismatches):,}"
    )

    # --------------------------------------------------------
    # E. 正式执行合并，并检查行数
    # --------------------------------------------------------

    print_section("E. MERGE VALIDATION")

    merged = sales.merge(
        features,
        on=["store_id", "date"],
        how="left",
        validate="many_to_one",
        suffixes=("_sales", "_features"),
    )

    merged = merged.merge(
        stores,
        on="store_id",
        how="left",
        validate="many_to_one",
    )

    print(f"Sales rows before merge: {len(sales):,}")
    print(f"Rows after merge:        {len(merged):,}")

    duplicate_business_keys = merged.duplicated(
        subset=["store_id", "department", "date"],
        keep=False,
    ).sum()

    print(
        "Rows involved in duplicate business keys after merge: "
        f"{duplicate_business_keys:,}"
    )

    missing_after_merge = merged.isna().sum()

    # holiday_name 缺失属于预期情况，因此先排除
    unexpected_missing = (
        missing_after_merge
        .drop(labels=["holiday_name"], errors="ignore")
    )

    unexpected_missing = unexpected_missing[
        unexpected_missing > 0
    ]

    print("\nUnexpected missing values after merge:")

    if unexpected_missing.empty:
        print("  None")
    else:
        for column, count in unexpected_missing.items():
            print(f"  {column}: {count:,}")

    # --------------------------------------------------------
    # F. 最终判断
    # --------------------------------------------------------

    print_section("F. FINAL VALIDATION RESULT")

    all_checks_passed = all(
        [
            len(missing_feature_keys) == 0,
            len(unused_feature_keys) == 0,
            len(missing_store_ids) == 0,
            len(unused_store_ids) == 0,
            len(inconsistent_sales_holiday_keys) == 0,
            len(holiday_mismatches) == 0,
            len(merged) == len(sales),
            duplicate_business_keys == 0,
            unexpected_missing.empty,
        ]
    )

    if all_checks_passed:
        print("PASS")
        print(
            "The three datasets have complete key coverage and "
            "can be merged safely without row inflation."
        )
    else:
        print("FOLLOW-UP REQUIRED")
        print(
            "At least one relationship or merge validation check "
            "requires review."
        )


if __name__ == "__main__":
    main()