from pathlib import Path

import pandas as pd


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


# ============================================================
# 2. 读取数据
# ============================================================

raw_sales = pd.read_csv(
    RAW_DATA_DIR / "sales.csv",
    parse_dates=["date"],
)

cleaned = pd.read_csv(
    PROCESSED_DATA_DIR / "retail_sales_cleaned_merged.csv",
    parse_dates=["date"],
)


def print_section(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


# ============================================================
# 3. 行数与销售总额检查
# ============================================================

print_section("A. ROW COUNT AND SALES PRESERVATION")

raw_row_count = len(raw_sales)
cleaned_row_count = len(cleaned)

raw_sales_total = raw_sales["weekly_sales"].sum()
cleaned_sales_total = cleaned["weekly_sales"].sum()

sales_difference = cleaned_sales_total - raw_sales_total

print(f"Raw sales rows:       {raw_row_count:,}")
print(f"Cleaned sales rows:   {cleaned_row_count:,}")

print(f"\nRaw sales total:      {raw_sales_total:,.2f}")
print(f"Cleaned sales total:  {cleaned_sales_total:,.2f}")
print(f"Difference:           {sales_difference:,.6f}")


# ============================================================
# 4. 业务主键检查
# ============================================================

print_section("B. BUSINESS KEY CHECK")

duplicate_keys = cleaned.duplicated(
    subset=["store_id", "department", "date"],
    keep=False,
).sum()

print(
    "Rows involved in duplicate business keys: "
    f"{duplicate_keys:,}"
)


# ============================================================
# 5. 销售值检查
# ============================================================

print_section("C. SALES VALUE CHECK")

negative_sales_count = int(
    (cleaned["weekly_sales"] < 0).sum()
)

zero_sales_count = int(
    (cleaned["weekly_sales"] == 0).sum()
)

print(f"Negative sales rows: {negative_sales_count:,}")
print(f"Zero sales rows:     {zero_sales_count:,}")

print("\nWeekly sales summary:")
print(
    cleaned["weekly_sales"]
    .describe()
    .to_string()
)

if "is_negative_sales" in cleaned.columns:
    flag_mismatch = int(
        (
            cleaned["is_negative_sales"].astype(bool)
            != (cleaned["weekly_sales"] < 0)
        ).sum()
    )

    print(
        "\nNegative-sales flag mismatches: "
        f"{flag_mismatch:,}"
    )


# ============================================================
# 6. Holiday 检查
# ============================================================

print_section("D. HOLIDAY CHECK")

print("is_holiday distribution:")
print(
    cleaned["is_holiday"]
    .value_counts(dropna=False)
    .sort_index()
    .to_string()
)

if "holiday_name" in cleaned.columns:
    print("\nTop holiday_name values:")
    print(
        cleaned["holiday_name"]
        .value_counts(dropna=False)
        .head(15)
        .to_string()
    )

non_holiday_labels = [
    "Non-Holiday",
    "No Holiday",
    "None",
    "",
    "nan",
]

holiday_name_cleaned = (
    cleaned["holiday_name"]
    .fillna("")
    .astype(str)
    .str.strip()
)

# 检查：节假日行是否错误地使用了“非节假日”占位符
holiday_rows_without_name = int(
    (
        (cleaned["is_holiday"] == 1)
        & holiday_name_cleaned.isin(non_holiday_labels)
    ).sum()
)

# 检查：非节假日行是否错误地填写了具体节假日名称
non_holiday_rows_with_holiday_name = int(
    (
        (cleaned["is_holiday"] == 0)
        & ~holiday_name_cleaned.isin(non_holiday_labels)
    ).sum()
)

print(
    "\nHoliday rows using a non-holiday placeholder: "
    f"{holiday_rows_without_name:,}"
)

print(
    "Non-holiday rows using a holiday name: "
    f"{non_holiday_rows_with_holiday_name:,}"
)


# ============================================================
# 7. 日期特征检查
# ============================================================

print_section("E. DATE FEATURE CHECK")

expected_summer = (
    cleaned["date"]
    .dt.month
    .isin([6, 7, 8])
    .astype(int)
)

summer_flag_mismatches = int(
    (cleaned["is_summer"] != expected_summer).sum()
)

print(
    "is_summer mismatches: "
    f"{summer_flag_mismatches:,}"
)

if "year" in cleaned.columns:
    year_mismatches = int(
        (cleaned["year"] != cleaned["date"].dt.year).sum()
    )
    print(f"Year mismatches:      {year_mismatches:,}")

if "month" in cleaned.columns:
    month_mismatches = int(
        (cleaned["month"] != cleaned["date"].dt.month).sum()
    )
    print(f"Month mismatches:     {month_mismatches:,}")


# ============================================================
# 8. 类别字段检查
# ============================================================

print_section("F. CATEGORY CHECK")

for column in ["store_type", "region", "season"]:
    if column in cleaned.columns:
        print(f"\n{column}:")
        print(
            cleaned[column]
            .value_counts(dropna=False)
            .to_string()
        )


# ============================================================
# 9. 数值范围检查
# ============================================================

print_section("G. NUMERIC FEATURE RANGES")

numeric_columns = [
    "temperature",
    "fuel_price",
    "markdown_1",
    "markdown_2",
    "markdown_3",
    "markdown_4",
    "markdown_5",
    "cpi",
    "unemployment",
    "store_size",
]

available_numeric_columns = [
    column
    for column in numeric_columns
    if column in cleaned.columns
]

range_summary = cleaned[
    available_numeric_columns
].agg(["min", "max", "mean"]).T

print(range_summary.to_string())


# ============================================================
# 10. 最终判断
# ============================================================

print_section("H. FINAL AUDIT RESULT")

checks_passed = all(
    [
        raw_row_count == cleaned_row_count,
        abs(sales_difference) < 1e-6,
        duplicate_keys == 0,
        summer_flag_mismatches == 0,
    ]
)

if checks_passed:
    print("PASS")
    print(
        "The cleaning process preserved all sales records "
        "and the original total sales amount."
    )
else:
    print("FOLLOW-UP REQUIRED")
    print(
        "At least one cleaning audit check requires review."
    )