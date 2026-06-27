from pathlib import Path
import pandas as pd


# ============================================================
# 0. 路径设置
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[1]

RAW_DIR = BASE_DIR / "data" / "raw"
PROCESSED_DIR = BASE_DIR / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SALES_PATH = RAW_DIR / "sales.csv"
FEATURES_PATH = RAW_DIR / "features.csv"
STORES_PATH = RAW_DIR / "stores.csv"


# ============================================================
# 1. 读取数据
# ============================================================

sales = pd.read_csv(SALES_PATH)
features = pd.read_csv(FEATURES_PATH)
stores = pd.read_csv(STORES_PATH)


# ============================================================
# 2. 统一字段名
# ============================================================

def clean_columns(df):
    df = df.copy()
    df.columns = (
        df.columns
        .str.strip()
        .str.lower()
        .str.replace(" ", "_")
        .str.replace("-", "_")
    )
    return df


sales = clean_columns(sales)
features = clean_columns(features)
stores = clean_columns(stores)


# ============================================================
# 3. 日期格式处理
# ============================================================

sales["date"] = pd.to_datetime(sales["date"])
features["date"] = pd.to_datetime(features["date"])


# ============================================================
# 4. 基础检查
# ============================================================

print("\n========== BASIC INFO ==========")
print("sales shape:", sales.shape)
print("features shape:", features.shape)
print("stores shape:", stores.shape)

print("\n========== DATE RANGE ==========")
print("sales:", sales["date"].min(), "to", sales["date"].max())
print("features:", features["date"].min(), "to", features["date"].max())

print("\n========== MISSING VALUES: SALES ==========")
print(sales.isna().sum())

print("\n========== MISSING VALUES: FEATURES ==========")
print(features.isna().sum())

print("\n========== MISSING VALUES: STORES ==========")
print(stores.isna().sum())

print("\n========== DUPLICATES ==========")
print("sales duplicated:", sales.duplicated().sum())
print("features duplicated:", features.duplicated().sum())
print("stores duplicated:", stores.duplicated().sum())

# ============================================================
# 5. 业务主键唯一性检查
# ============================================================

def validate_unique_key(
    df: pd.DataFrame,
    key_columns: list[str],
    dataset_name: str,
) -> None:
    """
    检查业务主键是否唯一。

    不直接删除重复记录，因为静默去重可能掩盖原始数据问题。
    如果发现重复业务主键，程序会停止并显示部分问题记录。
    """

    duplicate_mask = df.duplicated(
        subset=key_columns,
        keep=False,
    )

    duplicate_count = int(duplicate_mask.sum())

    if duplicate_count > 0:
        duplicate_sample = (
            df.loc[duplicate_mask, key_columns]
            .sort_values(key_columns)
            .head(10)
        )

        raise ValueError(
            f"\n{dataset_name} contains {duplicate_count:,} rows "
            f"involved in duplicate business keys.\n"
            f"Key columns: {key_columns}\n"
            f"Sample duplicate keys:\n"
            f"{duplicate_sample.to_string(index=False)}"
        )

    print(
        f"{dataset_name} business key validation: PASS "
        f"({key_columns})"
    )


validate_unique_key(
    sales,
    ["store_id", "department", "date"],
    "Sales",
)

validate_unique_key(
    features,
    ["store_id", "date"],
    "Features",
)

validate_unique_key(
    stores,
    ["store_id"],
    "Stores",
)

# ============================================================
# 6. 清洗 sales 表
# ============================================================

# 删除 weekly_sales 为空的记录
sales = sales.dropna(subset=["weekly_sales"])

# 负销售额不直接删，先打标记
# 在零售场景里，负数有时可能代表退货或调整
sales["is_negative_sales"] = sales["weekly_sales"] < 0

# 添加日期特征
sales["year"] = sales["date"].dt.year
sales["month"] = sales["date"].dt.month
sales["week_of_year"] = sales["date"].dt.isocalendar().week.astype(int)


# ============================================================
# 7. 清洗 features 表
# ============================================================

# markdown 促销字段缺失值填 0
# 业务含义：没有 markdown 记录，可以理解为没有促销折扣
markdown_cols = [col for col in features.columns if col.startswith("markdown")]

for col in markdown_cols:
    features[col] = features[col].fillna(0)

# 数值型外部特征用同一门店的中位数填补
numeric_cols = ["temperature", "fuel_price", "cpi", "unemployment"]

for col in numeric_cols:
    if col in features.columns:
        features[col] = features.groupby("store_id")[col].transform(
            lambda x: x.fillna(x.median())
        )
        features[col] = features[col].fillna(features[col].median())

# 节假日名称缺失填 Non-Holiday
if "holiday_name" in features.columns:
    features["holiday_name"] = features["holiday_name"].fillna("Non-Holiday")

if "season" in features.columns:
    features["season"] = features["season"].fillna("Unknown")


# ============================================================
# 8. 清洗 stores 表
# ============================================================

if "store_type" in stores.columns:
    stores["store_type"] = stores["store_type"].fillna("Unknown")

if "region" in stores.columns:
    stores["region"] = stores["region"].fillna("Unknown")

if "store_size" in stores.columns:
    stores["store_size"] = stores["store_size"].fillna(stores["store_size"].median())


# ============================================================
# 9. 合并三张表
# sales + features + stores
# ============================================================

sales_row_count = len(sales)

# Sales 中的多条部门记录，对应 Features 中的一条门店周度记录
df = sales.merge(
    features,
    on=["store_id", "date"],
    how="left",
    suffixes=("_sales", "_features"),
    validate="many_to_one",
    indicator="features_merge_status",
)

unmatched_feature_rows = int(
    (df["features_merge_status"] != "both").sum()
)

if unmatched_feature_rows > 0:
    raise ValueError(
        f"{unmatched_feature_rows:,} sales rows did not match "
        "a Features record."
    )

df = df.drop(columns=["features_merge_status"])

# Sales 中的多条记录，对应 Stores 中的一条门店资料
df = df.merge(
    stores,
    on="store_id",
    how="left",
    validate="many_to_one",
    indicator="stores_merge_status",
)

unmatched_store_rows = int(
    (df["stores_merge_status"] != "both").sum()
)

if unmatched_store_rows > 0:
    raise ValueError(
        f"{unmatched_store_rows:,} sales rows did not match "
        "a Stores record."
    )

df = df.drop(columns=["stores_merge_status"])

# 合并后行数必须和 Sales 原始行数相同
if len(df) != sales_row_count:
    raise ValueError(
        "Row count changed unexpectedly after merging: "
        f"{sales_row_count:,} → {len(df):,}"
    )

print(
    f"Merge row-count validation: PASS "
    f"({sales_row_count:,} rows)"
)

# 合并前已经分别保留了 Sales 和 Features 的 holiday 字段
if {
    "is_holiday_sales",
    "is_holiday_features",
}.issubset(df.columns):

    holiday_mismatch_count = int(
        (
            df["is_holiday_sales"]
            != df["is_holiday_features"]
        ).sum()
    )

    if holiday_mismatch_count > 0:
        raise ValueError(
            f"Holiday flag mismatch found in "
            f"{holiday_mismatch_count:,} merged rows."
        )

    print("Holiday consistency validation: PASS")
# ============================================================
# 10. 统一 holiday 字段
# ============================================================

if "is_holiday_sales" in df.columns:
    df["is_holiday"] = df["is_holiday_sales"]
elif "is_holiday_features" in df.columns:
    df["is_holiday"] = df["is_holiday_features"]

drop_cols = [
    col for col in ["is_holiday_sales", "is_holiday_features"]
    if col in df.columns
]

df = df.drop(columns=drop_cols)


# ============================================================
# 11. 添加夏季字段
# ============================================================

df["is_summer"] = df["month"].isin([6, 7, 8]).astype(int)


# ============================================================
# 12. 合并后检查
# ============================================================

print("\n========== MERGED DATA ==========")
print("merged shape:", df.shape)
print(df.head())

print("\n========== MISSING VALUES AFTER MERGE ==========")
print(df.isna().sum().sort_values(ascending=False).head(20))


# ============================================================
# 13. 保存清洗结果
# ============================================================

sales.to_csv(PROCESSED_DIR / "sales_cleaned.csv", index=False)
features.to_csv(PROCESSED_DIR / "features_cleaned.csv", index=False)
stores.to_csv(PROCESSED_DIR / "stores_cleaned.csv", index=False)
df.to_csv(PROCESSED_DIR / "retail_sales_cleaned_merged.csv", index=False)

print("\n========== SAVED FILES ==========")
print(PROCESSED_DIR / "sales_cleaned.csv")
print(PROCESSED_DIR / "features_cleaned.csv")
print(PROCESSED_DIR / "stores_cleaned.csv")
print(PROCESSED_DIR / "retail_sales_cleaned_merged.csv")

print("\nData cleaning completed successfully.")