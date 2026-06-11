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
# 5. 去重
# ============================================================

sales = sales.drop_duplicates()
features = features.drop_duplicates()
stores = stores.drop_duplicates()


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

df = sales.merge(
    features,
    on=["store_id", "date"],
    how="left",
    suffixes=("_sales", "_features")
)

df = df.merge(
    stores,
    on="store_id",
    how="left"
)


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