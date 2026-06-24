from pathlib import Path

import pandas as pd


# ============================================================
# 1. 项目路径
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"

DATA_FILES = {
    "sales": RAW_DATA_DIR / "sales.csv",
    "features": RAW_DATA_DIR / "features.csv",
    "stores": RAW_DATA_DIR / "stores.csv",
}


# ============================================================
# 2. 工具函数：忽略大小写寻找字段
# ============================================================

def find_column(df: pd.DataFrame, expected_name: str) -> str | None:
    """
    在 DataFrame 中查找字段，忽略大小写和首尾空格。
    例如可以同时识别 Date、date 或 DATE。
    """
    column_map = {
        str(column).strip().lower(): str(column)
        for column in df.columns
    }
    return column_map.get(expected_name.strip().lower())


def inspect_dataframe(
    name: str,
    df: pd.DataFrame,
    expected_key_names: list[str],
) -> None:
    """打印一张表的基础结构和数据质量信息。"""

    print("\n" + "=" * 80)
    print(f"DATASET: {name.upper()}")
    print("=" * 80)

    print(f"\nShape: {df.shape[0]:,} rows × {df.shape[1]} columns")

    print("\nColumns:")
    for number, column in enumerate(df.columns, start=1):
        print(f"  {number:>2}. {column}")

    print("\nData types:")
    print(df.dtypes.to_string())

    print("\nFirst 3 rows:")
    print(df.head(3).to_string(index=False))

    print("\nMissing values:")
    missing = df.isna().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if missing.empty:
        print("  No missing values.")
    else:
        for column, count in missing.items():
            percentage = count / len(df) * 100
            print(f"  {column}: {count:,} ({percentage:.2f}%)")

    duplicate_rows = int(df.duplicated().sum())
    print(f"\nFully duplicated rows: {duplicate_rows:,}")

    # 日期范围检查
    date_column = find_column(df, "Date")

    if date_column is not None:
        parsed_dates = pd.to_datetime(df[date_column], errors="coerce")

        print("\nDate information:")
        print(f"  Earliest date: {parsed_dates.min()}")
        print(f"  Latest date:   {parsed_dates.max()}")
        print(f"  Invalid dates: {parsed_dates.isna().sum():,}")

    # 业务主键检查
    actual_key_columns = []

    for expected_name in expected_key_names:
        actual_name = find_column(df, expected_name)

        if actual_name is not None:
            actual_key_columns.append(actual_name)

    print("\nExpected business key:")
    print(f"  {expected_key_names}")

    if len(actual_key_columns) != len(expected_key_names):
        print("  Some expected key columns were not found.")
    else:
        duplicate_keys = int(
            df.duplicated(subset=actual_key_columns, keep=False).sum()
        )

        unique_keys = int(
            df[actual_key_columns].drop_duplicates().shape[0]
        )

        print(f"  Actual key columns: {actual_key_columns}")
        print(f"  Unique key combinations: {unique_keys:,}")
        print(f"  Rows involved in duplicate keys: {duplicate_keys:,}")


# ============================================================
# 3. 主程序
# ============================================================

def main() -> None:
    expected_keys = {
        "sales": ["store_id", "department", "date"],
        "features": ["store_id", "date"],
        "stores": ["store_id"],
    }

    for dataset_name, file_path in DATA_FILES.items():
        if not file_path.exists():
            raise FileNotFoundError(
                f"File not found: {file_path}"
            )

        dataframe = pd.read_csv(file_path)

        inspect_dataframe(
            name=dataset_name,
            df=dataframe,
            expected_key_names=expected_keys[dataset_name],
        )


if __name__ == "__main__":
    main()