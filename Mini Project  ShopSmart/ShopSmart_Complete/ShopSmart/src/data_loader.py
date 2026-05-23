"""
src/data_loader.py
==================
ShopSmart — Data Loading & Cleaning

Functions
---------
load_data           load raw CSV, coerce types
validate_data       schema + quality checks, print report
clean_data          impute nulls, clip ranges, strip whitespace
load_clean_data     convenience one-shot: load → validate → clean
get_date_range      (min_date, max_date) of the dataset
get_category_summary aggregated stats per category

Expected CSV columns
--------------------
date             : "YYYY-MM-DD"
product_id       : unique product identifier
product_name     : human-readable name
category         : product category string
price            : float, unit price USD
units_sold       : int, units sold in this session
customer_rating  : float 1.0–5.0 (nullable)
sales            : float = price × units_sold (derived if absent)
"""

from __future__ import annotations
import logging
from pathlib import Path
from typing import Optional
import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

REQUIRED_COLUMNS = {"date","product_name","category","price","units_sold","customer_rating"}
NUMERIC_COLUMNS  = ["price","units_sold","customer_rating","sales"]
DATE_COLUMN      = "date"
VALID_RANGES: dict[str, tuple[float, float]] = {
    "price"          : (0.01, 10_000),
    "units_sold"     : (0,    10_000),
    "customer_rating": (1.0,  5.0),
    "sales"          : (0.0,  1_000_000),
}


def load_data(filepath: str | Path, parse_dates: bool = True) -> pd.DataFrame:
    """
    Load the raw shop CSV and apply minimal type coercions.

    Parameters
    ----------
    filepath    : path to shop_data.csv
    parse_dates : cast 'date' column to datetime64 when True

    Returns
    -------
    pd.DataFrame with correct dtypes.
    Raises FileNotFoundError if the file is missing.
    """
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Data file not found: {filepath}")

    df = pd.read_csv(filepath)

    # Validate required columns exist
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing required columns: {missing}")

    # Type coercions
    if parse_dates and DATE_COLUMN in df.columns:
        df[DATE_COLUMN] = pd.to_datetime(df[DATE_COLUMN], errors="coerce")
    for col in NUMERIC_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # Derive 'sales' if not present
    if "sales" not in df.columns:
        df["sales"] = df["price"] * df["units_sold"]

    logger.info("Loaded %d rows × %d columns from %s", len(df), len(df.columns), filepath)
    return df


def validate_data(df: pd.DataFrame, verbose: bool = True) -> dict:
    """
    Run data-quality checks and return a report dictionary.

    Checks: missing values, duplicates, out-of-range values, invalid dates.

    Parameters
    ----------
    df      : DataFrame from load_data()
    verbose : print report to stdout when True

    Returns
    -------
    dict with keys: n_rows, n_cols, null_counts, n_duplicates, range_violations
    """
    report = {
        "n_rows"      : len(df),
        "n_cols"      : len(df.columns),
        "null_counts" : df.isnull().sum().to_dict(),
        "n_duplicates": int(df.duplicated().sum()),
        "range_violations": {},
    }
    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            bad = df[col].dropna()
            bad = bad[(bad < lo) | (bad > hi)]
            if len(bad):
                report["range_violations"][col] = len(bad)

    if verbose:
        print("=" * 55)
        print("DATA VALIDATION REPORT")
        print("=" * 55)
        print(f"  Rows          : {report['n_rows']:,}")
        print(f"  Columns       : {report['n_cols']}")
        print(f"  Duplicates    : {report['n_duplicates']}")
        null_total = sum(v for v in report["null_counts"].values() if v)
        print(f"  Total nulls   : {null_total}")
        if null_total:
            for col, cnt in report["null_counts"].items():
                if cnt:
                    print(f"    {col:<22}: {cnt} ({cnt/report['n_rows']*100:.1f}%)")
        violations = report["range_violations"]
        print(f"  Range issues  : {len(violations)} column(s)")
        for col, cnt in violations.items():
            print(f"    {col:<22}: {cnt} rows outside {VALID_RANGES[col]}")
        print("=" * 55)

    return report


def clean_data(df: pd.DataFrame,
               drop_duplicates: bool = True,
               fill_strategy: str = "median") -> pd.DataFrame:
    """
    Clean raw DataFrame for downstream use.

    Steps
    -----
    1. Drop exact duplicate rows (optional)
    2. Fill missing numeric columns using median / mean, or drop rows
    3. Clip values to valid ranges
    4. Strip whitespace from string columns
    5. Re-derive 'sales' for consistency

    Parameters
    ----------
    df              : output of load_data()
    drop_duplicates : remove identical rows when True
    fill_strategy   : "median" | "mean" | "drop"

    Returns
    -------
    pd.DataFrame (new copy; original unmodified)
    """
    if fill_strategy not in {"median", "mean", "drop"}:
        raise ValueError(f"fill_strategy must be 'median', 'mean', or 'drop'. Got: {fill_strategy!r}")

    df = df.copy()
    n0 = len(df)

    if drop_duplicates:
        df = df.drop_duplicates()
        if len(df) < n0:
            logger.info("Removed %d duplicate rows.", n0 - len(df))

    fillable = [c for c in ["customer_rating", "units_sold", "price"] if c in df.columns]
    if fill_strategy == "drop":
        before = len(df)
        df = df.dropna(subset=fillable)
        logger.info("Dropped %d rows with missing values.", before - len(df))
    else:
        agg = df[fillable].median() if fill_strategy == "median" else df[fillable].mean()
        df[fillable] = df[fillable].fillna(agg)

    for col, (lo, hi) in VALID_RANGES.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lo, upper=hi)

    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].str.strip()

    df["sales"] = (df["price"] * df["units_sold"]).round(2)
    logger.info("clean_data: %d → %d rows.", n0, len(df))
    return df


def load_clean_data(filepath: str | Path,
                    fill_strategy: str = "median",
                    verbose: bool = True) -> pd.DataFrame:
    """
    One-shot: load → validate → clean.

    Parameters
    ----------
    filepath      : path to shop_data.csv
    fill_strategy : passed to clean_data()
    verbose       : print validation report

    Returns
    -------
    Clean pd.DataFrame ready for feature engineering.
    """
    df = load_data(filepath)
    validate_data(df, verbose=verbose)
    return clean_data(df, fill_strategy=fill_strategy)


def get_date_range(df: pd.DataFrame) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Return (min_date, max_date) from the 'date' column."""
    col = df[DATE_COLUMN]
    return col.min(), col.max()


def get_category_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate sales, units, average price and rating per category.

    Returns
    -------
    pd.DataFrame sorted by total_sales descending.
    """
    return (
        df.groupby("category", sort=False)
        .agg(
            total_sales    = ("sales",          "sum"),
            total_units    = ("units_sold",      "sum"),
            avg_price      = ("price",           "mean"),
            avg_rating     = ("customer_rating", "mean"),
            n_transactions = ("sales",           "count"),
        )
        .round(2)
        .sort_values("total_sales", ascending=False)
        .reset_index()
    )


if __name__ == "__main__":
    import sys
    fp = sys.argv[1] if len(sys.argv) > 1 else "data/shop_data.csv"
    df = load_clean_data(fp)
    print(f"\nShape: {df.shape}")
    print(get_category_summary(df).to_string(index=False))
