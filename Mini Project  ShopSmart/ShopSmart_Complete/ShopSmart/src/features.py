"""
src/features.py
===============
ShopSmart — Feature Engineering

All functions are stateless, accept a DataFrame, and return a new
DataFrame (no in-place mutation) so they can be safely chained.

Public API
----------
add_price_bins              discretise price into labelled bands
add_date_features           extract day/week/month from 'date'
add_sales_per_unit          revenue-per-unit + category-normalised ratio
add_high_value_flag         binary flag for top-rated premium products
add_category_stats          merge group-level aggregates to each row
encode_categoricals         label-encode or one-hot categorical columns
scale_features              StandardScale selected numeric columns
build_feature_matrix        run all steps in the correct order
"""

from __future__ import annotations
import logging
from typing import Optional
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

logger = logging.getLogger(__name__)

PRICE_BIN_EDGES  = [0, 15, 30, 60, 100, 200, np.inf]
PRICE_BIN_LABELS = ["Budget","Economy","Mid-Range","Premium","Luxury","Ultra-Luxury"]


# ── 1. Price binning ─────────────────────────────────────────────────────────

def add_price_bins(df: pd.DataFrame, price_col: str = "price") -> pd.DataFrame:
    """
    Add 'price_bin' (string label) and 'price_bin_code' (int 0–5).

    Bins: Budget($0-15) | Economy($15-30) | Mid-Range($30-60)
          Premium($60-100) | Luxury($100-200) | Ultra-Luxury(>$200)

    Parameters
    ----------
    df        : clean DataFrame
    price_col : name of the price column (default 'price')

    Returns
    -------
    pd.DataFrame with two extra columns.
    """
    df = df.copy()
    df["price_bin"] = pd.cut(
        df[price_col], bins=PRICE_BIN_EDGES,
        labels=PRICE_BIN_LABELS, right=True,
    ).astype(str)
    label_to_code = {lbl: i for i, lbl in enumerate(PRICE_BIN_LABELS)}
    df["price_bin_code"] = df["price_bin"].map(label_to_code).fillna(0).astype(int)
    return df


# ── 2. Date features ─────────────────────────────────────────────────────────

def add_date_features(df: pd.DataFrame, date_col: str = "date") -> pd.DataFrame:
    """
    Extract temporal features from a datetime column.

    New columns: day_of_week (0=Mon), day_name, week_of_year,
                 month (int), month_name, is_weekend (0/1), day_of_month.

    Parameters
    ----------
    df       : DataFrame with a datetime column
    date_col : name of the datetime column

    Returns
    -------
    pd.DataFrame with 7 new temporal columns.
    """
    if date_col not in df.columns:
        logger.warning("Column '%s' not found; skipping date features.", date_col)
        return df
    df = df.copy()
    dt = df[date_col]
    df["day_of_week"]  = dt.dt.dayofweek
    df["day_name"]     = dt.dt.day_name()
    df["week_of_year"] = dt.dt.isocalendar().week.astype(int)
    df["month"]        = dt.dt.month
    df["month_name"]   = dt.dt.month_name()
    df["is_weekend"]   = (dt.dt.dayofweek >= 5).astype(int)
    df["day_of_month"] = dt.dt.day
    return df


# ── 3. Revenue per unit ──────────────────────────────────────────────────────

def add_sales_per_unit(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add 'revenue_per_unit' and 'revenue_vs_category_avg'.

    revenue_per_unit        = sales / units_sold  (0 when units=0)
    revenue_vs_category_avg = revenue_per_unit / category mean
                              >1 means above-average for that category.

    Returns
    -------
    pd.DataFrame with two new float columns.
    """
    df = df.copy()
    df["revenue_per_unit"] = np.where(
        df["units_sold"] > 0,
        (df["sales"] / df["units_sold"]).round(4),
        0.0,
    )
    cat_mean = df.groupby("category")["revenue_per_unit"].transform("mean")
    df["revenue_vs_category_avg"] = (
        df["revenue_per_unit"] / cat_mean.replace(0, np.nan)
    ).fillna(1.0).round(4)
    return df


# ── 4. High-value product flag ───────────────────────────────────────────────

def add_high_value_flag(df: pd.DataFrame,
                        rating_threshold: float = 4.0,
                        price_quantile: float = 0.75) -> pd.DataFrame:
    """
    Set 'is_high_value' = 1 when:
      customer_rating >= rating_threshold  AND
      price >= the top price_quantile within its category.

    Parameters
    ----------
    rating_threshold : minimum rating (default 4.0)
    price_quantile   : category-level price quantile cutoff (default 0.75)

    Returns
    -------
    pd.DataFrame with binary int column 'is_high_value'.
    """
    df = df.copy()
    cat_q = (
        df.groupby("category")["price"]
        .quantile(price_quantile)
        .rename("cat_price_threshold")
    )
    df = df.join(cat_q, on="category")
    df["is_high_value"] = (
        (df["customer_rating"] >= rating_threshold) &
        (df["price"] >= df["cat_price_threshold"])
    ).astype(int)
    df.drop(columns=["cat_price_threshold"], inplace=True)
    return df


# ── 5. Category-level aggregate features ────────────────────────────────────

def add_category_stats(df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge five category-level statistics back to every row.

    New columns (prefix 'cat_')
    ---------------------------
    cat_avg_price    : mean price within category
    cat_total_sales  : total revenue within category
    cat_avg_rating   : mean rating within category
    cat_total_units  : total units sold within category
    cat_sales_share  : this category's fraction of overall revenue (0–1)

    Returns
    -------
    pd.DataFrame with five extra float columns.
    """
    df = df.copy()
    stats = df.groupby("category").agg(
        cat_avg_price   = ("price",           "mean"),
        cat_total_sales = ("sales",            "sum"),
        cat_avg_rating  = ("customer_rating",  "mean"),
        cat_total_units = ("units_sold",        "sum"),
    ).round(4)
    stats["cat_sales_share"] = (
        stats["cat_total_sales"] / stats["cat_total_sales"].sum()
    ).round(4)
    df = df.join(stats, on="category")
    return df


# ── 6. Categorical encoding ──────────────────────────────────────────────────

def encode_categoricals(
    df: pd.DataFrame,
    columns: Optional[list[str]] = None,
    method: str = "label",
) -> tuple[pd.DataFrame, dict[str, LabelEncoder]]:
    """
    Encode categorical string columns as integers.

    Parameters
    ----------
    df      : feature DataFrame
    columns : columns to encode (defaults to ["category","price_bin"])
    method  : "label" → LabelEncoder (single int column per feature)
               "onehot" → pd.get_dummies (multiple binary columns)

    Returns
    -------
    (df_encoded, encoders)
    encoders : dict column → fitted LabelEncoder (label mode only)
    """
    if method not in {"label", "onehot"}:
        raise ValueError(f"method must be 'label' or 'onehot'. Got: {method!r}")
    if columns is None:
        columns = [c for c in ["category", "price_bin"] if c in df.columns]

    df = df.copy()
    encoders: dict[str, LabelEncoder] = {}

    if method == "label":
        for col in columns:
            if col not in df.columns:
                continue
            le = LabelEncoder()
            df[col + "_enc"] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
    else:
        df = pd.get_dummies(df, columns=columns, prefix=columns, drop_first=False)

    return df, encoders


# ── 7. Feature scaling ───────────────────────────────────────────────────────

def scale_features(
    df: pd.DataFrame,
    columns: list[str],
) -> tuple[pd.DataFrame, StandardScaler]:
    """
    Apply StandardScaler (zero mean, unit variance) to specified columns.

    Parameters
    ----------
    df      : feature DataFrame
    columns : numeric columns to scale

    Returns
    -------
    (df_scaled, fitted_scaler)
    Save the scaler for use at inference time.
    """
    df = df.copy()
    existing = [c for c in columns if c in df.columns]
    if not existing:
        return df, StandardScaler()
    scaler = StandardScaler()
    df[existing] = scaler.fit_transform(df[existing])
    return df, scaler


# ── 8. Master pipeline ───────────────────────────────────────────────────────

def build_feature_matrix(
    df: pd.DataFrame,
    encode_method: str = "label",
    scale: bool = False,
    scale_cols: Optional[list[str]] = None,
) -> tuple[pd.DataFrame, dict]:
    """
    Run all feature-engineering steps in order and return a model-ready
    DataFrame plus a dict of fitted artefacts.

    Steps
    -----
    1. add_price_bins
    2. add_date_features
    3. add_sales_per_unit
    4. add_high_value_flag
    5. add_category_stats
    6. encode_categoricals
    7. scale_features  (optional)

    Parameters
    ----------
    df            : clean DataFrame from data_loader.load_clean_data()
    encode_method : "label" or "onehot"
    scale         : StandardScale numeric columns when True
    scale_cols    : columns to scale (sensible default if None)

    Returns
    -------
    df_feat   : feature-engineered DataFrame
    artifacts : {
        "encoders"        : dict of fitted LabelEncoders,
        "scaler"          : fitted StandardScaler or None,
        "feature_columns" : list of final numeric column names,
    }
    """
    artifacts: dict = {"encoders": {}, "scaler": None, "feature_columns": []}

    df = add_price_bins(df)
    df = add_date_features(df)
    df = add_sales_per_unit(df)
    df = add_high_value_flag(df)
    df = add_category_stats(df)
    df, artifacts["encoders"] = encode_categoricals(df, method=encode_method)

    if scale:
        if scale_cols is None:
            scale_cols = ["price","units_sold","customer_rating","sales",
                          "revenue_per_unit","cat_avg_price","cat_total_sales"]
        df, artifacts["scaler"] = scale_features(df, scale_cols)

    artifacts["feature_columns"] = list(df.select_dtypes(include=[np.number]).columns)
    logger.info("build_feature_matrix: %s, %d numeric features",
                df.shape, len(artifacts["feature_columns"]))
    return df, artifacts


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.data_loader import load_clean_data
    df_clean = load_clean_data("data/shop_data.csv", verbose=False)
    df_feat, arts = build_feature_matrix(df_clean)
    print(f"Feature matrix: {df_feat.shape}")
    print(f"Numeric features ({len(arts['feature_columns'])}): {arts['feature_columns'][:6]} …")
