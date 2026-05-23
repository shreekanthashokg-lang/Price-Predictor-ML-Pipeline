"""
models/train_model.py
=====================
ShopSmart — Price Prediction Model

Goal
----
Predict the unit ``price`` of a product from its behavioural and
category features using a RandomForestRegressor inside an sklearn Pipeline.

Pipeline architecture
---------------------
    ColumnTransformer
        StandardScaler   → NUMERIC_FEATURES
        OneHotEncoder    → CATEGORICAL_FEATURES
    ↓
    RandomForestRegressor

Why Random Forest?
  • Handles mixed numerical + categorical features well
  • Robust to outliers (split-based, not distance-based)
  • Built-in feature importances
  • No distributional assumptions on the target

Outputs
-------
    models/model.pkl      – serialised fitted Pipeline (joblib)
    models/metrics.json   – test + CV metrics + config

Usage
-----
    python models/train_model.py
    python models/train_model.py --data data/shop_data.csv
                                  --output models/model.pkl
"""

from __future__ import annotations
import argparse, logging, sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ── Project root on sys.path so we can import from src/ ──────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import load_clean_data
from src.features import add_category_stats, add_date_features, add_price_bins
from src.utils import (ensure_dir, format_metrics, print_section,
                        save_json, save_model, timer)

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s | %(levelname)-8s | %(message)s",
                    datefmt="%H:%M:%S")
logger = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
TARGET       = "price"
RANDOM_STATE = 42
TEST_SIZE    = 0.20
CV_FOLDS     = 5

# Features fed into the model (all produced by feature-engineering functions)
NUMERIC_FEATURES: list[str] = [
    "units_sold",
    "customer_rating",
    "cat_avg_price",      # mean price within category
    "cat_avg_rating",     # mean rating within category
    "cat_total_units",    # total units sold within category
    "cat_sales_share",    # this category's revenue fraction (0–1)
]
CATEGORICAL_FEATURES: list[str] = [
    "category",
    "price_bin",          # coarse tier label: Budget / Mid-Range / Luxury …
]

RF_PARAMS: dict = {
    "n_estimators"     : 200,
    "max_depth"        : 12,
    "min_samples_leaf" : 5,
    "max_features"     : "sqrt",
    "random_state"     : RANDOM_STATE,
    "n_jobs"           : -1,
}


# ── 1. Feature preparation ────────────────────────────────────────────────────

def prepare_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply the same feature-engineering steps as used at inference time.

    We call the same src/features.py functions so training and inference
    are guaranteed to produce identical transformations.

    Steps
    -----
    add_date_features   : extract day/week/month temporal columns
    add_price_bins      : add price_bin label + price_bin_code
    add_category_stats  : merge cat_avg_price, cat_avg_rating, etc.

    Note: StandardScaling and OneHotEncoding happen inside the sklearn
    Pipeline (see build_pipeline) to prevent data leakage during CV.

    Parameters
    ----------
    df : clean DataFrame from load_clean_data()

    Returns
    -------
    pd.DataFrame with all NUMERIC_FEATURES and CATEGORICAL_FEATURES present.
    """
    df = add_date_features(df)
    df = add_price_bins(df)
    df = add_category_stats(df)
    return df


# ── 2. Pipeline ───────────────────────────────────────────────────────────────

def build_pipeline() -> Pipeline:
    """
    Build the unfitted sklearn Pipeline.

    Step 1 — ColumnTransformer
        "num" : StandardScaler on NUMERIC_FEATURES
        "cat" : OneHotEncoder on CATEGORICAL_FEATURES
                  drop='first'         → avoids dummy variable trap
                  handle_unknown='ignore' → safe at inference with new categories
    Step 2 — RandomForestRegressor

    Wrapping in Pipeline prevents leakage: scaler/encoder fit only on
    training folds during cross-validation, never on validation folds.

    Returns
    -------
    sklearn.pipeline.Pipeline  (not yet fitted)
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), NUMERIC_FEATURES),
            ("cat",
             OneHotEncoder(drop="first", handle_unknown="ignore",
                           sparse_output=False),
             CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )
    return Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor",    RandomForestRegressor(**RF_PARAMS)),
    ])


# ── 3. Evaluation ─────────────────────────────────────────────────────────────

def evaluate(pipeline: Pipeline, X: pd.DataFrame,
             y: pd.Series, label: str = "Set") -> dict[str, float]:
    """
    Compute MAE, RMSE, and R² for a fitted pipeline on a given dataset.

    Metrics explained
    -----------------
    MAE  : Mean Absolute Error — avg $ distance from true price. Interpretable.
    RMSE : Root Mean Squared Error — penalises large errors more than MAE.
    R²   : Fraction of variance explained (1.0 = perfect; 0 = no better than mean).

    Parameters
    ----------
    pipeline : fitted sklearn Pipeline
    X        : feature DataFrame
    y        : true target values
    label    : display label for the printed summary

    Returns
    -------
    dict with keys "MAE", "RMSE", "R2"
    """
    y_pred = pipeline.predict(X)
    metrics = {
        "MAE" : float(mean_absolute_error(y, y_pred)),
        "RMSE": float(np.sqrt(mean_squared_error(y, y_pred))),
        "R2"  : float(r2_score(y, y_pred)),
    }
    print_section(f"Metrics — {label}")
    print(format_metrics(metrics, currency_keys=["MAE", "RMSE"]))
    return metrics


def run_cross_validation(pipeline: Pipeline,
                          X: pd.DataFrame,
                          y: pd.Series,
                          cv: int = CV_FOLDS) -> dict[str, float]:
    """
    k-fold cross-validation on the training set to estimate generalisation error.

    Uses neg_root_mean_squared_error scoring (sklearn convention, negated back
    to positive RMSE for display).

    Parameters
    ----------
    pipeline : unfitted Pipeline (fitted fresh per fold internally)
    X        : training features
    y        : training target
    cv       : number of folds (default 5)

    Returns
    -------
    dict with "cv_rmse_mean", "cv_rmse_std", "cv_folds"
    """
    scores = cross_val_score(pipeline, X, y, cv=cv,
                              scoring="neg_root_mean_squared_error", n_jobs=-1)
    rmse = -scores
    result = {
        "cv_rmse_mean": float(rmse.mean()),
        "cv_rmse_std" : float(rmse.std()),
        "cv_folds"    : cv,
    }
    print_section(f"{cv}-Fold Cross-Validation (training set)")
    print(f"  RMSE per fold : {' | '.join(f'${v:.2f}' for v in rmse)}")
    print(f"  Mean RMSE     : ${result['cv_rmse_mean']:.4f}")
    print(f"  Std  RMSE     : ±${result['cv_rmse_std']:.4f}")
    return result


# ── 4. Feature importance ─────────────────────────────────────────────────────

def report_feature_importance(pipeline: Pipeline) -> pd.DataFrame:
    """
    Extract and display Random Forest feature importances (mean decrease Gini).

    The ColumnTransformer prefixes ("num__", "cat__") are stripped for
    clean output.  Returns a DataFrame sorted by importance descending.

    Parameters
    ----------
    pipeline : fitted Pipeline with 'preprocessor' and 'regressor' steps

    Returns
    -------
    pd.DataFrame  columns: ['feature', 'importance']
    """
    names = pipeline.named_steps["preprocessor"].get_feature_names_out()
    imps  = pipeline.named_steps["regressor"].feature_importances_

    df_imp = (
        pd.DataFrame({"feature": names, "importance": imps})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    df_imp["feature"] = (df_imp["feature"]
                         .str.replace("num__", "", regex=False)
                         .str.replace("cat__", "", regex=False))

    print_section("Top 12 Feature Importances")
    print(f"  {'Rank':<5} {'Feature':<35} {'Importance':>10}  Bar")
    print(f"  {'─'*5} {'─'*35} {'─'*10}  {'─'*20}")
    for rank, (_, row) in enumerate(df_imp.head(12).iterrows(), 1):
        bar = "█" * int(row["importance"] * 300)
        print(f"  {rank:<5} {row['feature']:<35} {row['importance']:>10.4f}  {bar}")

    return df_imp


# ── 5. Inference demo ─────────────────────────────────────────────────────────

def inference_demo(pipeline: Pipeline, X_test: pd.DataFrame,
                   y_test: pd.Series, n: int = 8) -> None:
    """
    Side-by-side table of actual vs predicted prices for the first n samples.

    A quick visual sanity check that the model output looks reasonable.
    """
    print_section(f"Inference Demo — first {n} test samples")
    Xs = X_test.head(n); ys = y_test.head(n).values
    preds = pipeline.predict(Xs); errs = preds - ys
    print(f"  {'#':<4} {'Actual $':>10} {'Predicted $':>12} {'Error $':>10}  {'Within $5':>9}")
    print(f"  {'─'*4} {'─'*10} {'─'*12} {'─'*10}  {'─'*9}")
    for i, (act, pred, err) in enumerate(zip(ys, preds, errs), 1):
        print(f"  {i:<4} ${act:>9.2f} ${pred:>11.2f} ${err:>+9.2f}  "
              f"{'✓' if abs(err)<=5 else '✗':>9}")


# ── 6. Main training pipeline ─────────────────────────────────────────────────

def main(data_path   : str = "data/shop_data.csv",
         output_path : str = "models/model.pkl",
         metrics_path: str = "models/metrics.json") -> None:
    """
    End-to-end training run in 6 numbered steps.

    Steps
    -----
    1  Load & clean data
    2  Engineer features
    3  Train / test split
    4  Cross-validate (training set only)
    5  Fit final model; evaluate on held-out test set
    6  Save model + metrics to disk

    Parameters
    ----------
    data_path    : path to shop_data.csv
    output_path  : destination .pkl for the fitted Pipeline
    metrics_path : destination .json for evaluation metrics
    """
    print_section("ShopSmart — Price Prediction Model Training")

    # ── Step 1 ────────────────────────────────────────────────────────────────
    logger.info("[1/6] Loading & cleaning data …")
    with timer("Data loading"):
        df = load_clean_data(data_path, verbose=False)
    logger.info("  Clean shape: %d rows × %d columns", *df.shape)

    # ── Step 2 ────────────────────────────────────────────────────────────────
    logger.info("[2/6] Engineering features …")
    with timer("Feature engineering"):
        df = prepare_features(df)

    missing = [c for c in NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]
               if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns after feature engineering: {missing}")

    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]
    logger.info("  Feature matrix: %d rows × %d features | target='%s'",
                len(X), len(X.columns), TARGET)

    # ── Step 3 ────────────────────────────────────────────────────────────────
    logger.info("[3/6] Train/test split (%.0f%% test) …", TEST_SIZE*100)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=df.loc[X.index, "price_bin"])
    logger.info("  Train: %d | Test: %d", len(X_train), len(X_test))

    # ── Step 4 ────────────────────────────────────────────────────────────────
    logger.info("[4/6] Cross-validation …")
    with timer("Cross-validation"):
        cv_metrics = run_cross_validation(build_pipeline(), X_train, y_train)

    # ── Step 5 ────────────────────────────────────────────────────────────────
    logger.info("[5/6] Fitting final model …")
    final_pipe = build_pipeline()
    with timer("Final model training"):
        final_pipe.fit(X_train, y_train)

    test_metrics  = evaluate(final_pipe, X_test,  y_test,  "Test set")
    train_metrics = evaluate(final_pipe, X_train, y_train, "Training set")

    gap = test_metrics["RMSE"] - train_metrics["RMSE"]
    if gap > 10:
        logger.warning("Possible overfit — test exceeds train RMSE by $%.2f", gap)
    else:
        logger.info("Train–test RMSE gap: $%.2f — OK", abs(gap))

    imp_df = report_feature_importance(final_pipe)
    inference_demo(final_pipe, X_test, y_test)

    # ── Step 6 ────────────────────────────────────────────────────────────────
    logger.info("[6/6] Saving artefacts …")
    ensure_dir(Path(output_path).parent)
    save_model(final_pipe, output_path, compress=3)

    save_json({
        "test"            : test_metrics,
        "train"           : train_metrics,
        "cross_validation": cv_metrics,
        "config"          : {
            "target"              : TARGET,
            "n_train"             : len(X_train),
            "n_test"              : len(X_test),
            "test_size"           : TEST_SIZE,
            "random_state"        : RANDOM_STATE,
            "numeric_features"    : NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
            "rf_params"           : RF_PARAMS,
        },
        "top_10_features" : imp_df.head(10).to_dict(orient="records"),
    }, metrics_path)

    print_section("Training Complete ✓")
    print(f"  Model   → {output_path}")
    print(f"  Metrics → {metrics_path}")
    print(f"\n  Test MAE  : ${test_metrics['MAE']:.2f}")
    print(f"  Test RMSE : ${test_metrics['RMSE']:.2f}")
    print(f"  Test R²   : {test_metrics['R2']:.4f}  "
          f"({'good' if test_metrics['R2'] > 0.7 else 'needs improvement'})")
    print(f"  CV RMSE   : ${cv_metrics['cv_rmse_mean']:.2f} ± "
          f"${cv_metrics['cv_rmse_std']:.2f}  ({CV_FOLDS}-fold)")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train ShopSmart price prediction model",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data",    default="data/shop_data.csv")
    parser.add_argument("--output",  default="models/model.pkl")
    parser.add_argument("--metrics", default="models/metrics.json")
    args = parser.parse_args()
    main(args.data, args.output, args.metrics)
