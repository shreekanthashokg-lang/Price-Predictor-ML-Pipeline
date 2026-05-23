"""
src/utils.py
============
ShopSmart — General-Purpose Utilities

Sections
--------
1.  Logging setup
2.  Model save / load  (joblib)
3.  Timer context manager
4.  Directory helpers
5.  Pretty printing
6.  Metric formatting
7.  Type-checking guards
8.  JSON helpers
"""

from __future__ import annotations
import json, logging, os, time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd


# ── 1. Logging ────────────────────────────────────────────────────────────────

def setup_logging(
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    fmt: str = "%(asctime)s | %(levelname)-8s | %(name)s — %(message)s",
    datefmt: str = "%Y-%m-%d %H:%M:%S",
) -> logging.Logger:
    """
    Configure the root logger with a StreamHandler (+ optional FileHandler).

    Call once at the top of any script or notebook; all subsequent
    ``logging.getLogger(__name__)`` calls inherit this configuration.

    Parameters
    ----------
    level    : logging level (default logging.INFO)
    log_file : optional path for a FileHandler (parent dirs auto-created)
    fmt      : log record format string
    datefmt  : datetime format inside each log line

    Returns
    -------
    logging.Logger  (root logger, configured and ready)

    Example
    -------
    >>> logger = setup_logging(logging.DEBUG, log_file="logs/run.log")
    >>> logger.info("Started.")
    """
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file:
        os.makedirs(Path(log_file).parent, exist_ok=True)
        handlers.append(logging.FileHandler(log_file, mode="a", encoding="utf-8"))
    logging.basicConfig(level=level, format=fmt, datefmt=datefmt,
                        handlers=handlers, force=True)
    root = logging.getLogger()
    root.info("Logging configured — level: %s", logging.getLevelName(level))
    return root


# ── 2. Model persistence ──────────────────────────────────────────────────────

def save_model(model: Any, path: str, compress: int = 3) -> None:
    """
    Serialise a trained model to disk with joblib.

    joblib outperforms pickle for sklearn objects containing large NumPy
    arrays (faster + smaller files via memory-mapped I/O).

    Parameters
    ----------
    model    : any Python object (sklearn Pipeline, estimator, dict …)
    path     : destination .pkl file path; parent dirs auto-created
    compress : joblib compression level 0–9 (default 3 = good balance)

    Example
    -------
    >>> save_model(fitted_pipeline, "models/price_model.pkl")
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, p, compress=compress)
    size_kb = p.stat().st_size / 1024
    logging.getLogger(__name__).info(
        "Model saved → %s  (%.1f KB, compress=%d)", p, size_kb, compress)


def load_model(path: str) -> Any:
    """
    Load a joblib-serialised model from disk.

    Parameters
    ----------
    path : path to the .pkl file created by save_model()

    Returns
    -------
    The deserialised object (e.g. a fitted sklearn Pipeline).

    Raises
    ------
    FileNotFoundError  if the file does not exist.

    Example
    -------
    >>> pipeline = load_model("models/price_model.pkl")
    >>> preds = pipeline.predict(X_new)
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(
            f"Model file not found: {p}\n"
            "Run models/train_model.py first to create it.")
    model = joblib.load(p)
    logging.getLogger(__name__).info("Model loaded ← %s", p)
    return model


# ── 3. Timer context manager ──────────────────────────────────────────────────

@contextmanager
def timer(label: str = "Block"):
    """
    Context manager that measures and logs wall-clock execution time.

    Uses time.perf_counter() for sub-millisecond precision.

    Parameters
    ----------
    label : description of the timed block (shown in the log message)

    Example
    -------
    >>> with timer("Model training"):
    ...     pipeline.fit(X_train, y_train)
    # logs: Model training — completed in 3.42 s
    """
    log = logging.getLogger(__name__)
    log.info("%s — started …", label)
    t0 = time.perf_counter()
    try:
        yield
    finally:
        log.info("%s — completed in %.3f s", label, time.perf_counter() - t0)


# ── 4. Directory helpers ──────────────────────────────────────────────────────

def ensure_dir(path: str | Path) -> Path:
    """
    Create ``path`` (and all parents) if it does not exist.

    Idempotent — safe to call even when the directory already exists.
    Returns the Path so the function can be used inline.

    Example
    -------
    >>> fig.savefig(ensure_dir("reports/figures") / "chart.png")
    """
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def list_files(directory: str | Path, suffix: str = "") -> list[Path]:
    """
    Sorted list of files in ``directory`` optionally filtered by suffix.

    Parameters
    ----------
    directory : directory to scan
    suffix    : e.g. ".csv", ".pkl" (empty = all files)

    Returns
    -------
    list[Path]  — empty list if directory does not exist.
    """
    p = Path(directory)
    if not p.is_dir():
        return []
    pat = f"*{suffix}" if suffix else "*"
    return sorted(f for f in p.glob(pat) if f.is_file())


# ── 5. Pretty printing ────────────────────────────────────────────────────────

def print_section(title: str, width: int = 60) -> None:
    """
    Print a bold ASCII section header to stdout.

    Example
    -------
    >>> print_section("Model Evaluation")
    ============================================================
      MODEL EVALUATION
    ============================================================
    """
    bar = "=" * width
    print(f"\n{bar}\n  {title.upper()}\n{bar}")


def print_dict(d: dict, indent: int = 2) -> None:
    """
    Pretty-print a (possibly nested) dict.

    Floats are rounded to 4 decimal places; nested dicts are indented.

    Parameters
    ----------
    d      : dict to print
    indent : leading spaces per nesting level

    Example
    -------
    >>> print_dict({"mae": 12.3456, "config": {"n": 200}})
      mae                   : 12.3456
      config                :
        n                   : 200
    """
    pad = " " * indent
    for k, v in d.items():
        if isinstance(v, dict):
            print(f"{pad}{k}:")
            print_dict(v, indent + 2)
        elif isinstance(v, float):
            print(f"{pad}{k:<22}: {v:.4f}")
        else:
            print(f"{pad}{k:<22}: {v}")


def summarise_dataframe(df: pd.DataFrame, label: str = "DataFrame") -> None:
    """
    Print a concise structural summary of a DataFrame to stdout.

    Shows: shape, memory, null counts per column, dtypes, and descriptive
    statistics for the first six numeric columns.

    Parameters
    ----------
    df    : any pandas DataFrame
    label : display name for the section header
    """
    print_section(f"{label} — Summary")
    print(f"  Shape        : {df.shape[0]:,} rows × {df.shape[1]} columns")
    print(f"  Memory       : {df.memory_usage(deep=True).sum()/1024:.1f} KB")
    print(f"  Total nulls  : {df.isnull().sum().sum()}")
    print(f"\n  {'Column':<28} {'Dtype':<12} {'Nulls':>6}")
    print(f"  {'─'*28} {'─'*12} {'─'*6}")
    for col in df.columns:
        nulls = int(df[col].isnull().sum())
        flag  = " ⚠" if nulls else ""
        print(f"  {col:<28} {str(df[col].dtype):<12} {nulls:>6}{flag}")
    num = df.select_dtypes(include=[np.number]).columns.tolist()
    if num:
        print(f"\n  Numeric statistics (first 6 columns):")
        print(df[num[:6]].describe().round(3).to_string())


# ── 6. Metric formatting ──────────────────────────────────────────────────────

def format_metrics(
    metrics: dict[str, float],
    currency_keys: Optional[list[str]] = None,
) -> str:
    """
    Format a flat dict of metric → float as a readable multi-line string.

    Keys in ``currency_keys`` are prefixed with "$".

    Parameters
    ----------
    metrics       : {"MAE": 12.45, "RMSE": 19.02, "R2": 0.78}
    currency_keys : optional list of keys to render as dollar amounts

    Returns
    -------
    str  multi-line text, ready for print() or logging.

    Example
    -------
    >>> print(format_metrics({"MAE": 12.45, "R2": 0.78},
    ...                       currency_keys=["MAE"]))
      MAE                  :  $12.4500
      R2                   :  0.7800
    """
    ck = currency_keys or []
    lines = []
    for name, val in metrics.items():
        if isinstance(val, float):
            lines.append(f"  {name:<22}: {'$' if name in ck else ' '}{val:.4f}")
        else:
            lines.append(f"  {name:<22}:  {val}")
    return "\n".join(lines)


# ── 7. Type-checking guards ───────────────────────────────────────────────────

def require_columns(df: pd.DataFrame,
                    columns: list[str],
                    caller: str = "") -> None:
    """
    Raise ValueError if any of ``columns`` are missing from df.

    Produces a clear, actionable message instead of a cryptic KeyError.

    Parameters
    ----------
    df      : DataFrame to inspect
    columns : required column names
    caller  : optional function name to include in the error message
    """
    missing = [c for c in columns if c not in df.columns]
    if missing:
        where = f" (called from {caller})" if caller else ""
        raise ValueError(
            f"Missing required columns{where}: {missing}\n"
            f"Available: {sorted(df.columns.tolist())}")


def require_no_nulls(df: pd.DataFrame,
                     columns: list[str],
                     caller: str = "") -> None:
    """
    Raise ValueError if any of the specified columns still contain NaN.

    Use after cleaning to confirm that imputation worked correctly.
    """
    bad = {c: int(df[c].isnull().sum())
           for c in columns if c in df.columns and df[c].isnull().any()}
    if bad:
        where = f" (called from {caller})" if caller else ""
        raise ValueError(
            f"Unexpected nulls detected{where}:\n"
            + "\n".join(f"  {c}: {n} null(s)" for c, n in bad.items()))


def require_numeric(df: pd.DataFrame, columns: list[str]) -> None:
    """Raise TypeError if any listed columns are non-numeric dtype."""
    bad = [c for c in columns
           if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
    if bad:
        raise TypeError(
            f"Expected numeric columns but got non-numeric: {bad}\n"
            f"Dtypes: { {c: str(df[c].dtype) for c in bad} }")


# ── 8. JSON helpers ───────────────────────────────────────────────────────────

def save_json(obj: Any, path: str, indent: int = 2) -> None:
    """
    Save a JSON-serialisable object to disk.

    numpy integers, floats, and arrays are auto-converted to Python natives
    so the object serialises cleanly without manual type conversion.

    Parameters
    ----------
    obj    : JSON-serialisable object (dict, list, str, numbers)
    path   : destination file; parent dirs auto-created
    indent : JSON indentation width (default 2)

    Example
    -------
    >>> save_json({"MAE": 12.45, "N": np.int64(800)}, "reports/metrics.json")
    """
    def _default(o: Any) -> Any:
        if isinstance(o, np.integer): return int(o)
        if isinstance(o, np.floating): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        if isinstance(o, pd.Timestamp): return o.isoformat()
        raise TypeError(f"{type(o).__name__} is not JSON serialisable")

    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=indent, default=_default)
    logging.getLogger(__name__).info("JSON saved → %s  (%d bytes)", p, p.stat().st_size)


def load_json(path: str) -> Any:
    """
    Load and return the contents of a JSON file.

    Raises
    ------
    FileNotFoundError  if the file does not exist.
    """
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"JSON file not found: {p}")
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    log = setup_logging(logging.DEBUG)
    print_section("utils self-test")

    with timer("Sleep 0.1 s"):
        time.sleep(0.1)

    td = ensure_dir("/tmp/shopsmart_utils_test")
    assert td.is_dir()
    print("✓  ensure_dir OK")

    sample = {"MAE": 12.3456, "N": np.int64(800), "arr": np.array([1,2,3])}
    save_json(sample, "/tmp/shopsmart_utils_test/test.json")
    loaded = load_json("/tmp/shopsmart_utils_test/test.json")
    assert abs(loaded["MAE"] - 12.3456) < 1e-6
    assert loaded["arr"] == [1, 2, 3]
    print("✓  JSON round-trip OK")

    print("\nformat_metrics:")
    print(format_metrics({"MAE":12.3456,"RMSE":19.876,"R2":0.8312},
                          currency_keys=["MAE","RMSE"]))

    df_t = pd.DataFrame({"price":[1.0],"sales":[5.0]})
    try:
        require_columns(df_t, ["price","units_sold"], caller="self-test")
    except ValueError as e:
        print(f"✓  require_columns guard OK: {str(e)[:60]}…")

    print_section("All utils tests passed ✓")
