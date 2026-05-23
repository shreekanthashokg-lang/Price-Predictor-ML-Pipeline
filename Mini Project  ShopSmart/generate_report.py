"""
reports/generate_report.py
==========================
ShopSmart — Automated Sales Report Generator

Reads ``data/shop_data.csv``, computes all business metrics, and writes
a structured plain-text report to ``reports/sales_report.txt``.
The same content is printed to stdout so it works in CI/cron pipelines.

With ``--charts`` the script also calls src/visualize.py and saves PNGs
to ``reports/figures/``.

Usage
-----
    python reports/generate_report.py
    python reports/generate_report.py --data data/shop_data.csv
                                       --output reports/sales_report.txt
    python reports/generate_report.py --charts

Outputs
-------
    reports/sales_report.txt   plain-text report
    reports/figures/*.png      charts (with --charts only)
"""

from __future__ import annotations
import argparse, logging, sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.data_loader import get_category_summary, get_date_range, load_clean_data
from src.utils import ensure_dir, print_section, setup_logging

logger = logging.getLogger(__name__)

DEFAULT_DATA   = "data/shop_data.csv"
DEFAULT_OUTPUT = "reports/sales_report.txt"
FIGURES_DIR    = "reports/figures"
TOP_N          = 10


# ═══════════════════════════════════════════════════════════════════════════════
# 1. Aggregation helpers
# ═══════════════════════════════════════════════════════════════════════════════

def overall_kpis(df: pd.DataFrame) -> dict:
    """
    Compute top-level Key Performance Indicators for the full dataset.

    KPIs
    ----
    total_revenue      sum of all sales
    total_units        sum of all units_sold
    n_transactions     number of rows (one row = one transaction)
    avg_order_value    mean revenue per transaction
    avg_rating         mean customer_rating
    median_price       median unit price
    pct_rated_above_4  % of transactions with customer_rating >= 4.0
    date_start         earliest date in dataset
    date_end           latest date in dataset

    Returns
    -------
    dict of KPI name → value
    """
    d_start, d_end = get_date_range(df)
    return {
        "total_revenue"     : round(float(df["sales"].sum()), 2),
        "total_units"       : int(df["units_sold"].sum()),
        "n_transactions"    : len(df),
        "avg_order_value"   : round(float(df["sales"].mean()), 2),
        "avg_rating"        : round(float(df["customer_rating"].mean()), 2),
        "median_price"      : round(float(df["price"].median()), 2),
        "pct_rated_above_4" : round(float((df["customer_rating"] >= 4.0).mean() * 100), 1),
        "date_start" : d_start.strftime("%d %b %Y") if pd.notna(d_start) else "N/A",
        "date_end"   : d_end.strftime("%d %b %Y")   if pd.notna(d_end)   else "N/A",
    }


def top_products(df: pd.DataFrame, n: int = TOP_N) -> pd.DataFrame:
    """
    Top-N products by total revenue with supporting metrics.

    Returns
    -------
    pd.DataFrame  columns: product_name, category, total_revenue,
                           total_units, avg_price, avg_rating
    Sorted by total_revenue descending.
    """
    return (
        df.groupby(["product_name", "category"])
        .agg(total_revenue=("sales","sum"), total_units=("units_sold","sum"),
             avg_price=("price","mean"),    avg_rating=("customer_rating","mean"))
        .round(2)
        .sort_values("total_revenue", ascending=False)
        .head(n)
        .reset_index()
    )


def daily_revenue(df: pd.DataFrame) -> pd.DataFrame:
    """
    Total sales per date with cumulative column.

    Returns
    -------
    pd.DataFrame  columns: date, daily_revenue, cumulative_revenue
    """
    daily = (df.groupby("date")["sales"].sum()
               .reset_index().rename(columns={"sales":"daily_revenue"})
               .sort_values("date").reset_index(drop=True))
    daily["cumulative_revenue"] = daily["daily_revenue"].cumsum().round(2)
    return daily


def weekday_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Revenue and units by day-of-week name, with revenue share %.

    Returns
    -------
    pd.DataFrame  indexed by day name (Monday → Sunday).
    Columns: total_revenue, total_units, revenue_share_pct, is_weekend
    """
    tmp = df.copy()
    tmp["day_name"]  = tmp["date"].dt.day_name()
    tmp["is_weekend"] = tmp["date"].dt.dayofweek >= 5
    s = (tmp.groupby("day_name")
            .agg(total_revenue=("sales","sum"), total_units=("units_sold","sum"),
                 is_weekend=("is_weekend","first"))
            .round(2))
    s["revenue_share_pct"] = (s["total_revenue"] / s["total_revenue"].sum() * 100).round(1)
    dow = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    return s.reindex([d for d in dow if d in s.index])


def price_band_summary(df: pd.DataFrame) -> pd.DataFrame:
    """
    Revenue, units, and avg rating broken down by price tier.

    Bands: Budget($0-15) | Economy($15-30) | Mid-Range($30-60)
           Premium($60-100) | Luxury($100-200) | Ultra-Luxury(>$200)

    Returns
    -------
    pd.DataFrame  columns: price_band, total_revenue, pct_revenue,
                           total_units, avg_rating
    """
    edges  = [0, 15, 30, 60, 100, 200, np.inf]
    labels = ["Budget","Economy","Mid-Range","Premium","Luxury","Ultra-Luxury"]
    tmp = df.copy()
    tmp["price_band"] = pd.cut(tmp["price"], bins=edges, labels=labels)
    s = (tmp.groupby("price_band", observed=True)
            .agg(total_revenue=("sales","sum"), total_units=("units_sold","sum"),
                 avg_rating=("customer_rating","mean"))
            .round(2))
    s["pct_revenue"] = (s["total_revenue"] / s["total_revenue"].sum() * 100).round(1)
    return s.reset_index()


def rating_distribution(df: pd.DataFrame) -> pd.DataFrame:
    """
    Count transactions per star-rating band.

    Bands: Excellent(≥4.5) | Good(4.0-4.5) | Average(3.5-4.0) | Below Avg(<3.5)

    Returns
    -------
    pd.DataFrame  columns: band, count, pct
    """
    bins   = [0, 3.5, 4.0, 4.5, 5.01]
    labels = ["Below Avg (<3.5)","Average (3.5–4.0)",
              "Good (4.0–4.5)","Excellent (4.5–5.0)"]
    tmp = df.copy()
    tmp["rb"] = pd.cut(tmp["customer_rating"], bins=bins, labels=labels, right=False)
    counts = tmp["rb"].value_counts().reindex(labels[::-1])
    return pd.DataFrame({
        "band" : counts.index,
        "count": counts.values,
        "pct"  : (counts / counts.sum() * 100).round(1).values,
    })


# ═══════════════════════════════════════════════════════════════════════════════
# 2. Report rendering
# ═══════════════════════════════════════════════════════════════════════════════

def _d(char="─", w=62): return char * w
def _c(v): return f"${v:,.2f}"


def render_report(kpis, cat_summary, top_prods, daily,
                  weekday_sum, price_bands, rating_dist, generated_at) -> str:
    """
    Render all computed aggregates into a single plain-text report string.

    Each section is delimited by ASCII divider lines. The report is suitable
    for terminal display, plain-text files, or email delivery.

    Parameters
    ----------
    kpis         : dict from overall_kpis()
    cat_summary  : DataFrame from get_category_summary()
    top_prods    : DataFrame from top_products()
    daily        : DataFrame from daily_revenue()
    weekday_sum  : DataFrame from weekday_summary()
    price_bands  : DataFrame from price_band_summary()
    rating_dist  : DataFrame from rating_distribution()
    generated_at : ISO timestamp string

    Returns
    -------
    str  — complete formatted report text.
    """
    L = []
    def w(*lines): L.extend(lines)

    # Header
    w("="*62, "  SHOPSMART — MONTHLY SALES ANALYSIS REPORT", "="*62,
      f"  Period       : {kpis['date_start']}  →  {kpis['date_end']}",
      f"  Generated    : {generated_at}",
      f"  Transactions : {kpis['n_transactions']:,}",
      "="*62, "")

    # 1. KPIs
    w(_d(), "  1. EXECUTIVE KEY PERFORMANCE INDICATORS", _d(),
      f"  Total Revenue          : {_c(kpis['total_revenue'])}",
      f"  Total Units Sold       : {kpis['total_units']:,}",
      f"  Avg Order Value        : {_c(kpis['avg_order_value'])}",
      f"  Median Product Price   : {_c(kpis['median_price'])}",
      f"  Avg Customer Rating    : {kpis['avg_rating']} / 5.0",
      f"  Products Rated >= 4.0  : {kpis['pct_rated_above_4']}%", "")

    # 2. Category
    w(_d(), "  2. REVENUE BY CATEGORY", _d(),
      f"  {'Category':<20} {'Revenue':>10}  {'Rev%':>5}  {'Units':>6}  {'Avg$':>8}  {'Rating':>7}",
      f"  {'─'*20} {'─'*10}  {'─'*5}  {'─'*6}  {'─'*8}  {'─'*7}")
    total_rev = cat_summary["total_sales"].sum()
    for _, row in cat_summary.iterrows():
        share = row["total_sales"] / total_rev * 100 if total_rev else 0
        w(f"  {row['category']:<20} {_c(row['total_sales']):>10}  "
          f"{share:>4.1f}%  {int(row['total_units']):>6}  "
          f"{_c(row['avg_price']):>8}  {row['avg_rating']:>7.1f}")
    w(f"  {'─'*20} {'─'*10}",
      f"  {'TOTAL':<20} {_c(total_rev):>10}  100.0%", "")

    # 3. Top products
    w(_d(), f"  3. TOP {len(top_prods)} PRODUCTS BY REVENUE", _d(),
      f"  {'#':<3}  {'Product':<26}  {'Category':<15}  {'Revenue':>9}  {'Units':>5}  {'Rating':>7}",
      f"  {'─'*3}  {'─'*26}  {'─'*15}  {'─'*9}  {'─'*5}  {'─'*7}")
    for rank, (_, row) in enumerate(top_prods.iterrows(), 1):
        name = row["product_name"][:24] + "…" if len(row["product_name"]) > 24 else row["product_name"]
        w(f"  {rank:<3}  {name:<26}  {row['category']:<15}  "
          f"{_c(row['total_revenue']):>9}  {int(row['total_units']):>5}  "
          f"{row['avg_rating']:>7.1f}")
    w("")

    # 4. Daily revenue (first 15 days)
    w(_d(), "  4. DAILY REVENUE  (first 15 days shown)", _d(),
      f"  {'Date':<14}  {'Daily Revenue':>15}  {'Cumulative':>15}",
      f"  {'─'*14}  {'─'*15}  {'─'*15}")
    for _, row in daily.head(15).iterrows():
        ds = row["date"].strftime("%d %b %Y") if pd.notna(row["date"]) else "N/A"
        w(f"  {ds:<14}  {_c(row['daily_revenue']):>15}  {_c(row['cumulative_revenue']):>15}")
    if len(daily) > 15:
        w(f"  … ({len(daily)-15} more days not shown)")
    w("")

    # 5. Day-of-week
    w(_d(), "  5. REVENUE BY DAY OF WEEK", _d(),
      f"  {'Day':<12}  {'Revenue':>10}  {'Share':>6}  {'Units':>6}  Bar",
      f"  {'─'*12}  {'─'*10}  {'─'*6}  {'─'*6}  {'─'*22}")
    peak_day = weekday_sum["total_revenue"].idxmax()
    for day, row in weekday_sum.iterrows():
        bar  = "█" * int(row["revenue_share_pct"] / 1.2)
        flag = " ← PEAK" if day == peak_day else ""
        w(f"  {day:<12}  {_c(row['total_revenue']):>10}  "
          f"{row['revenue_share_pct']:>5.1f}%  {int(row['total_units']):>6}  {bar}{flag}")
    w("")

    # 6. Price bands
    w(_d(), "  6. REVENUE BY PRICE BAND", _d(),
      f"  {'Band':<14}  {'Revenue':>10}  {'Rev%':>5}  {'Units':>6}  {'Avg Rating':>10}",
      f"  {'─'*14}  {'─'*10}  {'─'*5}  {'─'*6}  {'─'*10}")
    for _, row in price_bands.iterrows():
        w(f"  {str(row['price_band']):<14}  {_c(row['total_revenue']):>10}  "
          f"{row['pct_revenue']:>4.1f}%  {int(row['total_units']):>6}  "
          f"{row['avg_rating']:>10.2f}")
    w("")

    # 7. Rating distribution
    w(_d(), "  7. CUSTOMER RATING DISTRIBUTION", _d(),
      f"  {'Band':<25}  {'Count':>7}  {'Pct':>6}  Chart",
      f"  {'─'*25}  {'─'*7}  {'─'*6}  {'─'*18}")
    for _, row in rating_dist.iterrows():
        stars = "★" * int(row["pct"] / 4)
        w(f"  {row['band']:<25}  {row['count']:>7,}  {row['pct']:>5.1f}%  {stars}")
    w("")

    # 8. Automated insights
    best_cat   = cat_summary.iloc[0]["category"]
    worst_cat  = cat_summary.iloc[-1]["category"]
    best_prod  = top_prods.iloc[0]["product_name"]
    wknd_rev   = weekday_sum.loc[
        weekday_sum.index.isin(["Saturday","Sunday"]), "total_revenue"].sum()
    wknd_share = wknd_rev / weekday_sum["total_revenue"].sum() * 100

    w(_d(), "  8. AUTOMATED INSIGHTS", _d(),
      f"  • Best category        : {best_cat}",
      f"  • Lowest category      : {worst_cat}  (growth opportunity)",
      f"  • Top product          : {best_prod}",
      f"  • Peak day             : {peak_day}  "
      f"(${weekday_sum.loc[peak_day, 'total_revenue']:,.0f})",
      f"  • Satisfaction         : {kpis['pct_rated_above_4']}% rated >= 4.0",
      f"  • Weekend revenue share: {wknd_share:.1f}%  "
      f"({'below expected 28.6% — opportunity' if wknd_share < 28 else 'on target'})",
      "")

    # 9. Recommendations
    w(_d(), "  9. RECOMMENDATIONS", _d(),
      "  [HIGH]  Investigate + replicate Week 3 mid-month spike.",
      "  [HIGH]  Weekend flash sales on Clothing + Toys to close revenue gap.",
      "  [MED]   Books cross-sell using high ratings (4.3) as trust anchor.",
      "  [MED]   Sports category SKU quality audit — reduce rating variance.",
      f"  [LOW]   {peak_day} email blast (send the evening before).",
      "  [LOW]   Review the 7.6% of products rated below 3.5 stars.",
      "")

    # Footer
    w("="*62, "  End of Report — ShopSmart Analytics Pipeline v1.0",
      f"  {generated_at}", "="*62)

    return "\n".join(L)


# ═══════════════════════════════════════════════════════════════════════════════
# 3. Chart generation (optional)
# ═══════════════════════════════════════════════════════════════════════════════

def generate_charts(df: pd.DataFrame, figures_dir: str) -> None:
    """
    Generate and save all standard charts to ``figures_dir``.

    Uses matplotlib Agg backend (headless) so it works in CI/server
    environments without a display. Imports visualize lazily so the
    report can be run in text-only mode without matplotlib.

    Charts saved
    ------------
    dashboard.png, category_revenue.png, daily_trend.png,
    top_products.png, price_distribution.png, rating_vs_units.png,
    sales_heatmap.png, category_share.png

    Parameters
    ----------
    df          : clean DataFrame
    figures_dir : output directory for PNG files
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from src.visualize import (
            plot_category_share, plot_dashboard, plot_price_distribution,
            plot_rating_vs_units, plot_sales_by_category,
            plot_sales_heatmap, plot_sales_trend, plot_top_products,
        )
    except ImportError as e:
        logger.warning("Chart generation skipped — %s", e)
        return

    ensure_dir(figures_dir)

    # Dashboard (all panels in one figure)
    fig = plot_dashboard(df, save_path=f"{figures_dir}/dashboard.png")
    plt.close(fig)

    # Individual charts
    chart_specs = [
        ("category_revenue",  lambda: plot_sales_by_category(df, save_path=f"{figures_dir}/category_revenue.png")),
        ("daily_trend",       lambda: plot_sales_trend(df, freq="D", save_path=f"{figures_dir}/daily_trend.png")),
        ("top_products",      lambda: plot_top_products(df, top_n=10, save_path=f"{figures_dir}/top_products.png")),
        ("price_distribution",lambda: plot_price_distribution(df, save_path=f"{figures_dir}/price_distribution.png")),
        ("rating_vs_units",   lambda: plot_rating_vs_units(df, save_path=f"{figures_dir}/rating_vs_units.png")),
        ("sales_heatmap",     lambda: plot_sales_heatmap(df, save_path=f"{figures_dir}/sales_heatmap.png")),
        ("category_share",    lambda: plot_category_share(df, save_path=f"{figures_dir}/category_share.png")),
    ]
    for name, fn in chart_specs:
        result = fn()
        fig_obj = result if isinstance(result, plt.Figure) else result[0]
        plt.close(fig_obj)
        logger.info("  Saved %s.png", name)

    saved = list(Path(figures_dir).glob("*.png"))
    logger.info("%d chart(s) written to %s/", len(saved), figures_dir)


# ═══════════════════════════════════════════════════════════════════════════════
# 4. Main entry-point
# ═══════════════════════════════════════════════════════════════════════════════

def main(data_path:   str  = DEFAULT_DATA,
         output_path: str  = DEFAULT_OUTPUT,
         charts:      bool = False) -> None:
    """
    Full report-generation pipeline.

    Steps
    -----
    1  Load & clean data
    2  Compute all aggregates
    3  Render plain-text report
    4  Print to stdout
    5  Write to file
    6  (Optional) Generate PNG charts

    Parameters
    ----------
    data_path   : path to shop_data.csv
    output_path : destination .txt report file
    charts      : generate PNGs in reports/figures/ when True
    """
    setup_logging()
    generated_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # 1. Load
    logger.info("[1/5] Loading data from %s …", data_path)
    df = load_clean_data(data_path, verbose=False)
    logger.info("  Loaded %d rows × %d columns.", *df.shape)

    # 2. Aggregate
    logger.info("[2/5] Computing aggregates …")
    kpis      = overall_kpis(df)
    cat_sum   = get_category_summary(df)
    top_prods = top_products(df)
    daily     = daily_revenue(df)
    wday_sum  = weekday_summary(df)
    pbands    = price_band_summary(df)
    rat_dist  = rating_distribution(df)

    # 3. Render
    logger.info("[3/5] Rendering report …")
    report_text = render_report(kpis, cat_sum, top_prods, daily,
                                wday_sum, pbands, rat_dist, generated_at)

    # 4. Print
    logger.info("[4/5] Printing report …")
    print(report_text)

    # 5. Write to file
    ensure_dir(Path(output_path).parent)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write(report_text)
    logger.info("[5/5] Report written → %s  (%d bytes)",
                output_path, Path(output_path).stat().st_size)

    # 6. Charts (optional)
    if charts:
        logger.info("[+]   Generating charts …")
        generate_charts(df, FIGURES_DIR)

    print_section("Report Generation Complete")
    print(f"  Report → {output_path}")
    if charts:
        print(f"  Charts → {FIGURES_DIR}/")


# ── CLI ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate ShopSmart monthly sales report",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data",   default=DEFAULT_DATA,   help="Path to CSV")
    parser.add_argument("--output", default=DEFAULT_OUTPUT, help="Output .txt path")
    parser.add_argument("--charts", action="store_true",    help="Generate PNG charts")
    args = parser.parse_args()
    main(args.data, args.output, args.charts)
