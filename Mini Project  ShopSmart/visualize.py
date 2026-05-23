"""
src/visualize.py
================
ShopSmart — Visualisation Layer

All public functions:
  • Accept an optional ``ax``; create a new Figure if None.
  • Never call plt.show() — the caller decides when to display.
  • Accept ``save_path`` (str | None); save at 150 dpi when given.
  • Return (fig, ax), except plot_dashboard which returns fig.

Functions
---------
plot_sales_by_category   horizontal bar — total revenue per category
plot_sales_trend         line — daily/weekly/monthly revenue over time
plot_top_products        vertical bar — top-N products by revenue
plot_price_distribution  box or violin — price spread per category
plot_rating_vs_units     scatter — customer rating vs units sold
plot_sales_heatmap       heatmap — day-of-week × week-of-month
plot_category_share      donut/pie — category revenue share
plot_dashboard           3×2 dashboard combining all six charts
"""

from __future__ import annotations
import logging, os
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# ── Palette ───────────────────────────────────────────────────────────────────
PALETTE = {
    "primary"  : "#2563EB",
    "secondary": "#F97316",
    "success"  : "#16A34A",
    "danger"   : "#DC2626",
    "muted"    : "#94A3B8",
    "bg"       : "#F8FAFC",
}
CATEGORY_PALETTE = ["#2563EB","#F97316","#16A34A","#DC2626","#7C3AED","#0891B2","#CA8A04"]
DPI = 150; FONT_TITLE = 13; FONT_LABEL = 10


def _style(fig, ax):
    fig.patch.set_facecolor(PALETTE["bg"]); ax.set_facecolor(PALETTE["bg"])
    ax.spines[["top","right"]].set_visible(False)
    ax.spines[["left","bottom"]].set_color("#CBD5E1")
    ax.tick_params(colors="#475569")
    ax.xaxis.label.set_color("#334155"); ax.yaxis.label.set_color("#334155")
    ax.title.set_color("#1E293B")


def _save(fig, save_path):
    if save_path:
        os.makedirs(Path(save_path).parent, exist_ok=True)
        fig.savefig(save_path, dpi=DPI, bbox_inches="tight",
                    facecolor=fig.get_facecolor())
        logger.info("Saved → %s", save_path)


# ── 1. Sales by Category ─────────────────────────────────────────────────────

def plot_sales_by_category(
    df: pd.DataFrame,
    value_col: str = "sales",
    top_n: int = 10,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Horizontal bar chart — total revenue (or any metric) per category.

    The leading category is highlighted in orange; the rest in blue.
    Dollar value labels are placed to the right of each bar.

    Parameters
    ----------
    df        : clean DataFrame with 'category' and value_col
    value_col : column to sum (default 'sales')
    top_n     : max categories to display
    ax        : existing Axes; new Figure created if None
    save_path : optional PNG path

    Returns
    -------
    (fig, ax)
    """
    agg = df.groupby("category")[value_col].sum().sort_values().tail(top_n)
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, max(4, len(agg)*0.60)))
    else:
        fig = ax.get_figure()

    colors = [PALETTE["primary"]]*len(agg); colors[-1] = PALETTE["secondary"]
    bars = ax.barh(agg.index, agg.values, color=colors, edgecolor="white", height=0.65)
    for bar, val in zip(bars, agg.values):
        ax.text(bar.get_width()*1.01, bar.get_y()+bar.get_height()/2,
                f"${val:,.0f}", va="center", fontsize=9, color="#334155")

    ax.set_xlabel(f"Total {value_col.replace('_',' ').title()} (USD)", fontsize=FONT_LABEL)
    ax.set_title(f"Total {value_col.replace('_',' ').title()} by Category",
                 fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax.xaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x,_: f"${x/1_000:.0f}K" if x>=1_000 else f"${x:.0f}"))
    _style(fig, ax); plt.tight_layout(); _save(fig, save_path)
    return fig, ax


# ── 2. Sales Trend ───────────────────────────────────────────────────────────

def plot_sales_trend(
    df: pd.DataFrame,
    freq: str = "D",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Line chart of total sales resampled at ``freq``.

    The peak period is annotated automatically with an arrow.

    Parameters
    ----------
    df        : DataFrame with datetime 'date' and 'sales'
    freq      : "D" daily (default) | "W" weekly | "ME" month-end
    ax        : optional existing Axes
    save_path : optional PNG path

    Returns
    -------
    (fig, ax)
    """
    labels = {"D":"Daily","W":"Weekly","ME":"Monthly","M":"Monthly"}
    ts = df.set_index("date")["sales"].resample(freq).sum().reset_index()
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4))
    else:
        fig = ax.get_figure()

    ax.plot(ts["date"], ts["sales"], color=PALETTE["primary"], linewidth=2, zorder=3)
    ax.fill_between(ts["date"], ts["sales"], alpha=0.12, color=PALETTE["primary"])

    peak = ts.loc[ts["sales"].idxmax()]
    ax.annotate(f"Peak\n${peak['sales']:,.0f}", xy=(peak["date"], peak["sales"]),
                xytext=(10, 10), textcoords="offset points", fontsize=8,
                color=PALETTE["secondary"],
                arrowprops=dict(arrowstyle="->", color=PALETTE["secondary"], lw=1.2))

    ax.set_xlabel("Date", fontsize=FONT_LABEL)
    ax.set_ylabel("Sales Revenue (USD)", fontsize=FONT_LABEL)
    ax.set_title(f"{labels.get(freq, freq)} Sales Trend",
                 fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x:,.0f}"))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, color="#CBD5E1")
    _style(fig, ax); plt.tight_layout(); _save(fig, save_path)
    return fig, ax


# ── 3. Top Products ──────────────────────────────────────────────────────────

def plot_top_products(
    df: pd.DataFrame,
    top_n: int = 10,
    value_col: str = "sales",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Vertical bar chart — top-N products by total value_col.

    Uses a Blues gradient (darkest = rank 1). Dollar labels above each bar.

    Parameters
    ----------
    df        : DataFrame with 'product_name' and value_col
    top_n     : number of products (default 10)
    value_col : metric to rank on (default 'sales')
    ax        : optional existing Axes
    save_path : optional PNG path

    Returns
    -------
    (fig, ax)
    """
    agg = df.groupby("product_name")[value_col].sum().sort_values(ascending=False).head(top_n)
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 4.5))
    else:
        fig = ax.get_figure()

    blues = plt.cm.Blues(np.linspace(0.45, 0.85, len(agg)))[::-1]
    bars = ax.bar(range(len(agg)), agg.values, color=blues, edgecolor="white", width=0.7)
    ax.set_xticks(range(len(agg)))
    ax.set_xticklabels(
        [n[:20]+"…" if len(n)>20 else n for n in agg.index],
        rotation=35, ha="right", fontsize=8.5)

    ax.set_ylabel(f"Total {value_col.replace('_',' ').title()} (USD)", fontsize=FONT_LABEL)
    ax.set_title(f"Top {top_n} Products by {value_col.replace('_',' ').title()}",
                 fontsize=FONT_TITLE, fontweight="bold", pad=10)

    max_val = agg.values.max()
    for bar, val in zip(bars, agg.values):
        ax.text(bar.get_x()+bar.get_width()/2, bar.get_height()+max_val*0.01,
                f"${val:,.0f}", ha="center", va="bottom", fontsize=7.5, color="#334155")

    ax.yaxis.set_major_formatter(
        mtick.FuncFormatter(lambda x,_: f"${x/1_000:.0f}K" if x>=1_000 else f"${x:.0f}"))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, color="#CBD5E1")
    _style(fig, ax); plt.tight_layout(); _save(fig, save_path)
    return fig, ax


# ── 4. Price Distribution ────────────────────────────────────────────────────

def plot_price_distribution(
    df: pd.DataFrame,
    kind: str = "box",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Box-plot or violin-plot of price spread per category.

    Categories are sorted by median price (highest on the left).

    Parameters
    ----------
    df        : clean DataFrame with 'category' and 'price'
    kind      : "box" (default) or "violin"
    ax        : optional existing Axes
    save_path : optional PNG path

    Returns
    -------
    (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(11, 5))
    else:
        fig = ax.get_figure()

    order = df.groupby("category")["price"].median().sort_values(ascending=False).index.tolist()
    # Build a colour mapping dict so seaborn receives hue + palette properly
    # (avoids the deprecated bare-palette warning in seaborn >= 0.13)
    cat_color_map = {cat: CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)]
                     for i, cat in enumerate(order)}
    plot_fn = sns.violinplot if kind == "violin" else sns.boxplot
    kwargs = dict(inner="quartile", linewidth=0.8) if kind=="violin" else dict(linewidth=0.9, fliersize=3)
    plot_fn(data=df, x="category", y="price", order=order,
            hue="category", palette=cat_color_map, legend=False,
            ax=ax, **kwargs)

    ax.set_xlabel("Category", fontsize=FONT_LABEL)
    ax.set_ylabel("Price (USD)", fontsize=FONT_LABEL)
    ax.set_title(f"Price Distribution by Category ({kind.title()} Plot)",
                 fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=20)
    ax.yaxis.set_major_formatter(mtick.FuncFormatter(lambda x,_: f"${x:.0f}"))
    ax.grid(True, axis="y", linestyle="--", alpha=0.4, color="#CBD5E1")
    _style(fig, ax); plt.tight_layout(); _save(fig, save_path)
    return fig, ax


# ── 5. Rating vs Units ───────────────────────────────────────────────────────

def plot_rating_vs_units(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Scatter — customer_rating (x) vs units_sold (y), coloured by category.

    A linear trend line across all data answers: do higher-rated products
    actually sell more units?

    Returns
    -------
    (fig, ax)
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=(9, 5))
    else:
        fig = ax.get_figure()

    for i, cat in enumerate(sorted(df["category"].unique())):
        sub = df[df["category"]==cat]
        ax.scatter(sub["customer_rating"], sub["units_sold"], label=cat,
                   alpha=0.65, s=45, color=CATEGORY_PALETTE[i % len(CATEGORY_PALETTE)],
                   edgecolors="white", linewidths=0.4)

    valid = df[["customer_rating","units_sold"]].dropna()
    z = np.polyfit(valid["customer_rating"], valid["units_sold"], 1)
    x_line = np.linspace(valid["customer_rating"].min(), valid["customer_rating"].max(), 100)
    ax.plot(x_line, np.poly1d(z)(x_line), "--", color="#64748B", linewidth=1.5, label="Trend")

    ax.set_xlabel("Customer Rating (1–5)", fontsize=FONT_LABEL)
    ax.set_ylabel("Units Sold", fontsize=FONT_LABEL)
    ax.set_title("Customer Rating vs Units Sold", fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax.legend(fontsize=8, bbox_to_anchor=(1.01,1), loc="upper left", borderaxespad=0)
    ax.grid(True, linestyle="--", alpha=0.35, color="#CBD5E1")
    _style(fig, ax); plt.tight_layout(); _save(fig, save_path)
    return fig, ax


# ── 6. Sales Heatmap ─────────────────────────────────────────────────────────

def plot_sales_heatmap(
    df: pd.DataFrame,
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Heatmap — total sales by day-of-week (columns) × week-of-month (rows).

    Reveals which day/week combinations generate the most revenue.
    Date temporal features are computed on the fly if not already present.

    Returns
    -------
    (fig, ax)
    """
    tmp = df.copy()
    if "day_name" not in tmp.columns:
        tmp["day_name"] = tmp["date"].dt.day_name()
    tmp["week_of_month"] = (tmp["date"].dt.day - 1) // 7 + 1

    pivot = tmp.groupby(["week_of_month","day_name"])["sales"].sum().unstack(fill_value=0)
    dow = ["Monday","Tuesday","Wednesday","Thursday","Friday","Saturday","Sunday"]
    pivot = pivot.reindex(columns=[d for d in dow if d in pivot.columns])

    if ax is None:
        fig, ax = plt.subplots(figsize=(10, 4))
    else:
        fig = ax.get_figure()

    sns.heatmap(pivot, ax=ax, cmap="Blues", annot=True, fmt=".0f",
                linewidths=0.5, linecolor="#E2E8F0",
                cbar_kws={"label":"Total Sales (USD)","shrink":0.8},
                annot_kws={"size":8})
    ax.set_xlabel("Day of Week", fontsize=FONT_LABEL)
    ax.set_ylabel("Week of Month", fontsize=FONT_LABEL)
    ax.set_title("Sales Heatmap — Day of Week × Week of Month",
                 fontsize=FONT_TITLE, fontweight="bold", pad=10)
    ax.tick_params(axis="x", rotation=30); ax.tick_params(axis="y", rotation=0)
    _style(fig, ax); plt.tight_layout(); _save(fig, save_path)
    return fig, ax


# ── 7. Category Share ────────────────────────────────────────────────────────

def plot_category_share(
    df: pd.DataFrame,
    value_col: str = "sales",
    kind: str = "donut",
    ax: Optional[plt.Axes] = None,
    save_path: Optional[str] = None,
) -> tuple[plt.Figure, plt.Axes]:
    """
    Donut or pie chart — each category's share of total revenue.

    White percentage labels are rendered inside each segment.

    Parameters
    ----------
    df        : DataFrame with 'category' and value_col
    value_col : metric to summarise (default 'sales')
    kind      : "donut" (default) or "pie"
    ax        : optional existing Axes
    save_path : optional PNG path

    Returns
    -------
    (fig, ax)
    """
    agg = df.groupby("category")[value_col].sum().sort_values(ascending=False)
    if ax is None:
        fig, ax = plt.subplots(figsize=(7, 7))
    else:
        fig = ax.get_figure()

    wp = dict(width=0.45) if kind=="donut" else {}
    wedges, texts, autotexts = ax.pie(
        agg.values, labels=agg.index, autopct="%1.1f%%",
        colors=CATEGORY_PALETTE[:len(agg)], startangle=90,
        pctdistance=0.78, wedgeprops=wp)
    for t in texts: t.set_fontsize(9)
    for a in autotexts: a.set_fontsize(8); a.set_color("white"); a.set_fontweight("bold")

    ax.set_title(f"Category Share of {value_col.replace('_',' ').title()}",
                 fontsize=FONT_TITLE, fontweight="bold", pad=14)
    fig.patch.set_facecolor(PALETTE["bg"]); plt.tight_layout(); _save(fig, save_path)
    return fig, ax


# ── 8. Dashboard ─────────────────────────────────────────────────────────────

def plot_dashboard(
    df: pd.DataFrame,
    save_path: Optional[str] = None,
) -> plt.Figure:
    """
    3×2 summary dashboard — all six charts in a single Figure.

    Layout
    ------
    [0,0] Sales by category    [0,1] Daily sales trend
    [1,0] Top 8 products       [1,1] Price distribution (box)
    [2,0] Rating vs units      [2,1] Category revenue share (donut)

    Parameters
    ----------
    df        : clean DataFrame from load_clean_data()
    save_path : optional high-res PNG path

    Returns
    -------
    plt.Figure  (call plt.show() or plt.close() in the caller)
    """
    fig = plt.figure(figsize=(18, 16))
    fig.patch.set_facecolor(PALETTE["bg"])
    fig.suptitle("ShopSmart — Sales Analytics Dashboard",
                 fontsize=17, fontweight="bold", y=1.01, color="#1E293B")

    axes = [fig.add_subplot(3, 2, i+1) for i in range(6)]
    plot_sales_by_category(df,              ax=axes[0])
    plot_sales_trend(df, freq="D",          ax=axes[1])
    plot_top_products(df, top_n=8,          ax=axes[2])
    plot_price_distribution(df, kind="box", ax=axes[3])
    plot_rating_vs_units(df,                ax=axes[4])
    plot_category_share(df,                 ax=axes[5])

    plt.tight_layout(pad=2.5)
    _save(fig, save_path)
    return fig


# ── Self-test ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    import matplotlib; matplotlib.use("Agg")
    from src.data_loader import load_clean_data
    df = load_clean_data("data/shop_data.csv", verbose=False)
    os.makedirs("reports/figures", exist_ok=True)
    fig = plot_dashboard(df, save_path="reports/figures/dashboard.png")
    plt.close(fig)
    print("✓ Dashboard saved → reports/figures/dashboard.png")
