"""Charts for the simulator.

fan_chart() draws every simulated future as a band:
  - light band  = the middle 80% of outcomes (10th to 90th percentile)
  - darker band = the middle 50% of outcomes (25th to 75th percentile)
  - solid line  = the median (typical) outcome
  - dashed line = the steady-average projection (no randomness)
"""

import os
import subprocess
import sys

import matplotlib

matplotlib.use("Agg")  # draw to files; no window needed
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

# One hue, light -> dark, so the bands read as "more likely" toward the middle.
BLUE = "#2a78d6"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e6e5e1"
SURFACE = "#fcfcfb"


def _money(x: float, _pos=None) -> str:
    if abs(x) >= 1_000_000:
        return f"{x / 1_000_000:.1f}M"
    if abs(x) >= 1_000:
        return f"{x / 1_000:.0f}k"
    return f"{x:.0f}"


def _end_labels(ax, x_end, items, min_gap_frac=0.085) -> None:
    """Write labels just right of the last point, pushed apart so none overlap.

    items: list of (y_value, text, color, weight). Labels keep their order from
    bottom to top; any that would be closer than min_gap_frac of the chart's
    height get nudged upward.
    """
    ymax = max(y for y, *_ in items) * 1.05
    gap = ymax * min_gap_frac
    placed = []
    for y, text, color, weight in sorted(items, key=lambda t: t[0]):
        if placed and y < placed[-1] + gap:
            y = placed[-1] + gap
        placed.append(y)
        ax.annotate(text, (x_end, y), xytext=(6, 0), textcoords="offset points",
                    va="center", fontsize=9, color=color, fontweight=weight,
                    annotation_clip=False)


def fan_figure(
    balances: np.ndarray,
    steady: np.ndarray,
    goal: float | None = None,
):
    """Build the fan chart and return the matplotlib Figure (not saved).

    balances: (n_sims, years + 1) array from simulate_monte_carlo
    steady:   (years + 1,) array from project_deterministic
    """
    years = np.arange(balances.shape[1])
    p10, p25, p50, p75, p90 = np.percentile(balances, [10, 25, 50, 75, 90], axis=0)

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.fill_between(years, p10, p90, color=BLUE, alpha=0.18, linewidth=0)
    ax.fill_between(years, p25, p75, color=BLUE, alpha=0.32, linewidth=0)
    ax.plot(years, p50, color=BLUE, linewidth=2)
    ax.plot(years, steady, color=INK_SOFT, linewidth=1.5, linestyle=(0, (5, 4)))

    if goal is not None:
        ax.axhline(goal, color=INK_SOFT, linewidth=1, linestyle=":")
        ax.text(0.2, goal, f" Goal: {_money(goal)}", color=INK_SOFT,
                va="bottom", ha="left", fontsize=9)

    # Direct labels at the right edge instead of a legend box.
    x_end = years[-1]
    _end_labels(ax, x_end, [
        (p90[-1], f"Lucky (90th)  {_money(p90[-1])}", INK_SOFT, "normal"),
        (p50[-1], f"Typical (median)  {_money(p50[-1])}", INK, "bold"),
        (steady[-1], f"Steady average  {_money(steady[-1])}", INK_SOFT, "normal"),
        (p10[-1], f"Unlucky (10th)  {_money(p10[-1])}", INK_SOFT, "normal"),
    ])

    ax.set_title("Where your money could be (today's money)",
                 loc="left", fontsize=13, color=INK, pad=12)
    ax.set_xlabel("Years from now", color=INK_SOFT, fontsize=10)
    ax.set_ylabel("Balance", color=INK_SOFT, fontsize=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_money))
    ax.set_xlim(0, x_end)
    ax.set_ylim(bottom=0)

    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, length=0, labelsize=9)

    # Room on the right for the direct labels.
    fig.subplots_adjust(right=0.78, left=0.09, bottom=0.12, top=0.9)
    fig.text(0.09, 0.02,
             f"Shaded: middle 50% (dark) and middle 80% (light) of {balances.shape[0]:,} "
             "simulated futures. A model, not a prediction.",
             color=INK_SOFT, fontsize=8)
    return fig


def fan_chart(balances, steady, goal=None, path: str = "fan_chart.png") -> str:
    """Save the fan chart to `path` and return the path (used by the CLI)."""
    fig = fan_figure(balances, steady, goal)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def backtest_figure(
    balances: np.ndarray,
    start_months,
    goal: float | None = None,
):
    """Every real historical start month as one thin line, so you can see how
    much the starting date mattered. Worst start is solid dark, best is dashed.
    Returns the matplotlib Figure (not saved)."""
    years = np.arange(balances.shape[1])
    final = balances[:, -1]
    iw, ib = int(final.argmin()), int(final.argmax())
    median = np.median(balances, axis=0)

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)

    ax.plot(years, balances.T, color=BLUE, alpha=0.10, linewidth=0.8)
    ax.plot(years, median, color=BLUE, linewidth=2)
    ax.plot(years, balances[iw], color=INK, linewidth=1.8)
    ax.plot(years, balances[ib], color=INK_SOFT, linewidth=1.8, linestyle=(0, (5, 4)))

    if goal is not None:
        ax.axhline(goal, color=INK_SOFT, linewidth=1, linestyle=":")
        ax.text(0.2, goal, f" Goal: {_money(goal)}", color=INK_SOFT,
                va="bottom", ha="left", fontsize=9)

    x_end = years[-1]
    _end_labels(ax, x_end, [
        (final[ib], f"Best start {start_months[ib]}  {_money(final[ib])}", INK_SOFT, "normal"),
        (median[-1], f"Median  {_money(median[-1])}", INK, "bold"),
        (final[iw], f"Worst start {start_months[iw]}  {_money(final[iw])}", INK, "normal"),
    ])

    ax.set_title("What real market history would have done (today's money)",
                 loc="left", fontsize=13, color=INK, pad=12)
    ax.set_xlabel("Years from start", color=INK_SOFT, fontsize=10)
    ax.set_ylabel("Balance", color=INK_SOFT, fontsize=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_money))
    ax.set_xlim(0, x_end)
    ax.set_ylim(bottom=0)

    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, length=0, labelsize=9)

    fig.subplots_adjust(right=0.74, left=0.09, bottom=0.14, top=0.9)
    fig.text(0.09, 0.02,
             f"Each thin line = one real start month ({len(final)} windows, "
             f"{start_months[0]} to {start_months[-1]}). Windows overlap heavily, "
             "so they are not independent experiments.",
             color=INK_SOFT, fontsize=8)
    return fig


def backtest_chart(balances, start_months, goal=None, path: str = "backtest_chart.png") -> str:
    """Save the backtest chart to `path` and return the path (used by the CLI)."""
    fig = backtest_figure(balances, start_months, goal)
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)
    return path


def mix_figure(rows: list[dict], goal: float | None = None):
    """One bar per stock/bond split: the bar spans the unlucky-to-lucky range
    (10th to 90th percentile), the dot is the median. rows come from
    portfolio.compare_mixes. Returns the matplotlib Figure (not saved)."""
    rows = sorted(rows, key=lambda r: r["stock_weight"])
    x = np.array([r["stock_weight"] * 100 for r in rows])
    lo = np.array([r["worst_10pct"] for r in rows])
    mid = np.array([r["median"] for r in rows])
    hi = np.array([r["best_10pct"] for r in rows])
    spacing = np.min(np.diff(x)) if len(x) > 1 else 20
    width = spacing * 0.55

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.bar(x, hi - lo, bottom=lo, width=width, color=BLUE, alpha=0.25, linewidth=0)
    ax.plot(x, mid, color=BLUE, linewidth=1.5, alpha=0.6)
    ax.scatter(x, mid, color=BLUE, s=36, zorder=3, edgecolor=SURFACE, linewidth=1.5)
    for xi, m in zip(x, mid):
        ax.annotate(_money(m), (xi, m), xytext=(0, 9), textcoords="offset points",
                    ha="center", fontsize=9, color=INK, fontweight="bold")

    if goal is not None:
        ax.axhline(goal, color=INK_SOFT, linewidth=1, linestyle=":")
        ax.text(x[0] - width / 2, goal, f" Goal: {_money(goal)}", color=INK_SOFT,
                va="bottom", ha="left", fontsize=9)

    ax.set_xticks(x)
    ax.set_xticklabels([f"{v:.0f}%" for v in x])
    ax.set_title("Typical outcome and range, by share in stocks (today's money)",
                 loc="left", fontsize=13, color=INK, pad=12)
    ax.set_xlabel("Share of the portfolio in stocks (the rest is bonds)",
                  color=INK_SOFT, fontsize=10)
    ax.set_ylabel("Balance at the end", color=INK_SOFT, fontsize=10)
    ax.yaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(_money))
    ax.set_ylim(bottom=0)

    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, length=0, labelsize=9)

    fig.subplots_adjust(left=0.09, right=0.97, bottom=0.17, top=0.9)
    fig.text(0.09, 0.02,
             "Bar = unlucky (10th percentile) to lucky (90th percentile). "
             "Dot = typical (median). A model, not a prediction.",
             color=INK_SOFT, fontsize=8)
    return fig


def _price(x: float) -> str:
    return f"{x:,.0f}" if abs(x) >= 1000 else f"{x:,.2f}"


def price_fan_figure(
    paths: np.ndarray,
    p0: float,
    years: float,
    actual=None,
    title: str = "Where the price could be",
    xlabel: str = "Years from now",
):
    """Fan chart of simulated PRICE paths (shape n_sims x steps + 1).

    actual: optional (years_array, prices_array) of what really happened, drawn
    as a dark line over the fan (used by the 'test on the past' feature).
    Returns the matplotlib Figure (not saved)."""
    x = np.linspace(0, years, paths.shape[1])
    p10, p25, p50, p75, p90 = np.percentile(paths, [10, 25, 50, 75, 90], axis=0)

    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=150, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    ax.fill_between(x, p10, p90, color=BLUE, alpha=0.18, linewidth=0)
    ax.fill_between(x, p25, p75, color=BLUE, alpha=0.32, linewidth=0)
    ax.plot(x, p50, color=BLUE, linewidth=2)
    ax.axhline(p0, color=INK_SOFT, linewidth=1, linestyle=":")
    ax.text(0.0, p0, f" Price at start: {_price(p0)} ", color=INK_SOFT,
            va="bottom", ha="left", fontsize=9, zorder=6,
            bbox=dict(facecolor=SURFACE, edgecolor="none", pad=1.5, alpha=0.9))

    items = [
        (p90[-1], f"Lucky (90th)  {_price(p90[-1])}", INK_SOFT, "normal"),
        (p50[-1], f"Median  {_price(p50[-1])}", INK, "bold"),
        (p10[-1], f"Unlucky (10th)  {_price(p10[-1])}", INK_SOFT, "normal"),
    ]
    if actual is not None:
        ax_years, ax_prices = actual
        ax.plot(ax_years, ax_prices, color=INK, linewidth=1.8)
        items.append((ax_prices[-1], f"Actual  {_price(ax_prices[-1])}", INK, "bold"))
    _end_labels(ax, x[-1], items)

    ax.set_title(title, loc="left", fontsize=13, color=INK, pad=12)
    ax.set_xlabel(xlabel, color=INK_SOFT, fontsize=10)
    ax.set_ylabel("Price", color=INK_SOFT, fontsize=10)
    ax.set_xlim(0, x[-1])
    ax.set_ylim(bottom=0)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=INK_SOFT, length=0, labelsize=9)

    fig.subplots_adjust(right=0.78, left=0.09, bottom=0.14, top=0.9)
    fig.text(0.09, 0.02,
             f"Shaded: middle 50% (dark) and middle 80% (light) of {paths.shape[0]:,} simulated "
             "price paths. Scenarios from past behavior, not a prediction.",
             color=INK_SOFT, fontsize=8)
    return fig


def open_file(path: str) -> None:
    """Open an image in the computer's default viewer (Windows, macOS, Linux)."""
    try:
        if sys.platform.startswith("win"):
            os.startfile(path)  # type: ignore[attr-defined]
        elif sys.platform == "darwin":
            subprocess.run(["open", path], check=False)
        else:
            subprocess.run(["xdg-open", path], check=False)
    except OSError:
        print(f"Couldn't open it automatically. Open {path} from your project folder.")
