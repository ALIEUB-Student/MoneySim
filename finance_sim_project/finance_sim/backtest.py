"""Historical backtest: "what would have happened with REAL market history?"

Instead of random returns, we replay actual monthly returns. For every month
in the data we ask: "if you had started your plan THEN, where would you be
after `years` years?" Each possible start month is one historical window.

Honest limits (important):
  * Windows overlap heavily. Two windows starting a month apart share almost
    all their data, so 300 windows are NOT 300 independent experiments.
  * Short histories (e.g. 35 years) hold only one or two truly separate
    30-year periods.
  * Past returns don't guarantee future returns.
  * Inflation here is the plan's constant assumption, not real CPI data.
"""

from dataclasses import dataclass

import numpy as np
import pandas as pd

from finance_sim.engine import Plan, summarize


def load_prices(path: str) -> pd.DataFrame:
    """Read a CSV (first column = dates, other columns = prices, one per asset)
    and return month-end prices."""
    df = pd.read_csv(path, index_col=0, parse_dates=True).sort_index()
    df = df.apply(pd.to_numeric, errors="coerce").dropna(how="all")
    monthly = df.groupby(df.index.to_period("M")).last()
    return monthly.dropna()


def download_prices(tickers, start: str = "1980-01-01", path: str | None = None) -> pd.DataFrame:
    """Download adjusted prices with yfinance and return month-end prices.
    Needs internet and `pip install yfinance`."""
    try:
        import yfinance as yf
    except ImportError as e:
        raise SystemExit("Install yfinance first:  pip install yfinance") from e

    tickers = list(tickers)
    raw = yf.download(tickers, start=start, auto_adjust=True, progress=False)
    if raw is None or len(raw) == 0:
        raise SystemExit("No data came back. Check the tickers and your internet connection.")
    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    missing = [t for t in tickers if t not in prices.columns or prices[t].dropna().empty]
    if missing:
        raise SystemExit(f"No data for: {', '.join(missing)}")
    prices = prices[tickers].dropna()  # only dates where every asset has a price
    monthly = prices.groupby(prices.index.to_period("M")).last()
    if path:
        monthly.to_csv(path)
    return monthly


def monthly_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Month-over-month percentage change of each asset."""
    return prices.pct_change().dropna()


@dataclass
class BacktestResult:
    balances: np.ndarray        # (n_windows, years + 1), real terms
    start_months: pd.PeriodIndex  # first month of each window


def rolling_backtest(plan: Plan, returns: pd.DataFrame, weights) -> BacktestResult:
    """Replay history through the plan, once per possible start month.

    returns: monthly returns, one column per asset.
    weights: target share of each column (must sum to 1); the mix is rebalanced
             back to these weights every month (a simplification).
    """
    plan.validate()
    w = np.asarray(weights, dtype=float)
    if len(w) != returns.shape[1]:
        raise ValueError(f"{len(w)} weights given but the data has {returns.shape[1]} assets")
    if (w < 0).any() or not np.isclose(w.sum(), 1.0):
        raise ValueError("weights must be non-negative and add up to 1")

    port = returns.to_numpy() @ w
    n_months = plan.years * 12
    if len(port) < n_months:
        raise ValueError(
            f"Only {len(port) / 12:.1f} years of data, but the plan is {plan.years} years. "
            f"Use a plan of {len(port) // 12} years or fewer."
        )

    monthly_inflation = (1 + plan.inflation) ** (1 / 12) - 1
    real_gross = (1 + port) / (1 + monthly_inflation)  # deflate to today's money

    n_windows = len(port) - n_months + 1
    idx = np.arange(n_windows)
    balance = np.full(n_windows, float(plan.starting_balance))
    balances = np.empty((n_windows, plan.years + 1))
    balances[:, 0] = balance

    contribution = plan.monthly_contribution
    for m in range(n_months):
        balance = balance * real_gross[idx + m] + contribution
        if (m + 1) % 12 == 0:
            balances[:, (m + 1) // 12] = balance
            contribution *= 1 + plan.contribution_growth

    return BacktestResult(balances, returns.index[:n_windows])


def summarize_backtest(result: BacktestResult, plan: Plan, goal: float | None = None) -> dict:
    """Same summary as the simulation, plus the best and worst real start dates."""
    s = summarize(result.balances, plan, goal)
    final = result.balances[:, -1]
    iw, ib = int(final.argmin()), int(final.argmax())
    s.update(
        n_windows=len(final),
        worst_start=str(result.start_months[iw]),
        worst_final=float(final[iw]),
        best_start=str(result.start_months[ib]),
        best_final=float(final[ib]),
        first_start=str(result.start_months[0]),
        last_start=str(result.start_months[-1]),
    )
    return s
