"""Live market data and price scenarios for ONE stock or fund.

The pipeline:
  fetch_history  -> daily adjusted prices from Yahoo Finance (via yfinance)
  estimate       -> yearly growth and volatility measured from those prices
  simulate_price_paths -> thousands of random future price paths (Monte Carlo)
  holdout_test / calibration -> "would this method have worked in the past?"

Model: the log of the price takes random steps (geometric Brownian motion), so
the price is lognormal and can never go below zero. This is the same idea as
the yearly lognormal returns in engine.py, applied to a price path.

Honest limits: the average return measured from history is very noisy (see
Estimate.mean_std_error), and the model ignores news, crashes beyond a bell
curve, and the fact that the estimates themselves are uncertain.
"""

from dataclasses import dataclass
from math import erf, sqrt

import numpy as np
import pandas as pd


# ---------- data ----------

def fetch_history(ticker: str, years: int = 10) -> pd.Series:
    """Daily adjusted closing prices for the last `years` years.

    Adjusted prices include dividends and splits, so growth measured from them
    is total return. The last value is the latest available price (often
    delayed about 15 minutes during market hours).
    """
    symbol = ticker.strip().upper()
    if not symbol:
        raise ValueError("Enter a ticker symbol.")
    try:
        import yfinance as yf
    except ImportError as e:
        raise RuntimeError("Install yfinance first:  pip install yfinance") from e

    start = (pd.Timestamp.today().normalize() - pd.DateOffset(years=years)).strftime("%Y-%m-%d")
    df = yf.Ticker(symbol).history(start=start, interval="1d", auto_adjust=True)
    if df is None or df.empty or "Close" not in df:
        raise ValueError(
            f"No price data found for '{symbol}'. Check the ticker. Stocks on other exchanges "
            "need a suffix, for example 2222.SR or VOD.L."
        )
    prices = df["Close"].dropna()
    idx = pd.DatetimeIndex(prices.index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    prices.index = idx
    prices.name = symbol
    if len(prices) < 60:
        raise ValueError(f"Only {len(prices)} prices found for '{symbol}'. Need a longer history.")
    return prices


# ---------- estimation ----------

@dataclass
class Estimate:
    last_price: float
    span_years: float        # length of the history, in calendar years
    log_mean: float          # average yearly log return (m)
    log_vol: float           # yearly volatility of log returns (sigma)
    cagr: float              # steady yearly growth that matches the history: exp(m) - 1
    expected_return: float   # arithmetic average yearly return: exp(m + sigma^2/2) - 1
    vol: float               # std of yearly SIMPLE returns, as the plan simulator defines it
    mean_std_error: float    # approx. standard error of log_mean = sigma / sqrt(years)
    max_drawdown: float      # worst peak-to-trough fall in the history (negative number)


def plan_inputs(expected_return: float, log_vol: float) -> tuple[float, float]:
    """Translate (average return, log volatility) into the two numbers the plan
    simulator asks for: expected yearly return and std of yearly simple returns.
    lognormal_params() in engine.py inverts this exactly."""
    vol = (1 + expected_return) * np.sqrt(np.expm1(log_vol**2))
    return float(expected_return), float(vol)


def log_drift_from_expected(expected_return: float, log_vol: float) -> float:
    """Yearly log drift m that gives a chosen arithmetic average return."""
    return float(np.log1p(expected_return) - log_vol**2 / 2)


def estimate(prices: pd.Series) -> Estimate:
    """Measure yearly growth and volatility from a price history."""
    p = prices.dropna()
    if len(p) < 60:
        raise ValueError("Need at least 60 prices to estimate anything.")
    if (p <= 0).any():
        raise ValueError("Prices must be positive.")
    span_years = (p.index[-1] - p.index[0]).days / 365.25
    if span_years < 0.5:
        raise ValueError("Need at least 6 months of history.")

    values = p.to_numpy(dtype=float)
    r = np.diff(np.log(values))                 # daily log returns
    per_year = len(r) / span_years              # trading days per year in this data
    m = float(r.sum() / span_years)             # = ln(P_end / P_start) / years
    sigma = float(r.std(ddof=1) * np.sqrt(per_year))

    expected = float(np.expm1(m + sigma**2 / 2))
    _, vol = plan_inputs(expected, sigma)
    drawdown = float((values / np.maximum.accumulate(values) - 1).min())
    return Estimate(
        last_price=float(values[-1]),
        span_years=float(span_years),
        log_mean=m,
        log_vol=sigma,
        cagr=float(np.expm1(m)),
        expected_return=expected,
        vol=vol,
        mean_std_error=float(sigma / np.sqrt(span_years)),
        max_drawdown=drawdown,
    )


# ---------- simulation ----------

def simulate_price_paths(
    p0: float,
    log_mean: float,
    log_vol: float,
    years: float,
    n_sims: int = 5000,
    seed: int | None = None,
    steps_per_year: int = 12,
    drift_std: float = 0.0,
) -> np.ndarray:
    """Random future price paths, shape (n_sims, steps + 1); column 0 is today.

    Each step the log price moves by  m*dt + sigma*sqrt(dt)*Z  with Z a standard
    normal. Prices are therefore lognormal: median growth is exp(m) per year.

    drift_std > 0 adds uncertainty about m itself: every path gets its own drift
    drawn from Normal(m, drift_std). Use Estimate.mean_std_error here, because a
    drift measured from a few years of data is far from exact.
    """
    if p0 <= 0:
        raise ValueError("Starting price must be positive.")
    if years <= 0 or log_vol < 0:
        raise ValueError("years must be positive and volatility cannot be negative.")
    steps = max(1, int(round(years * steps_per_year)))
    dt = years / steps
    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_sims, steps))
    drift = log_mean + drift_std * rng.standard_normal((n_sims, 1)) if drift_std > 0 else log_mean
    log_steps = drift * dt + log_vol * np.sqrt(dt) * z
    log_path = np.concatenate([np.zeros((n_sims, 1)), np.cumsum(log_steps, axis=1)], axis=1)
    return p0 * np.exp(log_path)


def model_percentile(
    p0: float, actual: float, log_mean: float, log_vol: float, years: float, drift_std: float = 0.0
) -> float:
    """Where `actual` falls among the model's possible prices: 0.5 means it
    landed exactly at the median, 0.05 means only 5% of futures ended lower.

    Exact formula for the lognormal final price (no simulation noise). The
    variance of the final log price is sigma^2*T plus (drift_std*T)^2."""
    spread = np.sqrt(log_vol**2 * years + (drift_std * years) ** 2)
    z = (np.log(actual / p0) - log_mean * years) / spread
    return 0.5 * (1 + erf(float(z) / sqrt(2)))


# ---------- "would it have worked?" ----------

@dataclass
class HoldoutResult:
    est: Estimate                 # estimated from the training part only
    paths: np.ndarray             # simulated paths from the cutoff date
    actual_years: np.ndarray      # years since the cutoff, for the real prices
    actual_prices: np.ndarray     # what really happened
    cutoff: pd.Timestamp
    percentile: float             # where the actual final price fell in the model's range


def holdout_test(
    prices: pd.Series,
    years_ahead: int,
    n_sims: int = 5000,
    seed: int | None = 0,
    account_for_uncertainty: bool = True,
) -> HoldoutResult:
    """Pretend it is `years_ahead` years ago: fit on the earlier data only,
    forecast forward, and compare with what actually happened."""
    cutoff_target = prices.index[-1] - pd.DateOffset(years=years_ahead)
    train = prices[prices.index <= cutoff_target]
    test = prices[prices.index > cutoff_target]
    if len(test) < 2:
        raise ValueError("Not enough recent data to test on.")
    est = estimate(train)  # raises if the training part is too short
    if est.span_years < 2:
        raise ValueError("Need at least 2 years of history before the test period.")

    cutoff = train.index[-1]
    d_std = est.mean_std_error if account_for_uncertainty else 0.0
    paths = simulate_price_paths(est.last_price, est.log_mean, est.log_vol,
                                 years_ahead, n_sims, seed, drift_std=d_std)
    real = pd.concat([train.iloc[[-1]], test])
    years = ((real.index - cutoff).days / 365.25).to_numpy()
    keep = years <= years_ahead + 0.05
    pct = model_percentile(est.last_price, float(real.iloc[-1]), est.log_mean,
                           est.log_vol, years_ahead, d_std)
    return HoldoutResult(est, paths, years[keep], real.to_numpy()[keep], cutoff, pct)


def calibration(
    prices: pd.Series,
    years_ahead: int,
    min_train_years: int = 3,
    step: int = 63,
    account_for_uncertainty: bool = True,
) -> np.ndarray:
    """Repeat the hold-out test from many past start dates (every `step` trading
    days, about every quarter). Returns, for each test, the model percentile of
    the price that actually happened.

    If the model were perfect, about 10% of results would fall below 0.10, 10%
    above 0.90, and 80% in between. Tests overlap heavily, so treat them as a
    rough health check, not as independent experiments."""
    idx = prices.index
    first = idx.searchsorted(idx[0] + pd.DateOffset(years=min_train_years))
    out = []
    for pos in range(first, len(idx), step):
        j = idx.searchsorted(idx[pos] + pd.DateOffset(years=years_ahead))
        if j >= len(idx):
            break
        est = estimate(prices.iloc[: pos + 1])
        d_std = est.mean_std_error if account_for_uncertainty else 0.0
        out.append(model_percentile(est.last_price, float(prices.iloc[j]),
                                    est.log_mean, est.log_vol, years_ahead, d_std))
    return np.array(out)
