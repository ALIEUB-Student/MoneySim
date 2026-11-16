"""Portfolio mix: split money between stocks and bonds.

Key idea: assets don't move independently. Stocks and bonds each have their own
average return and volatility, and a CORRELATION says how much they move
together (1 = identical, 0 = unrelated, -1 = opposite). The lower the
correlation, the more mixing them smooths the ride. That's diversification.

Simplification: the portfolio is rebalanced back to its target weights every
year, so the portfolio's return in a year is the weighted average of the
assets' returns that year.
"""

from dataclasses import replace

import numpy as np

from finance_sim.engine import (
    Plan,
    lognormal_params,
    simulate_from_returns,
    summarize,
)

# Rough long-run ASSUMPTIONS (nominal). They are inputs you can change, not facts.
DEFAULT_BOND_RETURN = 0.04
DEFAULT_BOND_VOL = 0.06
DEFAULT_CORRELATION = 0.1


def correlated_gross_returns(
    expected_returns: list[float],
    volatilities: list[float],
    corr: np.ndarray,
    years: int,
    n_sims: int,
    seed: int | None = None,
) -> np.ndarray:
    """Random yearly gross returns for several assets at once.

    Returns shape (n_sims, years, n_assets).

    Method: draw independent standard normals, multiply by the Cholesky factor
    of the correlation matrix to make them correlated, then turn each into a
    lognormal return using that asset's own average and volatility.
    """
    corr = np.asarray(corr, dtype=float)
    k = len(expected_returns)
    if corr.shape != (k, k) or not np.allclose(corr, corr.T) or not np.allclose(np.diag(corr), 1):
        raise ValueError("correlation must be a symmetric k x k matrix with 1s on the diagonal")
    try:
        chol = np.linalg.cholesky(corr)
    except np.linalg.LinAlgError as e:
        raise ValueError("correlation matrix is not valid (not positive definite)") from e

    params = [lognormal_params(r, v) for r, v in zip(expected_returns, volatilities)]
    mu = np.array([p[0] for p in params])
    sigma = np.array([p[1] for p in params])

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n_sims, years, k)) @ chol.T
    return np.exp(mu + sigma * z)


def blended_return(stock_weight: float, stock_ret: float, bond_ret: float) -> float:
    """Expected yearly return of a rebalanced stock/bond mix."""
    return stock_weight * stock_ret + (1 - stock_weight) * bond_ret


def simulate_mix(
    plan: Plan,
    stock_weight: float,
    bond_ret: float = DEFAULT_BOND_RETURN,
    bond_vol: float = DEFAULT_BOND_VOL,
    corr: float = DEFAULT_CORRELATION,
    n_sims: int = 5000,
    seed: int | None = None,
) -> np.ndarray:
    """Monte Carlo for a stock/bond mix. Stocks use plan.expected_return and
    plan.volatility; bonds use the bond_* arguments."""
    plan.validate()
    if not 0 <= stock_weight <= 1:
        raise ValueError("stock_weight must be between 0 and 1")
    if not -1 < corr < 1:
        raise ValueError("correlation must be between -1 and 1 (exclusive)")

    gross = correlated_gross_returns(
        [plan.expected_return, bond_ret],
        [plan.volatility, bond_vol],
        np.array([[1.0, corr], [corr, 1.0]]),
        plan.years,
        n_sims,
        seed,
    )
    weights = np.array([stock_weight, 1 - stock_weight])
    return simulate_from_returns(plan, gross @ weights)


def steady_plan_for_mix(plan: Plan, stock_weight: float, bond_ret: float) -> Plan:
    """A copy of the plan whose expected return is the mix's blended return,
    for the 'steady average' reference line."""
    return replace(
        plan, expected_return=blended_return(stock_weight, plan.expected_return, bond_ret)
    )


def compare_mixes(
    plan: Plan,
    stock_weights=(1.0, 0.8, 0.6, 0.4, 0.2, 0.0),
    goal: float | None = None,
    bond_ret: float = DEFAULT_BOND_RETURN,
    bond_vol: float = DEFAULT_BOND_VOL,
    corr: float = DEFAULT_CORRELATION,
    n_sims: int = 5000,
    seed: int | None = 0,
) -> list[dict]:
    """Same plan, several stock/bond splits. The same seed is used for every
    mix so the comparison is fair (same 'luck' in every row)."""
    rows = []
    for w in stock_weights:
        balances = simulate_mix(plan, w, bond_ret, bond_vol, corr, n_sims, seed)
        row = summarize(balances, plan, goal)
        row["stock_weight"] = w
        rows.append(row)
    return rows
