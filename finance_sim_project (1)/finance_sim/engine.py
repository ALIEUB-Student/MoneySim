"""Core engine for the personal finance simulator.

Models:
  1. project_deterministic: fixed average return (the "textbook" answer).
  2. simulate_monte_carlo: random yearly returns for ONE asset.
  3. simulate_from_returns: run the plan through ANY matrix of yearly returns
     (used by portfolio.py for multi-asset mixes).

All money is handled in REAL terms (today's purchasing power) by deflating
returns with inflation.
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class Plan:
    """Everything a person needs to describe their situation."""

    starting_balance: float = 0.0       # money invested today
    monthly_contribution: float = 500.0  # amount added each month
    years: int = 30                      # how long you invest
    expected_return: float = 0.07        # average yearly return, nominal (7% = 0.07)
    volatility: float = 0.15             # yearly standard deviation of returns
    inflation: float = 0.025             # average yearly inflation
    contribution_growth: float = 0.0     # yearly raise in contributions (0.03 = +3%/yr)

    def validate(self) -> None:
        if self.years <= 0:
            raise ValueError("years must be positive")
        if self.starting_balance < 0 or self.monthly_contribution < 0:
            raise ValueError("balance and contributions cannot be negative")
        if self.volatility < 0:
            raise ValueError("volatility cannot be negative")
        if self.inflation <= -1 or self.expected_return <= -1:
            raise ValueError("rates must be greater than -100%")


def real_return(nominal: float, inflation: float) -> float:
    """Inflation-adjusted return."""
    return (1 + nominal) / (1 + inflation) - 1


def project_deterministic(plan: Plan) -> np.ndarray:
    """Balance at the end of each year (index 0 = today), no randomness."""
    plan.validate()
    r = real_return(plan.expected_return, plan.inflation)
    monthly_r = (1 + r) ** (1 / 12) - 1

    balance = plan.starting_balance
    contribution = plan.monthly_contribution
    path = [balance]
    for _year in range(plan.years):
        for _month in range(12):
            balance = balance * (1 + monthly_r) + contribution
        path.append(balance)
        contribution *= 1 + plan.contribution_growth
    return np.array(path)


def lognormal_params(expected_return: float, volatility: float) -> tuple[float, float]:
    """(mu, sigma) of the underlying normal so that a lognormal gross return
    has mean 1 + expected_return and standard deviation ~ volatility.

    Lognormal means a single year can never lose more than 100%.
    """
    mean_gross = 1 + expected_return
    sigma2 = np.log(1 + (volatility / mean_gross) ** 2)
    mu = np.log(mean_gross) - sigma2 / 2
    return float(mu), float(np.sqrt(sigma2))


def simulate_from_returns(plan: Plan, gross_returns: np.ndarray) -> np.ndarray:
    """Run the plan through a matrix of random NOMINAL yearly gross returns.

    gross_returns has shape (n_sims, years); 1.07 means +7% that year.
    Returns an array of shape (n_sims, years + 1): each row is one possible
    future, each column the balance at the end of that year (real terms).
    """
    plan.validate()
    n_sims, years = gross_returns.shape
    real_gross = gross_returns / (1 + plan.inflation)  # deflate to today's money

    balances = np.empty((n_sims, years + 1))
    balances[:, 0] = plan.starting_balance
    contribution = plan.monthly_contribution
    for y in range(years):
        yearly_contribution = contribution * 12
        # Simplification: contributions arrive evenly through the year, so on
        # average they earn about half a year of growth.
        half_year_growth = real_gross[:, y] ** 0.5
        balances[:, y + 1] = (
            balances[:, y] * real_gross[:, y] + yearly_contribution * half_year_growth
        )
        contribution *= 1 + plan.contribution_growth
    return balances


def simulate_monte_carlo(
    plan: Plan, n_sims: int = 5000, seed: int | None = None
) -> np.ndarray:
    """Run many random futures for a single asset (see simulate_from_returns)."""
    plan.validate()
    rng = np.random.default_rng(seed)
    mu, sigma = lognormal_params(plan.expected_return, plan.volatility)
    gross_returns = rng.lognormal(mu, sigma, size=(n_sims, plan.years))
    return simulate_from_returns(plan, gross_returns)


def total_contributed(plan: Plan) -> float:
    """Everything the person puts in over the whole plan."""
    return plan.starting_balance + sum(
        plan.monthly_contribution * 12 * (1 + plan.contribution_growth) ** y
        for y in range(plan.years)
    )


def summarize(balances: np.ndarray, plan: Plan, goal: float | None = None) -> dict:
    """Turn raw simulations into numbers a normal person can read."""
    final = balances[:, -1]
    contributed = total_contributed(plan)
    result = {
        "worst_10pct": float(np.percentile(final, 10)),
        "median": float(np.percentile(final, 50)),
        "best_10pct": float(np.percentile(final, 90)),
        "total_contributed": float(contributed),
        "chance_of_losing_money": float(np.mean(final < contributed)),
    }
    if goal is not None:
        result["chance_of_reaching_goal"] = float(np.mean(final >= goal))
    return result
