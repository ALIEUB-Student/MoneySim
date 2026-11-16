"""Command-line interface: python -m finance_sim.cli --help"""

import argparse

from finance_sim.engine import Plan, project_deterministic, simulate_monte_carlo, summarize


def money(x: float) -> str:
    return f"{x:,.0f}"


def print_report(plan: Plan, s: dict, steady: float | None, goal: float | None) -> None:
    print(f"\nAfter {plan.years} years (all amounts in today's money)")
    print("-" * 46)
    print(f"You will have put in:        {money(s['total_contributed'])}")
    if steady is not None:
        print(f"Steady-average projection:   {money(steady)}")
    print(f"Unlucky (10th percentile):   {money(s['worst_10pct'])}")
    print(f"Typical (median):            {money(s['median'])}")
    print(f"Lucky (90th percentile):     {money(s['best_10pct'])}")
    print(f"Chance of ending below what you put in: {s['chance_of_losing_money']:.1%}")
    if goal is not None:
        print(f"Chance of reaching {money(goal)}: {s['chance_of_reaching_goal']:.1%}")


def print_comparison(rows: list[dict], goal: float | None) -> None:
    print("\nSame plan, different stock/bond splits")
    head = f"{'Stocks':>7} {'Unlucky':>10} {'Median':>10} {'Lucky':>10} {'Below put-in':>13}"
    if goal is not None:
        head += f" {'Hit goal':>9}"
    print(head)
    print("-" * len(head))
    for r in rows:
        line = (f"{r['stock_weight']:>6.0%} {money(r['worst_10pct']):>10} {money(r['median']):>10} "
                f"{money(r['best_10pct']):>10} {r['chance_of_losing_money']:>13.1%}")
        if goal is not None:
            line += f" {r['chance_of_reaching_goal']:>9.1%}"
        print(line)


def run_monte_carlo(a, plan: Plan, plot_path: str | None) -> None:
    from finance_sim.portfolio import (
        blended_return,
        compare_mixes,
        simulate_mix,
        steady_plan_for_mix,
    )

    if a.stocks is None:
        steady_plan = plan
        sims = simulate_monte_carlo(plan, a.sims, a.seed)
    else:
        if not 0 <= a.stocks <= 1:
            raise SystemExit("--stocks must be between 0 and 1 (e.g. 0.6 for 60% stocks)")
        steady_plan = steady_plan_for_mix(plan, a.stocks, a.bond_ret)
        sims = simulate_mix(plan, a.stocks, a.bond_ret, a.bond_vol, a.corr, a.sims, a.seed)
        print(f"\nMix: {a.stocks:.0%} stocks / {1 - a.stocks:.0%} bonds "
              f"(blended expected return {blended_return(a.stocks, a.ret, a.bond_ret):.2%})")

    steady = project_deterministic(steady_plan)
    print_report(plan, summarize(sims, plan, a.goal), steady[-1], a.goal)

    if a.compare:
        rows = compare_mixes(plan, goal=a.goal, bond_ret=a.bond_ret, bond_vol=a.bond_vol,
                             corr=a.corr, n_sims=a.sims, seed=a.seed if a.seed is not None else 0)
        print_comparison(rows, a.goal)

    if plot_path:
        from finance_sim.plots import fan_chart

        fan_chart(sims, steady, a.goal, plot_path)
        print(f"\nChart saved to {plot_path}")


def run_backtest(a, plan: Plan, plot_path: str | None) -> None:
    from finance_sim.backtest import (
        load_prices,
        monthly_returns,
        rolling_backtest,
        summarize_backtest,
    )

    prices = load_prices(a.backtest)
    returns = monthly_returns(prices)
    names = list(returns.columns)
    if len(names) == 1:
        weights = [1.0]
    elif len(names) == 2:
        w = 1.0 if a.stocks is None else a.stocks
        if not 0 <= w <= 1:
            raise SystemExit("--stocks must be between 0 and 1")
        weights = [w, 1 - w]
    else:
        raise SystemExit("The data file can have at most 2 asset columns (stocks, then bonds).")

    result = rolling_backtest(plan, returns, weights)
    s = summarize_backtest(result, plan, a.goal)

    mix = " / ".join(f"{w:.0%} {n}" for w, n in zip(weights, names))
    print(f"\nBacktest on real history: {mix}")
    print(f"Data: {returns.index[0]} to {returns.index[-1]} "
          f"({s['n_windows']} overlapping {plan.years}-year windows)")
    print_report(plan, s, None, a.goal)
    print(f"Worst real start month:      {s['worst_start']}  ->  {money(s['worst_final'])}")
    print(f"Best real start month:       {s['best_start']}  ->  {money(s['best_final'])}")
    print("(Windows overlap heavily, so they are not independent experiments.)")

    if plot_path:
        from finance_sim.plots import backtest_chart

        backtest_chart(result.balances, result.start_months, a.goal, plot_path)
        print(f"\nChart saved to {plot_path}")


def main() -> None:
    p = argparse.ArgumentParser(description="Personal finance Monte Carlo simulator")
    p.add_argument("--start", type=float, default=0, help="starting balance")
    p.add_argument("--monthly", type=float, default=500, help="monthly contribution")
    p.add_argument("--years", type=int, default=30)
    p.add_argument("--ret", type=float, default=0.07, help="expected yearly return (stocks), e.g. 0.07")
    p.add_argument("--vol", type=float, default=0.15, help="yearly volatility (stocks), e.g. 0.15")
    p.add_argument("--inflation", type=float, default=0.025)
    p.add_argument("--raise", dest="raise_", type=float, default=0.0,
                   help="yearly increase in contributions, e.g. 0.03")
    p.add_argument("--goal", type=float, default=None, help="target amount (today's money)")
    p.add_argument("--sims", type=int, default=5000)
    p.add_argument("--seed", type=int, default=None)

    p.add_argument("--stocks", type=float, default=None,
                   help="share of the portfolio in stocks, 0 to 1 (the rest is bonds)")
    p.add_argument("--bond-ret", type=float, default=0.04, help="expected yearly bond return")
    p.add_argument("--bond-vol", type=float, default=0.06, help="yearly bond volatility")
    p.add_argument("--corr", type=float, default=0.1, help="stock/bond correlation (-1 to 1)")
    p.add_argument("--compare", action="store_true",
                   help="also print a table comparing stock/bond splits")
    p.add_argument("--backtest", metavar="CSV", default=None,
                   help="replay real history from a prices CSV instead of random simulation")

    p.add_argument("--plot", nargs="?", const="auto", default=None, metavar="FILE",
                   help="save a chart (default name chosen automatically)")
    p.add_argument("--show", action="store_true", help="save the chart and open it")
    a = p.parse_args()

    plan = Plan(a.start, a.monthly, a.years, a.ret, a.vol, a.inflation, a.raise_)

    plot_path = a.plot
    if a.show and plot_path is None:
        plot_path = "auto"
    if plot_path == "auto":
        plot_path = "backtest_chart.png" if a.backtest else "fan_chart.png"

    try:
        if a.backtest:
            run_backtest(a, plan, plot_path)
        else:
            run_monte_carlo(a, plan, plot_path)
    except ValueError as e:
        raise SystemExit(f"Error: {e}") from e

    if a.show and plot_path:
        from finance_sim.plots import open_file

        open_file(plot_path)

    print("\nThis is a model, not a prediction. Real markets can behave differently.\n")


if __name__ == "__main__":
    main()
