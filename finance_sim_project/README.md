# Personal Finance Simulator

A tool anyone can use to see a *range* of possible financial futures instead of a single number.

## Setup

```
pip install -r requirements.txt
```

Run every command from the folder that CONTAINS `app.py` and the `finance_sim` folder.

## The web app (easiest way to use it)

```
streamlit run app.py
```

Your browser opens with sliders for your plan, four tabs (Simulation, Compare portfolios, Real history, Live market),
and the charts drawn right in the page. Nothing to save or open by hand.

### Live market tab

Type a ticker (SPY, AAPL, 2222.SR, VOD.L ...) to fetch real daily prices from Yahoo Finance (via `yfinance`,
free, often delayed ~15 min). The app measures growth and volatility from that history, simulates thousands of
future price paths, and can send the numbers to your plan's sidebar. The average return measured from history is
noisy, so the range is widened to reflect that, and an in-app test checks how the method would have done on past data.
Tick **Auto-refresh prices** to re-fetch and redraw on a timer (1, 5 or 15 minutes). Results are scenarios, not predictions. Educational tool, not financial advice.

## Command line (for scripting and testing)

## Simulate random futures

```
python -m finance_sim.cli --start 2000 --monthly 400 --years 30 --goal 500000 --show
```

`--show` saves `fan_chart.png` and opens it. Use `--plot myname.png` to choose the file name.

## Portfolio mix (stocks + bonds)

```
python -m finance_sim.cli --start 2000 --monthly 400 --years 30 --goal 500000 --stocks 0.6 --compare --show
```

- `--stocks 0.6` = 60% stocks, 40% bonds (stocks use `--ret` / `--vol`)
- `--bond-ret`, `--bond-vol`, `--corr` change the bond assumptions and how much stocks and bonds move together
- `--compare` prints a table of splits from 100% stocks to 100% bonds

## Backtest on real history

```
python -m finance_sim.fetch_data VFINX VBMFX        # writes prices.csv (stocks first, then bonds)
python -m finance_sim.cli --start 2000 --monthly 400 --years 30 --goal 500000 --backtest prices.csv --stocks 0.6 --show
```

The plan length must be shorter than the history in the file. Windows overlap heavily, so treat the result as
a few real histories, not hundreds of independent experiments.

## How it works

- `engine.py`: steady projection, single-asset Monte Carlo, and `simulate_from_returns` (runs the plan through any returns).
- `portfolio.py`: correlated stock/bond returns (Cholesky), yearly rebalancing, mix comparison.
- `backtest.py`: loads prices, replays real monthly returns from every possible start month.
- `market.py`: live prices, estimation, price-path simulation, past-data tests.
- `plots.py`: fan chart, mix-range chart, backtest chart (as figures for the app, or saved as PNG files).
- `cli.py`: command-line front end.
- `app.py` (next to the folder): the Streamlit web app.
- Everything is shown in today's money (inflation-adjusted).

## Roadmap

1. Core model (done)
2. Charts (done)
3. Portfolio mix (done)
4. Historical backtesting (done)
5. Real CPI inflation in the backtest
6. Web interface (Streamlit) (done)
7. Live market data (done)
8. Deploy it online (Streamlit Community Cloud), save/load plans, more asset types
