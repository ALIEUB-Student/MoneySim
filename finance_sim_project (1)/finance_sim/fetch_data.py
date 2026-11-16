"""Download monthly prices for the backtest and save them to prices.csv.

Usage (from the folder that CONTAINS finance_sim):
    pip install yfinance pandas
    python -m finance_sim.fetch_data VFINX VBMFX

The first ticker should be your stock fund, the second your bond fund.
VFINX (S&P 500 index fund) and VBMFX (US bond index fund) are mutual funds with
history going back to the 1980s. Change the tickers to test anything else.
"""

import sys

from finance_sim.backtest import download_prices


def main() -> None:
    tickers = sys.argv[1:] or ["VFINX", "VBMFX"]
    prices = download_prices(tickers, start="1980-01-01", path="prices.csv")
    print(prices.head(3))
    print("...")
    print(prices.tail(3))
    years = len(prices) / 12
    print(f"\nSaved prices.csv: {', '.join(tickers)}, {len(prices)} months ({years:.1f} years).")
    print("Run a backtest with a plan no longer than that many years.")


if __name__ == "__main__":
    main()
