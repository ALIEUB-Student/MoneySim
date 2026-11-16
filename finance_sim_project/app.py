"""Streamlit front end for the personal finance simulator.

Run from the folder that contains this file and the finance_sim folder:
    streamlit run app.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

from finance_sim.backtest import (
    download_prices,
    load_prices,
    monthly_returns,
    rolling_backtest,
    summarize_backtest,
)
from finance_sim.engine import Plan, project_deterministic, summarize
from finance_sim.market import (
    calibration,
    estimate,
    fetch_history,
    holdout_test,
    log_drift_from_expected,
    plan_inputs,
    simulate_price_paths,
)
from finance_sim.plots import backtest_figure, fan_figure, mix_figure, price_fan_figure
from finance_sim.portfolio import (
    blended_return,
    compare_mixes,
    simulate_mix,
    steady_plan_for_mix,
)

st.set_page_config(page_title="Future Money Simulator", page_icon="📈", layout="wide")


# ---------- helpers ----------

def show(fig) -> None:
    """Draw a matplotlib figure in the page, then free its memory."""
    st.pyplot(fig, width="stretch")
    plt.close(fig)


@st.cache_data(show_spinner="Simulating thousands of futures...")
def run_mix(start, monthly, years, ret, vol, infl, raise_, w, bret, bvol, corr, n, seed):
    plan = Plan(start, monthly, years, ret, vol, infl, raise_)
    balances = simulate_mix(plan, w, bret, bvol, corr, n, seed)
    steady = project_deterministic(steady_plan_for_mix(plan, w, bret))
    return balances, steady


@st.cache_data(show_spinner="Comparing portfolios...")
def run_compare(start, monthly, years, ret, vol, infl, raise_, goal, bret, bvol, corr, n, seed):
    plan = Plan(start, monthly, years, ret, vol, infl, raise_)
    return compare_mixes(plan, goal=goal, bond_ret=bret, bond_vol=bvol, corr=corr,
                         n_sims=n, seed=seed)


@st.cache_data(ttl=60, show_spinner="Fetching prices...")
def get_history(ticker, years):
    """Prices are cached for 1 minute so reruns do not hammer Yahoo; the Refresh button clears this cache."""
    return fetch_history(ticker, years)


@st.cache_data(show_spinner="Repeating the test on past dates...")
def run_calibration(prices, years_ahead, widen):
    return calibration(prices, years_ahead, account_for_uncertainty=widen)


def use_in_plan(expected_return: float, vol: float) -> None:
    """Button callback: copy a stock's measured numbers into the sidebar."""
    st.session_state["ret_pct"] = float(min(max(round(expected_return * 100, 1), -20.0), 50.0))
    st.session_state["vol_pct"] = float(min(max(round(vol * 100, 1), 0.0), 100.0))
    st.toast("Sidebar updated: the other tabs now use these stock numbers.")


# ---------- sidebar: the user's situation ----------

st.sidebar.header("Your plan")
cur = st.sidebar.text_input("Currency symbol", "$", max_chars=4)


def fmt(x: float) -> str:
    return f"{cur}{x:,.0f}"


start = st.sidebar.number_input(f"Money you already have to invest ({cur})", 0, value=2000, step=500)
monthly = st.sidebar.number_input(f"Amount you add each month ({cur})", 0, value=400, step=50)
years = st.sidebar.slider("Years until you need the money", 1, 50, 30)
raise_pct = st.sidebar.number_input("Yearly raise in your monthly amount (%)", 0.0, 20.0, 0.0, step=0.5)
infl_pct = st.sidebar.number_input("Expected yearly inflation (%)", 0.0, 20.0, 2.5, step=0.5)
has_goal = st.sidebar.checkbox("I have a savings goal", value=True)
goal = (
    st.sidebar.number_input(f"Goal, in today's money ({cur})", 1, value=300_000, step=10_000)
    if has_goal else None
)

st.sidebar.header("Your investments")
stock_pct = st.sidebar.slider("Share in stocks (the rest is bonds)", 0, 100, 60, step=5, format="%d%%")
st.session_state.setdefault("ret_pct", 7.0)   # the Live market tab's button can set these two
st.session_state.setdefault("vol_pct", 15.0)
ret_pct = st.sidebar.number_input("Stocks: expected yearly return (%)", min_value=-20.0,
                                  max_value=50.0, step=0.5, key="ret_pct")
vol_pct = st.sidebar.number_input("Stocks: yearly volatility (%)", min_value=0.0, max_value=100.0,
                                  step=1.0, key="vol_pct",
                                  help="How much a typical year swings up or down. "
                                       "The stock market has historically been around 15-20%.")
with st.sidebar.expander("Bond assumptions"):
    bret_pct = st.number_input("Bonds: expected yearly return (%)", 0.0, 30.0, 4.0, step=0.5)
    bvol_pct = st.number_input("Bonds: yearly volatility (%)", 0.0, 80.0, 6.0, step=1.0)
    corr = st.slider("How closely stocks and bonds move together", -0.9, 0.9, 0.1, step=0.1,
                     help="1 = identical moves, 0 = unrelated, negative = opposite moves. "
                          "The lower it is, the more mixing them smooths the ride.")
with st.sidebar.expander("Advanced"):
    n_sims = st.number_input("Number of simulated futures", 1000, 50_000, 5000, step=1000)
    seed = st.number_input("Random seed", 0, 10_000, 0,
                           help="Same seed = same random numbers, so results stay stable when you "
                                "change other settings. Change it to see a different set of futures.")

w = stock_pct / 100
args = (start, monthly, years, ret_pct / 100, vol_pct / 100, infl_pct / 100, raise_pct / 100)
bond_args = (bret_pct / 100, bvol_pct / 100, corr)
plan = Plan(*args)

# ---------- main page ----------

st.title("Future Money Simulator")
st.caption("See a *range* of possible futures for your savings, not just one number. "
           "Everything is shown in today's money (inflation-adjusted). "
           "A model with assumptions, not a prediction or financial advice.")

tab_sim, tab_mix, tab_hist, tab_live = st.tabs(
    ["Simulation", "Compare portfolios", "Real history", "Live market"])

# ----- Tab 1: Monte Carlo -----
with tab_sim:
    balances, steady = run_mix(*args, w, *bond_args, int(n_sims), int(seed))
    s = summarize(balances, plan, goal)

    st.subheader(f"{stock_pct}% stocks / {100 - stock_pct}% bonds")
    st.caption(f"Blended expected return before inflation: "
               f"{blended_return(w, plan.expected_return, bond_args[0]):.2%} per year")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("You will put in", fmt(s["total_contributed"]))
    c2.metric("Unlucky (1 in 10)", fmt(s["worst_10pct"]),
              help="In 9 out of 10 simulated futures you end up with at least this much.")
    c3.metric("Typical (median)", fmt(s["median"]),
              help="Half of the simulated futures end above this, half below.")
    c4.metric("Lucky (1 in 10)", fmt(s["best_10pct"]),
              help="Only 1 in 10 simulated futures ends above this.")

    msg = f"Chance of ending with less than you put in: **{s['chance_of_losing_money']:.1%}**."
    if goal is not None:
        msg += f"  Chance of reaching your {fmt(goal)} goal: **{s['chance_of_reaching_goal']:.1%}**."
    st.markdown(msg)

    show(fan_figure(balances, steady, goal))

    with st.expander("How to read this chart"):
        st.markdown(
            "- Each simulated future draws a random return for every year, so no two are alike.\n"
            "- The **dark band** holds the middle half of the futures, the **light band** the middle "
            "80%. Your real future will most likely land somewhere in them, but it can land outside.\n"
            "- The **dashed line** is what you'd get if every year returned exactly the average. "
            "The typical (median) result ends *below* it: losses hurt more than equal gains help, "
            "so volatility drags down the typical outcome.\n"
            "- Try raising the volatility in the sidebar and watch the bands widen."
        )

# ----- Tab 2: compare mixes -----
with tab_mix:
    st.subheader("Same plan, different stock/bond splits")
    rows = run_compare(*args, goal, *bond_args, int(n_sims), int(seed))
    show(mix_figure(rows, goal))

    table = pd.DataFrame({
        "Stocks": [f"{r['stock_weight']:.0%}" for r in rows],
        "Unlucky (1 in 10)": [fmt(r["worst_10pct"]) for r in rows],
        "Typical (median)": [fmt(r["median"]) for r in rows],
        "Lucky (1 in 10)": [fmt(r["best_10pct"]) for r in rows],
        "Ends below what you put in": [f"{r['chance_of_losing_money']:.1%}" for r in rows],
    })
    if goal is not None:
        table["Reaches your goal"] = [f"{r['chance_of_reaching_goal']:.1%}" for r in rows]
    st.dataframe(table, hide_index=True, width="stretch")
    st.caption("Every row uses the same random numbers, so differences come from the mix, "
               "not from luck. More stocks usually means a higher typical result but a wider range. "
               "Notice that 100% bonds is not the safest choice here: its returns barely beat inflation.")

# ----- Tab 3: real history -----
with tab_hist:
    st.subheader("What real market history would have done")
    st.markdown(
        "Instead of random numbers, replay **real monthly returns**. For every possible start "
        "month we ask: *if I had started my plan then, where would I be after "
        f"{years} years?*  The share in stocks comes from the sidebar."
    )

    source = st.radio("Where should the price data come from?",
                      ["Download from Yahoo Finance", "Upload a CSV file"], horizontal=True)

    if source.startswith("Download"):
        c1, c2, c3 = st.columns(3)
        stock_t = c1.text_input("Stock fund ticker", "VFINX")
        bond_t = c2.text_input("Bond fund ticker (leave empty for stocks only)", "VBMFX")
        since = c3.number_input("Download history from year", 1950, 2020, 1980)
        if st.button("Download data"):
            tickers = [t.strip() for t in (stock_t, bond_t) if t.strip()]
            try:
                with st.spinner("Downloading..."):
                    st.session_state["prices"] = download_prices(tickers, start=f"{since}-01-01")
            except (SystemExit, Exception) as e:  # download_prices exits with a message on failure
                st.error(f"Couldn't download the data: {e}")
    else:
        st.caption("First column = dates, other columns = prices (stock fund first, then bond fund).")
        uploaded = st.file_uploader("Prices CSV", type="csv")
        if uploaded is not None:
            try:
                st.session_state["prices"] = load_prices(uploaded)
            except Exception as e:
                st.error(f"Couldn't read that file: {e}")

    prices = st.session_state.get("prices")
    if prices is None:
        st.info("Load some price data above to run the backtest.")
    else:
        returns = monthly_returns(prices)
        names = list(returns.columns)
        if len(names) == 1:
            weights = [1.0]
            st.caption(f"One asset loaded ({names[0]}), so the portfolio is 100% in it.")
        elif len(names) == 2:
            weights = [w, 1 - w]
        else:
            weights = None
            st.error("Please load at most 2 assets (stocks, then bonds).")

        if weights is not None:
            try:
                result = rolling_backtest(plan, returns, weights)
            except ValueError as e:
                st.error(str(e))
            else:
                b = summarize_backtest(result, plan, goal)
                st.caption(
                    f"{' / '.join(f'{wt:.0%} {n}' for wt, n in zip(weights, names))} · "
                    f"data {returns.index[0]} to {returns.index[-1]} · "
                    f"{b['n_windows']} overlapping {years}-year windows")

                c1, c2, c3, c4 = st.columns(4)
                c1.metric("You will put in", fmt(b["total_contributed"]))
                c2.metric("Worst real start", fmt(b["worst_final"]), help=f"Starting {b['worst_start']}")
                c3.metric("Median", fmt(b["median"]))
                c4.metric("Best real start", fmt(b["best_final"]), help=f"Starting {b['best_start']}")

                msg = f"Chance of ending with less than you put in: **{b['chance_of_losing_money']:.1%}**."
                if goal is not None:
                    msg += f"  Reached your {fmt(goal)} goal: **{b['chance_of_reaching_goal']:.1%}** of windows."
                st.markdown(msg)

                show(backtest_figure(result.balances, result.start_months, goal))
                st.caption(
                    "Windows starting a month apart share almost all their data, so these are only a "
                    "handful of truly different histories. Inflation here is your constant assumption, "
                    "not real inflation data. Past returns don't guarantee future returns.")


# ----- Tab 4: live market -----
with tab_live:
    st.subheader("Live market data and price scenarios")
    st.caption("Prices come from Yahoo Finance through the free yfinance library. They are often delayed "
               "by about 15 minutes and are not meant for trading decisions.")

    c1, c2, c3, c4 = st.columns([2, 1, 1, 1], vertical_alignment="bottom")
    ticker = c1.text_input(
        "Ticker symbol", "SPY",
        help="Examples: SPY, AAPL, VT. Stocks on other exchanges need a suffix, such as 2222.SR or VOD.L.",
    ).strip().upper()
    hist_years = c2.selectbox("History to learn from", [3, 5, 10, 20], index=2,
                              format_func=lambda y: f"{y} years")
    horizon = c3.slider("Forecast years", 1, 10, 5)
    if c4.button("Refresh prices"):
        get_history.clear()

    auto = st.checkbox("Auto-refresh prices", value=False,
                       help="Re-fetches prices and redraws everything below on a timer. Only this tab "
                            "updates; the rest of the app stays put.")
    every = None
    if auto:
        secs = st.selectbox("Refresh every", [60, 300, 900], index=1,
                            format_func=lambda s: f"{s // 60} minute" + ("s" if s > 60 else ""),
                            help="Yahoo data is delayed and free; refreshing faster than once a minute "
                                 "gets you blocked and adds nothing.")
        every = int(secs)

    def live_body(ticker, hist_years, horizon):
        prices = None
        if not ticker:
            st.info("Type a ticker symbol above.")
        else:
            try:
                prices = get_history(ticker, int(hist_years))
            except ValueError as e:
                st.error(str(e))
            except Exception as e:
                st.error("Couldn't get prices right now. Yahoo Finance may be busy, or may be limiting requests "
                         f"from this server, so wait a minute and press Refresh prices. (Details: {e})")

        if prices is not None:
            est = estimate(prices)
            last, prev = float(prices.iloc[-1]), float(prices.iloc[-2])
            past_year = prices[prices.index > prices.index[-1] - pd.DateOffset(years=1)]

            m1, m2, m3, m4 = st.columns(4)
            m1.metric(f"{ticker} latest price", f"{last:,.2f}", f"{last / prev - 1:+.2%} vs previous close")
            m2.metric("52-week high", f"{past_year.max():,.2f}")
            m3.metric("52-week low", f"{past_year.min():,.2f}")
            m4.metric("Data through", f"{prices.index[-1]:%d %b %Y}")
            st.line_chart(prices, height=280)

            st.markdown("#### What the history says")
            a1, a2, a3, a4 = st.columns(4)
            a1.metric("Average yearly growth", f"{est.cagr:.1%}",
                      help="The steady yearly rate that matches the whole history, dividends included.")
            a2.metric("Volatility", f"{est.log_vol:.1%}",
                      help="The typical size of a year's swing (annualized standard deviation of returns).")
            a3.metric("Worst fall", f"{est.max_drawdown:.0%}",
                      help="The biggest drop from a peak to a later low in this history.")
            a4.metric("History used", f"{est.span_years:.1f} years")
            st.warning(
                "How much to trust these: volatility is measured well, the average growth is not. With "
                f"{est.span_years:.0f} years of data the average yearly growth is uncertain by about "
                f"±{est.mean_std_error:.1%} (one standard error), so the true figure could easily be several "
                "points higher or lower. Longer histories help only slowly.")

            st.markdown("#### Price scenarios")
            choice = st.radio("Average yearly return to use",
                              ["Measured from this history", "My own assumption"], horizontal=True)
            d_std = 0.0
            if choice.startswith("Measured"):
                m_used, exp_used = est.log_mean, est.expected_return
                if st.checkbox("Widen the range to reflect uncertainty in that average (recommended)", value=True,
                               help="The measured average is noisy. Without this, the forecast acts as if it "
                                    "were exact and its ranges come out too narrow. See the test below."):
                    d_std = est.mean_std_error
            else:
                own = st.number_input("My expected yearly return (%)", -20.0, 50.0,
                                      float(st.session_state["ret_pct"]), step=0.5)
                exp_used = own / 100
                m_used = log_drift_from_expected(exp_used, est.log_vol)

            paths = simulate_price_paths(est.last_price, m_used, est.log_vol, horizon,
                                         int(n_sims), int(seed), drift_std=d_std)
            final = paths[:, -1]
            lo, mid, hi = np.percentile(final, [10, 50, 90])
            k1, k2, k3 = st.columns(3)
            k1.metric(f"Unlucky (1 in 10), in {horizon} years", f"{lo:,.2f}", f"{lo / last - 1:+.0%} vs today")
            k2.metric("Median", f"{mid:,.2f}", f"{mid / last - 1:+.0%} vs today")
            k3.metric("Lucky (1 in 10)", f"{hi:,.2f}", f"{hi / last - 1:+.0%} vs today")
            st.markdown(
                f"In these scenarios, the chance the price is above today's after {horizon} years is "
                f"**{np.mean(final > last):.0%}**, and the chance it has fallen by more than 30% is "
                f"**{np.mean(final < 0.7 * last):.0%}**.")
            show(price_fan_figure(paths, est.last_price, horizon))
            st.caption("These are scenarios if the future behaves like the past, not predictions. The model knows "
                       "nothing about earnings, news or crashes more extreme than a bell curve allows.")

            exp_plan, vol_plan = plan_inputs(exp_used, est.log_vol)
            st.button("Use these numbers as my plan's stock assumptions", on_click=use_in_plan,
                      args=(exp_plan, vol_plan))
            st.caption(
                f"Sets the sidebar's stock return to {exp_plan:.1%} and volatility to {vol_plan:.1%}, so the "
                "Simulation, Compare portfolios and Real history tabs treat this stock as the stock part of "
                "your portfolio. (The plan measures volatility as the spread of yearly simple returns, a bit "
                "above the log-return figure shown here.)")

            with st.expander("Does this method work? Test it on past data"):
                max_back = min(5, int(est.span_years) - 3)
                if max_back < 1:
                    st.write("Not enough history for a test. Choose a longer history above (at least 5 years).")
                else:
                    back = (st.slider("Pretend it is this many years ago", 1, max_back,
                                      min(max_back, int(horizon)))
                            if max_back > 1 else 1)
                    try:
                        res = holdout_test(prices, back, int(n_sims), int(seed), account_for_uncertainty=True)
                    except ValueError as e:
                        st.error(str(e))
                    else:
                        r_lo, r_hi = np.percentile(res.paths[:, -1], [10, 90])
                        st.markdown(
                            f"Using only data up to **{res.cutoff:%b %Y}**, the model's middle 80% range for the "
                            f"price {back} years later was **{r_lo:,.2f} to {r_hi:,.2f}**. The actual price was "
                            f"**{res.actual_prices[-1]:,.2f}**, at the **{res.percentile * 100:.0f}th percentile** "
                            "of the model's outcomes (50th would be dead centre).")
                        show(price_fan_figure(
                            res.paths, res.est.last_price, back,
                            actual=(res.actual_years, res.actual_prices),
                            title="Test on the past: what the model expected vs what happened",
                            xlabel="Years after the test start"))

                        cal_with = run_calibration(prices, back, True)
                        cal_without = run_calibration(prices, back, False)
                        if len(cal_with) >= 4:
                            def inside(c):
                                return float(np.mean((c >= 0.1) & (c <= 0.9)))
                            st.markdown(
                                f"Repeating this from **{len(cal_with)}** different start dates in the history, "
                                f"the real price landed inside the model's 80% range **{inside(cal_with):.0%}** "
                                f"of the time with the uncertainty widening, and **{inside(cal_without):.0%}** "
                                "without it. A perfect model would score about 80%.")
                            st.caption("The tests overlap heavily and use a single stock, so this is a rough "
                                       "health check, not proof. A model that fits its own past well can still "
                                       "fail on the next regime.")
                        else:
                            st.caption("Not enough history to repeat the test from several start dates.")

            st.caption("Educational tool, not financial advice.")

    # run_every=None means a normal one-time render; a number makes only this block re-run on a timer.
    st.fragment(live_body, run_every=every)(ticker, hist_years, horizon)
