import os
import re
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="Nifty 50 Downloader + Backtest", layout="wide")

st.title("Nifty 50 Yahoo Finance Downloader + Strategy Backtest")


INPUT_FILE = "Nifty 50 symbols.csv"
OUTPUT_FOLDER = "data"

INITIAL_CAPITAL = 1_000_000
MAX_POSITION_PCT = 0.10

DEFAULT_DOWNLOAD_START_DATE = "2023-01-01"
DEFAULT_BACKTEST_START_DATE = "2025-01-01"

SMA_50 = 50
SMA_150 = 150
EMA_220 = 220
LOOKBACK_52W = 252
DIP_LOOKBACK = 90
STOP_LOSS_PCT = 0.15


def clean_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", str(name))


def to_yahoo_symbol(symbol):
    symbol = str(symbol).strip().upper()
    return symbol if symbol.endswith(".NS") else f"{symbol}.NS"


def read_symbols(file_path):
    df = pd.read_csv(file_path, header=None)
    symbols = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()

    if not symbols:
        raise ValueError("No symbols found in CSV.")

    return symbols


@st.cache_data(show_spinner=False)
def download_stock_data(ticker, start_date):
    df = yf.download(
        ticker,
        start=start_date,
        end=date.today().strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
    )

    if df.empty:
        return pd.DataFrame()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])

    return df


def add_indicators(df):
    df = df.copy()

    df["SMA_50"] = df["Close"].rolling(SMA_50).mean()
    df["SMA_150"] = df["Close"].rolling(SMA_150).mean()
    df["EMA_220"] = df["Close"].ewm(span=EMA_220, adjust=False).mean()

    df["Low_52W"] = df["Low"].rolling(LOOKBACK_52W).min()
    df["High_52W_Prev"] = df["Close"].rolling(LOOKBACK_52W).max().shift(1)

    df["Dipped_Below_EMA_220"] = (
        (df["Low"] < df["EMA_220"])
        .rolling(DIP_LOOKBACK)
        .max()
        .fillna(0)
        .astype(bool)
    )

    df["Rule_1_150SMA_GT_220EMA"] = df["SMA_150"] > df["EMA_220"]
    df["Rule_2_Close_GT_50SMA"] = df["Close"] > df["SMA_50"]
    df["Rule_3_50SMA_GT_150SMA"] = df["SMA_50"] > df["SMA_150"]
    df["Rule_4_Close_GT_125pct_52WLow"] = df["Close"] > 1.25 * df["Low_52W"]
    df["Rule_5_Dipped_Below_220EMA_90D"] = df["Dipped_Below_EMA_220"]
    df["Rule_6_Close_Breakout_52WHigh"] = df["Close"] > df["High_52W_Prev"]

    df["Filter_Pass"] = (
        df["Rule_1_150SMA_GT_220EMA"]
        & df["Rule_2_Close_GT_50SMA"]
        & df["Rule_3_50SMA_GT_150SMA"]
        & df["Rule_4_Close_GT_125pct_52WLow"]
        & df["Rule_5_Dipped_Below_220EMA_90D"]
    )

    df["Breakout_52W_High"] = df["Rule_6_Close_Breakout_52WHigh"]
    df["Entry_Signal"] = df["Filter_Pass"] & df["Breakout_52W_High"]

    return df


def get_today_buy_candidates(stock_data):
    rows = []

    for symbol, df in stock_data.items():
        if df.empty:
            continue

        last = df.iloc[-1]

        if bool(last["Entry_Signal"]):
            rows.append({
                "Stock": symbol,
                "Close": round(last["Close"], 2),
                "SMA 50": round(last["SMA_50"], 2),
                "SMA 150": round(last["SMA_150"], 2),
                "EMA 220": round(last["EMA_220"], 2),
                "52W High Prev": round(last["High_52W_Prev"], 2),
                "52W Low": round(last["Low_52W"], 2),
                "% Above 52W Low": round(((last["Close"] / last["Low_52W"]) - 1) * 100, 2),
            })

    return pd.DataFrame(rows)


def get_near_trade_watchlist(stock_data):
    rows = []

    rule_columns = {
        "150 SMA > 220 EMA": "Rule_1_150SMA_GT_220EMA",
        "Close > 50 SMA": "Rule_2_Close_GT_50SMA",
        "50 SMA > 150 SMA": "Rule_3_50SMA_GT_150SMA",
        "Close > 1.25 x 52W Low": "Rule_4_Close_GT_125pct_52WLow",
        "Dipped below 220 EMA in 90D": "Rule_5_Dipped_Below_220EMA_90D",
        "Close breakout above 52W High": "Rule_6_Close_Breakout_52WHigh",
    }

    for symbol, df in stock_data.items():
        if df.empty:
            continue

        last = df.iloc[-1]

        rule_results = {
            rule_name: bool(last[col])
            for rule_name, col in rule_columns.items()
        }

        passed_count = sum(rule_results.values())
        failed_rules = [rule for rule, passed in rule_results.items() if not passed]

        distance_to_breakout = np.nan
        if pd.notna(last["High_52W_Prev"]) and last["High_52W_Prev"] > 0:
            distance_to_breakout = ((last["Close"] / last["High_52W_Prev"]) - 1) * 100

        distance_to_220ema = np.nan
        if pd.notna(last["EMA_220"]) and last["EMA_220"] > 0:
            distance_to_220ema = ((last["Close"] / last["EMA_220"]) - 1) * 100

        rows.append({
            "Stock": symbol,
            "Close": round(last["Close"], 2),
            "Rules Passed": passed_count,
            "Total Rules": 6,
            "Score %": round((passed_count / 6) * 100, 2),
            "Failed Rules": ", ".join(failed_rules) if failed_rules else "None",
            "% From 52W High Breakout": round(distance_to_breakout, 2),
            "% Above 220 EMA": round(distance_to_220ema, 2),
            "Exact Buy Signal": bool(last["Entry_Signal"]),
            "150 SMA > 220 EMA": rule_results["150 SMA > 220 EMA"],
            "Close > 50 SMA": rule_results["Close > 50 SMA"],
            "50 SMA > 150 SMA": rule_results["50 SMA > 150 SMA"],
            "Close > 1.25 x 52W Low": rule_results["Close > 1.25 x 52W Low"],
            "Dipped below 220 EMA in 90D": rule_results["Dipped below 220 EMA in 90D"],
            "Close breakout above 52W High": rule_results["Close breakout above 52W High"],
        })

    watchlist = pd.DataFrame(rows)

    if not watchlist.empty:
        watchlist = watchlist.sort_values(
            by=["Exact Buy Signal", "Rules Passed", "% From 52W High Breakout"],
            ascending=[False, False, False],
        )

    return watchlist


def run_backtest(stock_data, backtest_start_date):
    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    portfolio_values = []

    all_dates = sorted(set(d for df in stock_data.values() for d in df["Date"]))
    all_dates = [d for d in all_dates if d >= pd.to_datetime(backtest_start_date)]

    for current_date in all_dates:

        for symbol in list(positions.keys()):
            df = stock_data[symbol]
            row = df[df["Date"] == current_date]

            if row.empty:
                continue

            row = row.iloc[0]
            position = positions[symbol]

            exit_reason = None

            if row["Close"] < row["EMA_220"]:
                exit_reason = "Close below 220 EMA"
            elif row["Close"] <= position["Entry_Price"] * (1 - STOP_LOSS_PCT):
                exit_reason = "15% stop loss"

            if exit_reason:
                exit_price = row["Close"]
                proceeds = position["Shares"] * exit_price
                cash += proceeds

                pnl = proceeds - position["Invested"]
                pnl_pct = pnl / position["Invested"]

                trades.append({
                    "Symbol": symbol,
                    "Entry Date": position["Entry_Date"],
                    "Entry Price": round(position["Entry_Price"], 2),
                    "Exit Date": current_date,
                    "Exit Price": round(exit_price, 2),
                    "Shares": position["Shares"],
                    "Invested": round(position["Invested"], 2),
                    "PnL": round(pnl, 2),
                    "PnL %": round(pnl_pct * 100, 2),
                    "Exit Reason": exit_reason,
                })

                del positions[symbol]

        max_allocation = INITIAL_CAPITAL * MAX_POSITION_PCT

        for symbol, df in stock_data.items():
            if symbol in positions:
                continue

            signal_row = df[df["Date"] == current_date]

            if signal_row.empty:
                continue

            signal_row = signal_row.iloc[0]

            if not bool(signal_row["Entry_Signal"]):
                continue

            future_rows = df[df["Date"] > current_date]

            if future_rows.empty:
                continue

            entry_row = future_rows.iloc[0]
            entry_date = entry_row["Date"]
            entry_price = entry_row["Open"]

            if pd.isna(entry_price) or entry_price <= 0:
                continue

            allocation = min(max_allocation, cash)
            shares = int(allocation // entry_price)

            if shares <= 0:
                continue

            invested = shares * entry_price
            cash -= invested

            positions[symbol] = {
                "Entry_Date": entry_date,
                "Entry_Price": entry_price,
                "Shares": shares,
                "Invested": invested,
            }

        portfolio_value = cash

        for symbol, position in positions.items():
            df = stock_data[symbol]
            row = df[df["Date"] == current_date]

            if not row.empty:
                portfolio_value += position["Shares"] * row.iloc[0]["Close"]

        portfolio_values.append({
            "Date": current_date,
            "Portfolio Value": portfolio_value,
            "Cash": cash,
            "Open Positions": len(positions),
        })

    return pd.DataFrame(trades), pd.DataFrame(portfolio_values)


def calculate_summary(trades_df, portfolio_df):
    final_value = portfolio_df["Portfolio Value"].iloc[-1]
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

    portfolio_df["Daily Return"] = portfolio_df["Portfolio Value"].pct_change()
    portfolio_df["Peak"] = portfolio_df["Portfolio Value"].cummax()
    portfolio_df["Drawdown"] = (
        portfolio_df["Portfolio Value"] - portfolio_df["Peak"]
    ) / portfolio_df["Peak"]

    max_drawdown = portfolio_df["Drawdown"].min()

    if trades_df.empty:
        win_rate = 0
        avg_win = 0
        avg_loss = 0
    else:
        wins = trades_df[trades_df["PnL"] > 0]
        losses = trades_df[trades_df["PnL"] <= 0]

        win_rate = len(wins) / len(trades_df)
        avg_win = wins["PnL %"].mean() if not wins.empty else 0
        avg_loss = losses["PnL %"].mean() if not losses.empty else 0

    return {
        "Final Portfolio Value": round(final_value, 2),
        "Total Return %": round(total_return * 100, 2),
        "Max Drawdown %": round(max_drawdown * 100, 2),
        "Total Trades": len(trades_df),
        "Win Rate %": round(win_rate * 100, 2),
        "Average Win %": round(avg_win, 2),
        "Average Loss %": round(avg_loss, 2),
    }


st.sidebar.header("Settings")

download_start_date = st.sidebar.date_input(
    "Download start date",
    value=pd.to_datetime(DEFAULT_DOWNLOAD_START_DATE).date(),
)

backtest_start_date = st.sidebar.date_input(
    "Backtest start date",
    value=pd.to_datetime(DEFAULT_BACKTEST_START_DATE).date(),
)

min_rules_for_watchlist = st.sidebar.slider(
    "Minimum rules passed for near-trade watchlist",
    min_value=1,
    max_value=6,
    value=4,
)

st.sidebar.write("Initial capital:", f"₹{INITIAL_CAPITAL:,.0f}")
st.sidebar.write("Max allocation per stock:", f"{MAX_POSITION_PCT * 100:.0f}%")


if not os.path.exists(INPUT_FILE):
    st.error(f"CSV file not found: {INPUT_FILE}")
    st.stop()

try:
    symbols = read_symbols(INPUT_FILE)
    st.success(f"Loaded {len(symbols)} symbols from `{INPUT_FILE}`")
except Exception as e:
    st.error(f"Error reading symbols: {e}")
    st.stop()

with st.expander("View symbols"):
    st.write(symbols)


if st.button("Download Data and Run Backtest"):

    stock_data = {}
    failed_downloads = []

    progress = st.progress(0)
    status = st.empty()

    for i, symbol in enumerate(symbols):
        ticker = to_yahoo_symbol(symbol)
        status.write(f"Downloading {ticker}...")

        try:
            df = download_stock_data(ticker, download_start_date.strftime("%Y-%m-%d"))

            if df.empty:
                failed_downloads.append([symbol, ticker, "No data returned"])
            else:
                df = add_indicators(df)
                stock_data[ticker] = df

                os.makedirs(OUTPUT_FOLDER, exist_ok=True)
                df.to_csv(
                    os.path.join(OUTPUT_FOLDER, clean_filename(ticker) + ".csv"),
                    index=False,
                )

        except Exception as e:
            failed_downloads.append([symbol, ticker, str(e)])

        progress.progress((i + 1) / len(symbols))

    status.write("Download completed.")

    if failed_downloads:
        st.warning("Some symbols failed to download.")
        st.dataframe(
            pd.DataFrame(failed_downloads, columns=["Symbol", "Yahoo Ticker", "Reason"]),
            use_container_width=True,
        )

    if not stock_data:
        st.error("No stock data available for backtesting.")
        st.stop()

    st.subheader("Today's Exact Buy Candidates")

    candidates_df = get_today_buy_candidates(stock_data)

    if candidates_df.empty:
        st.warning("No stocks meet ALL strategy criteria today.")
    else:
        st.success(f"{len(candidates_df)} stocks qualify today.")
        st.dataframe(candidates_df, use_container_width=True)

        st.download_button(
            "Download Exact Buy Candidates CSV",
            data=candidates_df.to_csv(index=False),
            file_name="today_buy_candidates.csv",
            mime="text/csv",
        )

    st.subheader("Near-Trade Watchlist")

    watchlist_df = get_near_trade_watchlist(stock_data)
    near_df = watchlist_df[watchlist_df["Rules Passed"] >= min_rules_for_watchlist]

    if near_df.empty:
        st.warning("No stocks qualify for the near-trade watchlist.")
    else:
        st.write(
            "These are watchlist names only. They pass most rules but may still fail one or more conditions."
        )
        st.dataframe(near_df, use_container_width=True)

        st.download_button(
            "Download Near-Trade Watchlist CSV",
            data=near_df.to_csv(index=False),
            file_name="near_trade_watchlist.csv",
            mime="text/csv",
        )

    st.subheader("Backtest Results")

    trades_df, portfolio_df = run_backtest(
        stock_data,
        backtest_start_date.strftime("%Y-%m-%d"),
    )

    if portfolio_df.empty:
        st.error("Backtest could not run because portfolio data is empty.")
        st.stop()

    summary = calculate_summary(trades_df, portfolio_df)

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Final Value", f"₹{summary['Final Portfolio Value']:,.0f}")
    col2.metric("Total Return", f"{summary['Total Return %']}%")
    col3.metric("Max Drawdown", f"{summary['Max Drawdown %']}%")
    col4.metric("Total Trades", summary["Total Trades"])

    col5, col6, col7 = st.columns(3)
    col5.metric("Win Rate", f"{summary['Win Rate %']}%")
    col6.metric("Avg Win", f"{summary['Average Win %']}%")
    col7.metric("Avg Loss", f"{summary['Average Loss %']}%")

    st.subheader("Portfolio Value Over Time")
    st.line_chart(portfolio_df.set_index("Date")["Portfolio Value"])

    st.subheader("Drawdown")
    st.line_chart(portfolio_df.set_index("Date")["Drawdown"])

    st.subheader("Trades")

    if trades_df.empty:
        st.warning("No historical trades were generated with the current rules.")
    else:
        st.dataframe(trades_df, use_container_width=True)

        st.download_button(
            "Download Trades CSV",
            data=trades_df.to_csv(index=False),
            file_name="backtest_trades.csv",
            mime="text/csv",
        )

    st.subheader("Portfolio Data")
    st.dataframe(portfolio_df, use_container_width=True)

    st.download_button(
        "Download Portfolio CSV",
        data=portfolio_df.to_csv(index=False),
        file_name="backtest_portfolio.csv",
        mime="text/csv",
    )
