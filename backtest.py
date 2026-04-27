import os
import pandas as pd
import numpy as np


DATA_FOLDER = "data"
INITIAL_CAPITAL = 1_000_000
MAX_POSITION_PCT = 0.10
START_DATE = "2025-01-01"

SMA_50 = 50
SMA_150 = 150
EMA_220 = 220
LOW_52W = 252
HIGH_52W = 252
DIP_LOOKBACK = 90
STOP_LOSS_PCT = 0.15


def load_stock_data(file_path):
    df = pd.read_csv(file_path)

    if "Date" not in df.columns:
        df.rename(columns={df.columns[0]: "Date"}, inplace=True)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    df = df.dropna(subset=["Date"])
    df = df.sort_values("Date").reset_index(drop=True)

    numeric_cols = ["Open", "High", "Low", "Close", "Adj Close", "Volume"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["Open", "High", "Low", "Close"])
    return df


def add_indicators(df):
    df["SMA_50"] = df["Close"].rolling(SMA_50).mean()
    df["SMA_150"] = df["Close"].rolling(SMA_150).mean()
    df["EMA_220"] = df["Close"].ewm(span=EMA_220, adjust=False).mean()

    df["Low_52W"] = df["Low"].rolling(LOW_52W).min()
    df["High_52W_Prev"] = df["Close"].rolling(HIGH_52W).max().shift(1)

    df["Dipped_Below_EMA_220"] = (
        df["Low"].lt(df["EMA_220"])
        .rolling(DIP_LOOKBACK)
        .max()
        .astype(bool)
    )

    df["Filter_Pass"] = (
        (df["SMA_150"] > df["EMA_220"]) &
        (df["Close"] > df["SMA_50"]) &
        (df["SMA_50"] > df["SMA_150"]) &
        (df["Close"] > 1.25 * df["Low_52W"]) &
        (df["Dipped_Below_EMA_220"])
    )

    df["Breakout_52W_High"] = df["Close"] > df["High_52W_Prev"]

    df["Entry_Signal"] = df["Filter_Pass"] & df["Breakout_52W_High"]

    return df


def prepare_all_data(data_folder):
    stock_data = {}

    for file in os.listdir(data_folder):
        if file.endswith(".csv"):
            symbol = file.replace(".csv", "")
            file_path = os.path.join(data_folder, file)

            try:
                df = load_stock_data(file_path)
                df = add_indicators(df)
                stock_data[symbol] = df
            except Exception as e:
                print(f"Skipping {symbol}: {e}")

    return stock_data


def run_backtest(stock_data):
    cash = INITIAL_CAPITAL
    positions = {}
    trades = []
    portfolio_values = []

    all_dates = sorted(
        set(date for df in stock_data.values() for date in df["Date"])
    )

    for current_date in all_dates:
        # -----------------------------
        # 1. Check exits
        # -----------------------------
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
                    "Entry_Date": position["Entry_Date"],
                    "Entry_Price": position["Entry_Price"],
                    "Exit_Date": current_date,
                    "Exit_Price": exit_price,
                    "Shares": position["Shares"],
                    "Invested": position["Invested"],
                    "PnL": pnl,
                    "PnL_%": pnl_pct,
                    "Exit_Reason": exit_reason
                })

                del positions[symbol]

        # -----------------------------
        # 2. Check entries
        # Entries happen on next day's open
        # -----------------------------
        portfolio_value = cash

        for symbol, position in positions.items():
            df = stock_data[symbol]
            row = df[df["Date"] == current_date]
            if not row.empty:
                portfolio_value += position["Shares"] * row.iloc[0]["Close"]

        max_allocation = INITIAL_CAPITAL * MAX_POSITION_PCT

        for symbol, df in stock_data.items():
            if symbol in positions:
                continue

            signal_row = df[df["Date"] == current_date]

            if signal_row.empty:
                continue

            signal_row = signal_row.iloc[0]

            if not signal_row["Entry_Signal"]:
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

            if allocation <= 0:
                continue

            shares = int(allocation // entry_price)

            if shares <= 0:
                continue

            invested = shares * entry_price
            cash -= invested

            positions[symbol] = {
                "Entry_Date": entry_date,
                "Entry_Price": entry_price,
                "Shares": shares,
                "Invested": invested
            }

        # -----------------------------
        # 3. Portfolio value
        # -----------------------------
        portfolio_value = cash

        for symbol, position in positions.items():
            df = stock_data[symbol]
            row = df[df["Date"] == current_date]

            if not row.empty:
                portfolio_value += position["Shares"] * row.iloc[0]["Close"]

        portfolio_values.append({
            "Date": current_date,
            "Portfolio_Value": portfolio_value,
            "Cash": cash,
            "Open_Positions": len(positions)
        })

    # -----------------------------
    # Close open positions at final close
    # -----------------------------
    for symbol, position in list(positions.items()):
        df = stock_data[symbol]
        final_row = df.iloc[-1]
        exit_price = final_row["Close"]
        proceeds = position["Shares"] * exit_price

        pnl = proceeds - position["Invested"]
        pnl_pct = pnl / position["Invested"]

        trades.append({
            "Symbol": symbol,
            "Entry_Date": position["Entry_Date"],
            "Entry_Price": position["Entry_Price"],
            "Exit_Date": final_row["Date"],
            "Exit_Price": exit_price,
            "Shares": position["Shares"],
            "Invested": position["Invested"],
            "PnL": pnl,
            "PnL_%": pnl_pct,
            "Exit_Reason": "Final close"
        })

    trades_df = pd.DataFrame(trades)
    portfolio_df = pd.DataFrame(portfolio_values)

    return trades_df, portfolio_df


def calculate_performance(trades_df, portfolio_df):
    final_value = portfolio_df["Portfolio_Value"].iloc[-1]
    total_return = (final_value - INITIAL_CAPITAL) / INITIAL_CAPITAL

    portfolio_df["Daily_Return"] = portfolio_df["Portfolio_Value"].pct_change()
    portfolio_df["Peak"] = portfolio_df["Portfolio_Value"].cummax()
    portfolio_df["Drawdown"] = (
        portfolio_df["Portfolio_Value"] - portfolio_df["Peak"]
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
        avg_win = wins["PnL_%"].mean() if not wins.empty else 0
        avg_loss = losses["PnL_%"].mean() if not losses.empty else 0

    summary = {
        "Initial Capital": INITIAL_CAPITAL,
        "Final Portfolio Value": round(final_value, 2),
        "Total Return %": round(total_return * 100, 2),
        "Max Drawdown %": round(max_drawdown * 100, 2),
        "Total Trades": len(trades_df),
        "Win Rate %": round(win_rate * 100, 2),
        "Average Win %": round(avg_win * 100, 2),
        "Average Loss %": round(avg_loss * 100, 2)
    }

    return summary, portfolio_df


if __name__ == "__main__":
    stock_data = prepare_all_data(DATA_FOLDER)

    trades_df, portfolio_df = run_backtest(stock_data)
    summary, portfolio_df = calculate_performance(trades_df, portfolio_df)

    print("\nBacktest Summary")
    print("----------------")
    for key, value in summary.items():
        print(f"{key}: {value}")

    trades_df.to_csv("backtest_trades.csv", index=False)
    portfolio_df.to_csv("backtest_portfolio.csv", index=False)

    print("\nFiles saved:")
    print("backtest_trades.csv")
    print("backtest_portfolio.csv")
