import os
import re
from datetime import date

import pandas as pd
import streamlit as st
import yfinance as yf


st.set_page_config(page_title="Nifty 50 Data Downloader", layout="wide")

st.title("Nifty 50 Stock Data Downloader")
st.write("Downloads daily Yahoo Finance data for each Nifty 50 symbol and saves separate CSV files.")


INPUT_FILE = "Nifty 50 symbols.csv"
OUTPUT_FOLDER = "data"
DEFAULT_START_DATE = "2025-01-01"


def clean_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", str(name))


def to_yahoo_symbol(symbol):
    symbol = str(symbol).strip().upper()

    if symbol.endswith(".NS"):
        return symbol

    return f"{symbol}.NS"


def read_symbols(file_path):
    df = pd.read_csv(file_path, header=None)
    symbols = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()

    if not symbols:
        raise ValueError("No symbols found in the CSV file.")

    return symbols


def download_data(ticker, start_date):
    return yf.download(
        ticker,
        start=start_date,
        end=date.today().strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False,
    )


if not os.path.exists(INPUT_FILE):
    st.error(f"File not found: {INPUT_FILE}")
    st.stop()

try:
    symbols = read_symbols(INPUT_FILE)
    st.success(f"Loaded {len(symbols)} symbols from {INPUT_FILE}")
except Exception as e:
    st.error(f"Error reading CSV: {e}")
    st.stop()


with st.expander("View symbols"):
    st.write(symbols)


start_date = st.date_input(
    "Start date",
    value=pd.to_datetime(DEFAULT_START_DATE).date()
)


if st.button("Download Data"):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    success = []
    failed = []

    progress = st.progress(0)
    status = st.empty()

    for i, symbol in enumerate(symbols):
        ticker = to_yahoo_symbol(symbol)
        status.write(f"Downloading {ticker}...")

        try:
            data = download_data(ticker, start_date.strftime("%Y-%m-%d"))

            if data.empty:
                failed.append([symbol, ticker, "No data returned"])
            else:
                file_path = os.path.join(
                    OUTPUT_FOLDER,
                    clean_filename(ticker) + ".csv"
                )
                data.to_csv(file_path)
                success.append([symbol, ticker, file_path])

        except Exception as e:
            failed.append([symbol, ticker, str(e)])

        progress.progress((i + 1) / len(symbols))

    st.success(f"Completed. Downloaded data for {len(success)} stocks.")

    if success:
        st.subheader("Successful Downloads")
        st.dataframe(
            pd.DataFrame(success, columns=["Symbol", "Yahoo Ticker", "Saved File"])
        )

    if failed:
        st.subheader("Failed Downloads")
        st.dataframe(
            pd.DataFrame(failed, columns=["Symbol", "Yahoo Ticker", "Reason"])
        )
