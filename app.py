import os
import re
from datetime import date

import pandas as pd
import streamlit as st
import yfinance as yf


# -----------------------------
# App Config
# -----------------------------
st.set_page_config(
    page_title="Nifty 50 Stock Data Downloader",
    layout="wide"
)

st.title("Nifty 50 Stock Data Downloader")
st.write("Download daily Yahoo Finance data for Nifty 50 stocks and save each stock as a separate CSV file.")


# -----------------------------
# Constants
# -----------------------------
INPUT_FILE = "Nifty 50.csv"
OUTPUT_FOLDER = "data"
START_DATE = "2025-01-01"


# -----------------------------
# Helper Functions
# -----------------------------
def clean_filename(name):
    return re.sub(r'[\\/*?:"<>|]', "_", str(name)).strip()


def convert_to_yahoo_ticker(stock):
    stock = str(stock).strip().upper()

    if stock.endswith(".NS"):
        return stock

    return stock + ".NS"


def read_stock_list(file_path):
    df = pd.read_csv(file_path)

    if df.empty:
        raise ValueError("CSV file is empty.")

    stocks = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()

    if not stocks:
        raise ValueError("No stock names found in the first column of the CSV.")

    return stocks


def download_stock_data(ticker, start_date):
    data = yf.download(
        ticker,
        start=start_date,
        end=date.today().strftime("%Y-%m-%d"),
        progress=False,
        auto_adjust=False
    )

    return data


# -----------------------------
# Main App
# -----------------------------
st.subheader("Input File Check")

if not os.path.exists(INPUT_FILE):
    st.error(f"Input file not found: {INPUT_FILE}")
    st.stop()

st.success(f"Input file found: {INPUT_FILE}")

try:
    stock_list = read_stock_list(INPUT_FILE)
    st.write(f"Total stocks found: **{len(stock_list)}**")

    with st.expander("View stock list"):
        st.write(stock_list)

except Exception as e:
    st.error(f"Error reading stock list: {e}")
    st.stop()


st.subheader("Download Settings")

start_date = st.date_input(
    "Select start date",
    value=pd.to_datetime(START_DATE).date()
)

if st.button("Download Nifty 50 Data"):
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    success_count = 0
    failed_stocks = []

    progress_bar = st.progress(0)
    status_text = st.empty()

    for i, stock in enumerate(stock_list):
        ticker = convert_to_yahoo_ticker(stock)
        status_text.write(f"Downloading: {ticker}")

        try:
            data = download_stock_data(ticker, start_date.strftime("%Y-%m-%d"))

            if data.empty:
                failed_stocks.append((stock, "No data returned"))
            else:
                file_name = clean_filename(ticker) + ".csv"
                file_path = os.path.join(OUTPUT_FOLDER, file_name)
                data.to_csv(file_path)
                success_count += 1

        except Exception as e:
            failed_stocks.append((stock, str(e)))

        progress_bar.progress((i + 1) / len(stock_list))

    st.success(f"Download completed. Successfully downloaded: {success_count} stocks.")

    if failed_stocks:
        st.warning("Some stocks failed:")
        failed_df = pd.DataFrame(failed_stocks, columns=["Stock", "Reason"])
        st.dataframe(failed_df)

    st.subheader("Downloaded Files")

    downloaded_files = os.listdir(OUTPUT_FOLDER)

    if downloaded_files:
        st.write(f"Files saved inside `{OUTPUT_FOLDER}/` folder:")
        st.dataframe(pd.DataFrame(downloaded_files, columns=["File Name"]))
    else:
        st.write("No files downloaded.")
