import os
import re
import pandas as pd
import yfinance as yf


INPUT_FILE = "Nifty 50.csv"
OUTPUT_FOLDER = "data"
START_DATE = "2025-01-01"


def clean_filename(name):
    return re.sub(r'[\\/*?:"<>|&]', "_", name).strip()


def convert_to_yahoo_ticker(stock):
    stock = stock.strip().upper()

    if stock.endswith(".NS"):
        return stock

    return stock + ".NS"


def read_stock_list(file_path):
    df = pd.read_csv(file_path)

    stocks = df.iloc[:, 0].dropna().astype(str).str.strip().tolist()

    first_column = df.columns[0].strip()
    if first_column.lower() not in ["symbol", "stock", "stocks", "ticker", "company"]:
        stocks.insert(0, first_column)

    return list(dict.fromkeys(stocks))


def download_stock_data(ticker, start_date):
    data = yf.download(
        ticker,
        start=start_date,
        interval="1d",
        auto_adjust=False,
        progress=False
    )

    if data.empty:
        return None

    data.reset_index(inplace=True)
    return data


def main():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    stocks = read_stock_list(INPUT_FILE)

    print(f"Total stocks found: {len(stocks)}")

    for stock in stocks:
        ticker = convert_to_yahoo_ticker(stock)

        print(f"Downloading: {stock} → {ticker}")

        data = download_stock_data(ticker, START_DATE)

        if data is None:
            print(f"Skipped: No data found for {ticker}")
            continue

        file_name = clean_filename(stock) + ".csv"
        output_path = os.path.join(OUTPUT_FOLDER, file_name)

        data.to_csv(output_path, index=False)

        print(f"Saved: {output_path}")

    print("All downloads completed.")


if __name__ == "__main__":
    main()
