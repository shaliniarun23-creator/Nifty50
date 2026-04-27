# Nifty 50 Yahoo Finance Data Downloader

This project downloads daily stock price data for Nifty 50 stocks from Yahoo Finance and saves each stock's data as a separate CSV file.

## Features

- Reads stock symbols from a CSV file
- Downloads daily OHLCV data from Yahoo Finance
- Starts downloading data from 1 January 2025
- Saves each stock as a separate CSV file
- Creates a `data` folder automatically

## Project Structure

```text
nifty50-yahoo-downloader/
│
├── app.py
├── requirements.txt
├── README.md
├── .gitignore
├── Nifty 50(1).csv
└── data/
