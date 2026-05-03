# src/data_loader.py
from __future__ import annotations

import os
from typing import List

import numpy as np
import pandas as pd
import yfinance as yf

from src.config_loader import get_policy, get_ticker_universe

_POLICY = get_policy()
_TICKERS_CFG = get_ticker_universe()

TRAINING_TICKERS = _TICKERS_CFG.get("training_tickers", [])
BENCHMARK_TICKER = _POLICY.get("benchmark", "^NSEI")
MIN_RETURN_OBSERVATIONS = int(_POLICY.get("minimum_return_observations", 60))


def _extract_close_frame(raw: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    if raw.empty:
        return pd.DataFrame()

    if isinstance(raw.columns, pd.MultiIndex):
        if "Close" in raw.columns.get_level_values(0):
            close = raw["Close"].copy()
        elif "Adj Close" in raw.columns.get_level_values(0):
            close = raw["Adj Close"].copy()
        else:
            level0 = list(dict.fromkeys(raw.columns.get_level_values(0)))
            close_key = next((k for k in ("Close", "Adj Close") if k in level0), None)
            if close_key is None:
                raise RuntimeError("Could not locate close prices in downloaded data.")
            close = raw[close_key].copy()

        if isinstance(close, pd.Series):
            close = close.to_frame(name=tickers[0] if len(tickers) == 1 else close.name or tickers[0])

        ordered = [t for t in tickers if t in close.columns]
        if ordered:
            close = close[ordered]
        close.index = pd.to_datetime(close.index)
        close.sort_index(inplace=True)
        return close

    if "Close" in raw.columns:
        close = raw["Close"]
    elif "Adj Close" in raw.columns:
        close = raw["Adj Close"]
    else:
        raise RuntimeError("Could not locate Close / Adj Close in downloaded data.")

    if isinstance(close, pd.Series):
        close = close.to_frame(name=tickers[0] if len(tickers) == 1 else "Close")

    if len(tickers) == 1 and close.shape[1] == 1:
        close.columns = [tickers[0]]

    close.index = pd.to_datetime(close.index)
    close.sort_index(inplace=True)
    return close


def fetch_yfinance_prices(
    tickers: list,
    period: str = "2y",
    interval: str = "1d",
    auto_adjust: bool = True,
) -> pd.DataFrame:
    tickers = [t.strip().upper() for t in tickers if str(t).strip()]
    if not tickers:
        raise ValueError("No tickers provided.")

    raw = yf.download(
        tickers,
        period=period,
        interval=interval,
        auto_adjust=auto_adjust,
        progress=False,
        threads=True,
    )
    prices = _extract_close_frame(raw, tickers)
    prices.dropna(axis=1, how="all", inplace=True)
    prices.dropna(axis=0, how="all", inplace=True)
    return prices


def fetch_single_ticker(ticker: str, period: str = "2y") -> pd.DataFrame:
    ticker = ticker.strip().upper()
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data for '{ticker}'. Use .NS (NSE) or .BO (BSE) suffix.")
    df.index = pd.to_datetime(df.index)
    df.sort_index(inplace=True)
    df["Daily_Return"] = df["Close"].pct_change()
    return df


def fetch_benchmark(period: str = "2y") -> pd.Series:
    benchmark = yf.download(BENCHMARK_TICKER, period=period, auto_adjust=True, progress=False)
    if benchmark.empty:
        raise RuntimeError(f"Could not fetch benchmark: {BENCHMARK_TICKER}")

    if isinstance(benchmark.columns, pd.MultiIndex):
        if "Close" in benchmark.columns.get_level_values(0):
            close = benchmark["Close"]
            if isinstance(close, pd.DataFrame):
                close = close.iloc[:, 0]
        else:
            close = benchmark.iloc[:, 0]
    else:
        close = benchmark["Close"]

    close = pd.Series(close).dropna()
    close.index = pd.to_datetime(close.index)
    close.sort_index(inplace=True)
    close.name = "benchmark"
    return close


def load_kaggle_csv(path: str = "data/raw/nifty50_stocks.csv") -> pd.DataFrame:
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Kaggle CSV not found at '{path}'.\n"
            "Download from: https://www.kaggle.com/datasets/rohanrao/nifty50-stock-market-data"
        )

    df = pd.read_csv(path)

    cols = {c.lower(): c for c in df.columns}
    date_col = cols.get("date")
    ticker_col = cols.get("ticker") or cols.get("symbol") or cols.get("name")
    close_col = cols.get("close")

    if not date_col or not ticker_col or not close_col:
        raise ValueError("Kaggle CSV must contain Date, Ticker/Symbol, and Close columns.")

    df = df[[date_col, ticker_col, close_col]].copy()
    df.columns = ["Date", "ticker", "Close"]
    df["Date"] = pd.to_datetime(df["Date"])
    df.sort_values(["ticker", "Date"], inplace=True)
    return df


def kaggle_to_wide(df_long: pd.DataFrame) -> pd.DataFrame:
    wide = df_long.pivot(index="Date", columns="ticker", values="Close")
    wide.sort_index(inplace=True)
    return wide