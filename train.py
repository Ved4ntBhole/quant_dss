# train.py
from __future__ import annotations

import os
import sys

sys.path.insert(0, os.path.dirname(__file__))

from src.config_loader import get_policy
from src.data_loader import (
    TRAINING_TICKERS,
    fetch_benchmark,
    fetch_yfinance_prices,
    load_kaggle_csv,
    kaggle_to_wide,
)
from src.feature_engineering import auto_label, build_feature_matrix
from src.model_trainer import train_model

_POLICY = get_policy()

USE_KAGGLE = False
KAGGLE_CSV = "data/raw/nifty50_stocks.csv"
MODEL_TYPE = _POLICY.get("model", {}).get("type", "random_forest")
FETCH_PERIOD = _POLICY.get("training_period", "2y")


def main():
    print("=" * 60)
    print("  QuantDSS India — Model Training Pipeline")
    print(f"  Benchmark: {_POLICY.get('benchmark', '^NSEI')}")
    print("=" * 60)

    print("\n[Step 1] Loading price data...")
    benchmark = fetch_benchmark(period=FETCH_PERIOD)

    if USE_KAGGLE:
        print(f"  Source: Kaggle CSV ({KAGGLE_CSV})")
        long_df = load_kaggle_csv(KAGGLE_CSV)
        prices_wide = kaggle_to_wide(long_df)
        benchmark = benchmark.reindex(prices_wide.index).ffill().bfill()
    else:
        print(f"  Source: yfinance ({len(TRAINING_TICKERS)} NSE tickers, period={FETCH_PERIOD})")
        prices_wide = fetch_yfinance_prices(TRAINING_TICKERS, period=FETCH_PERIOD)

    print(f"  Price matrix: {prices_wide.shape[0]} days × {prices_wide.shape[1]} tickers")

    print("\n[Step 2] Computing features...")
    feature_df = build_feature_matrix(prices_wide, benchmark)
    os.makedirs("data/processed", exist_ok=True)
    feature_df.to_csv("data/processed/features.csv")
    print("  Features saved → data/processed/features.csv")

    print("\n[Step 3] Generating risk labels...")
    labels = auto_label(feature_df)
    labels_df = feature_df.copy()
    labels_df["risk_label"] = labels
    labels_df.to_csv("data/processed/labelled_features.csv")
    print("  Labelled features saved → data/processed/labelled_features.csv")

    print(f"\n[Step 4] Training {MODEL_TYPE} classifier...")
    pipeline, le = train_model(feature_df, labels, model_type=MODEL_TYPE, save=True)

    print("\n" + "=" * 60)
    print("  Training complete!")
    print("  Model : models/risk_classifier.pkl")
    print("  Encoder: models/label_encoder.pkl")
    print("\n  Now run:  streamlit run app/main.py")
    print("=" * 60)


if __name__ == "__main__":
    main()