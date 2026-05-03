# src/feature_engineering.py
from __future__ import annotations

import numpy as np
import pandas as pd

from src.config_loader import get_policy

_POLICY = get_policy()

TRADING_DAYS = int(_POLICY.get("trading_days_per_year", 252))
MIN_RETURN_OBSERVATIONS = int(_POLICY.get("minimum_return_observations", 60))
MIN_TRAINING_OBSERVATIONS = int(_POLICY.get("minimum_training_observations", 120))
RISK_FREE_RATE = float(_POLICY.get("risk_free_rate", 0.065))
BETA_LOW = float(_POLICY.get("beta_thresholds", {}).get("low", 0.7))
BETA_HIGH = float(_POLICY.get("beta_thresholds", {}).get("high", 1.3))

FEATURE_COLS = [
    "annualised_volatility",
    "annualised_return",
    "sharpe_ratio",
    "beta",
    "max_drawdown",
    "var_95",
    "downside_deviation",
]


def annualised_volatility(returns: pd.Series) -> float:
    return float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS))


def annualised_return(returns: pd.Series) -> float:
    return float(returns.mean() * TRADING_DAYS)


def sharpe_ratio(returns: pd.Series, risk_free: float = None) -> float:
    risk_free = RISK_FREE_RATE if risk_free is None else risk_free
    mu = annualised_return(returns)
    sigma = annualised_volatility(returns)
    if sigma == 0 or np.isnan(sigma):
        return 0.0
    return float((mu - risk_free) / sigma)


def beta(returns: pd.Series, benchmark_returns: pd.Series) -> float:
    aligned = pd.concat([returns, benchmark_returns], axis=1).dropna()
    if len(aligned) < 30:
        return 1.0
    x = aligned.iloc[:, 0].astype(float)
    y = aligned.iloc[:, 1].astype(float)
    var_bench = np.var(y, ddof=1)
    if var_bench == 0 or np.isnan(var_bench):
        return 1.0
    cov_matrix = np.cov(x, y)
    return float(cov_matrix[0, 1] / var_bench)


def max_drawdown(prices: pd.Series) -> float:
    prices = pd.Series(prices).dropna()
    if prices.empty:
        return 0.0
    roll_max = prices.cummax()
    daily_drawdown = (prices - roll_max) / roll_max
    return float(abs(daily_drawdown.min()))


def value_at_risk(returns: pd.Series, confidence: float = 0.95) -> float:
    returns = pd.Series(returns).dropna()
    if returns.empty:
        return 0.0
    return float(abs(np.percentile(returns, (1 - confidence) * 100)))


def downside_deviation(returns: pd.Series, mar: float = 0.0) -> float:
    returns = pd.Series(returns).dropna()
    downside = returns[returns < mar]
    if len(downside) == 0:
        return 0.0
    return float(downside.std(ddof=1) * np.sqrt(TRADING_DAYS))


def compute_features_for_ticker(prices: pd.Series, benchmark_prices: pd.Series) -> dict | None:
    prices = pd.Series(prices).dropna()
    benchmark_prices = pd.Series(benchmark_prices).dropna()

    returns = prices.pct_change().dropna()
    bench_returns = benchmark_prices.pct_change().dropna()

    if len(returns) < MIN_RETURN_OBSERVATIONS:
        return None

    return {
        "annualised_volatility": annualised_volatility(returns),
        "annualised_return": annualised_return(returns),
        "sharpe_ratio": sharpe_ratio(returns),
        "beta": beta(returns, bench_returns),
        "max_drawdown": max_drawdown(prices),
        "var_95": value_at_risk(returns, 0.95),
        "downside_deviation": downside_deviation(returns),
    }


def build_feature_matrix(prices_wide: pd.DataFrame, benchmark_prices: pd.Series) -> pd.DataFrame:
    records = []
    for ticker in prices_wide.columns:
        prices = prices_wide[ticker].dropna()
        if len(prices) < MIN_TRAINING_OBSERVATIONS:
            continue
        feats = compute_features_for_ticker(prices, benchmark_prices)
        if feats is not None:
            feats["ticker"] = ticker
            records.append(feats)

    if not records:
        empty = pd.DataFrame(columns=FEATURE_COLS)
        empty.index.name = "ticker"
        return empty

    df = pd.DataFrame(records).set_index("ticker")
    df.dropna(inplace=True)
    return df


def _label_from_score(score: float, bands: dict) -> str:
    for label, bounds in bands.items():
        low, high = bounds
        if low <= score <= high:
            return label
    return "Medium"


def auto_label(feature_df: pd.DataFrame) -> pd.Series:
    if feature_df.empty:
        return pd.Series(dtype="object")

    bands = _POLICY.get("auto_label_score_bands", {
        "Low": [0, 2],
        "Medium": [3, 5],
        "High": [6, 8],
    })

    df = feature_df.copy()

    def qscore(series: pd.Series, higher_is_riskier: bool = True) -> pd.Series:
        s = series.astype(float)
        q33 = s.quantile(0.33)
        q66 = s.quantile(0.66)
        if higher_is_riskier:
            return pd.cut(
                s,
                bins=[-np.inf, q33, q66, np.inf],
                labels=[0, 1, 2],
                include_lowest=True,
            ).astype(float)
        return pd.cut(
            s,
            bins=[-np.inf, q33, q66, np.inf],
            labels=[2, 1, 0],
            include_lowest=True,
        ).astype(float)

    beta_scores = pd.Series(1.0, index=df.index)
    beta_scores[df["beta"] < BETA_LOW] = 0
    beta_scores[(df["beta"] >= BETA_LOW) & (df["beta"] <= BETA_HIGH)] = 1
    beta_scores[df["beta"] > BETA_HIGH] = 2

    score = (
        qscore(df["annualised_volatility"], True).fillna(1)
        + beta_scores.fillna(1)
        + qscore(df["max_drawdown"], True).fillna(1)
        + qscore(df["var_95"], True).fillna(1)
    )

    labels = score.apply(lambda s: _label_from_score(s, bands))
    return labels