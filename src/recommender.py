# src/recommender.py
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass
from typing import Dict, List

import pandas as pd

from src.api_client import fetch_alpha_vantage_quote
from src.config_loader import get_etf_universe, get_policy, get_questionnaire_scoring
from src.data_loader import fetch_benchmark, fetch_yfinance_prices
from src.feature_engineering import compute_features_for_ticker
from src.gemini_client import gemini_request

_POLICY = get_policy()
_QUESTIONNAIRE_SCORING = get_questionnaire_scoring()
_ETF_UNIVERSE = get_etf_universe().get("etfs", [])

def _alpha_key():
    return os.getenv("ALPHA_VANTAGE_API_KEY", "").strip()


def _alpha_url():
    return os.getenv("ALPHA_VANTAGE_API_URL", "https://www.alphavantage.co/query").strip()


def _provider_override():
    return os.getenv("RECOMMENDER_PROVIDER", "").strip().lower()

CURRENCY = "₹"


@dataclass
class PortfolioRecommendation:
    user_risk_profile: str
    summary: str
    sector_allocations: Dict[str, float]
    etf_suggestions: List[Dict]
    key_insights: List[str]
    warning: str = ""
    data_source: str = "computed"
    ai_generated: bool = False


def _validate_risk(risk: str) -> str:
    risk = str(risk).strip().capitalize()
    if risk not in ("Low", "Medium", "High"):
        raise ValueError(f"Invalid risk profile '{risk}'. Expected Low / Medium / High.")
    return risk


def _auto_provider() -> str:
    """
    Detect which recommendation backend to use.
    Reads environment variables dynamically at runtime.
    """

    override = _provider_override()
    gemini_key = os.getenv("GEMINI_API_KEY", "").strip()
    alpha_key = _alpha_key()

    if override == "gemini" and gemini_key:
        return "gemini"

    if override == "alpha_vantage" and alpha_key:
        return "alpha_vantage"

    if override == "static":
        return "static"

    if gemini_key:
        return "gemini"

    if alpha_key:
        return "alpha_vantage"

    return "static"


def score_user_questionnaire(answers: dict) -> str:
    total = 0
    for value in answers.values():
        total += int(_QUESTIONNAIRE_SCORING.get(str(value), 0))

    bands = _POLICY.get("questionnaire_score_bands", {
        "Low": [0, 4],
        "Medium": [5, 9],
        "High": [10, 15],
    })

    for label, bounds in bands.items():
        low, high = bounds
        if low <= total <= high:
            return label
    return "Medium"


def suggest_weight_for_asset(asset_risk: str, user_risk: str, n_assets: int) -> float:
    if n_assets <= 0:
        return 0.0

    order = {"Low": 0, "Medium": 1, "High": 2}
    a = order.get(asset_risk, 1)
    u = order.get(user_risk, 1)
    base = 1.0 / n_assets

    if a == u:
        multiplier = 1.25
    elif abs(a - u) == 1:
        multiplier = 0.85
    else:
        multiplier = 0.55

    return round(base * multiplier, 4)


def _fetch_etf_metrics(period: str = None) -> List[dict]:
    period = period or _POLICY.get("training_period", "2y")
    tickers = [e["ticker"] for e in _ETF_UNIVERSE]

    prices = fetch_yfinance_prices(tickers, period=period, interval="1d")
    benchmark = fetch_benchmark(period=period)

    enriched = []
    for etf in _ETF_UNIVERSE:
        t = etf["ticker"]
        if t not in prices.columns:
            continue

        series = prices[t].dropna()
        if len(series) < int(_POLICY.get("minimum_return_observations", 60)):
            continue

        feats = compute_features_for_ticker(series, benchmark)
        if not feats:
            continue

        enriched.append({**etf, **feats, "full_ticker": t})

    return enriched


def _enrich_with_av_quotes(records: List[dict]) -> List[dict]:
    if not _alpha_key():
        return records

    for item in records:
        try:
            av_symbol = item["ticker"].replace(".NS", "").replace(".BO", "")
            q = fetch_alpha_vantage_quote(av_symbol, _alpha_key(), _alpha_url())
            item["live_price"] = q.get("price")
            item["day_change_pct"] = q.get("price_change_pct")
            item["volume"] = q.get("volume")
        except Exception:
            item.setdefault("live_price", None)
            item.setdefault("day_change_pct", None)
            item.setdefault("volume", None)
        time.sleep(0.2)

    return records


def _normalize(value: float, lo: float, hi: float, invert: bool = False) -> float:
    if hi == lo:
        return 0.5
    x = (value - lo) / (hi - lo)
    x = max(0.0, min(1.0, x))
    return 1.0 - x if invert else x


def _score_etfs(records: List[dict], risk: str, top_n: int = 5) -> List[dict]:
    if not records:
        return []

    vols = [r["annualised_volatility"] for r in records]
    rets = [r["annualised_return"] for r in records]
    betas = [r["beta"] for r in records]
    dds = [r["max_drawdown"] for r in records]

    scored = []
    for r in records:
        v_low = _normalize(r["annualised_volatility"], min(vols), max(vols), invert=True)
        v_high = _normalize(r["annualised_volatility"], min(vols), max(vols), invert=False)
        ret = _normalize(r["annualised_return"], min(rets), max(rets), invert=False)
        b_low = _normalize(r["beta"], min(betas), max(betas), invert=True)
        b_high = _normalize(r["beta"], min(betas), max(betas), invert=False)
        dd_low = _normalize(r["max_drawdown"], min(dds), max(dds), invert=True)
        sharpe = _normalize(r["sharpe_ratio"], -2, 3, invert=False)

        if risk == "Low":
            score = 0.30 * v_low + 0.25 * b_low + 0.25 * dd_low + 0.20 * sharpe
        elif risk == "Medium":
            score = 0.30 * ret + 0.25 * sharpe + 0.20 * v_low + 0.15 * b_low + 0.10 * dd_low
        else:
            score = 0.35 * ret + 0.25 * v_high + 0.20 * b_high + 0.20 * sharpe

        scored.append({**r, "_score": score})

    scored.sort(key=lambda x: x["_score"], reverse=True)
    top = scored[:top_n]
    total = sum(x["_score"] for x in top) or 1.0
    for item in top:
        item["suggested_weight"] = round(item["_score"] / total, 4)
    return top


def _build_sector_allocation(selected: List[dict]) -> Dict[str, float]:
    sectors: Dict[str, float] = {}
    for item in selected:
        cat = item.get("category", "Other")
        sectors[cat] = sectors.get(cat, 0.0) + item["suggested_weight"] * 100
    return {k: round(v, 1) for k, v in sorted(sectors.items(), key=lambda x: x[1], reverse=True)}


def _computed_reason(item: dict) -> str:
    parts = [
        f"2y return {item['annualised_return']:+.1%}",
        f"vol {item['annualised_volatility']:.1%}",
        f"beta vs NIFTY {item['beta']:.2f}",
        f"Sharpe {item['sharpe_ratio']:.2f}",
    ]
    if item.get("live_price") is not None:
        parts.append(f"{CURRENCY}{item['live_price']:.2f}")
    if item.get("day_change_pct") is not None:
        parts.append(f"day {item['day_change_pct']:+.2%}")
    return " | ".join(parts) + "."


def _computed_summary(selected: List[dict], risk: str) -> str:
    avg_vol = sum(i["annualised_volatility"] * i["suggested_weight"] for i in selected)
    avg_ret = sum(i["annualised_return"] * i["suggested_weight"] for i in selected)
    avg_beta = sum(i["beta"] * i["suggested_weight"] for i in selected)
    tickers = ", ".join(i["ticker"].replace(".NS", "").replace(".BO", "") for i in selected)

    return (
        f"For your {risk} risk profile, the model selected {tickers} using live NSE-listed market data "
        f"benchmarked against NIFTY 50. Weighted annualised volatility is {avg_vol:.1%}, "
        f"weighted 2-year return is {avg_ret:.1%}, and portfolio beta versus NIFTY 50 is {avg_beta:.2f}."
    )


def _computed_insights(selected: List[dict], risk: str) -> List[str]:
    avg_vol = sum(i["annualised_volatility"] * i["suggested_weight"] for i in selected)
    best_sh = max(selected, key=lambda x: x["sharpe_ratio"])
    worst_dd = max(selected, key=lambda x: x["max_drawdown"])
    best_ret = max(selected, key=lambda x: x["annualised_return"])

    return [
        f"Portfolio weighted volatility is {avg_vol:.1%}; keep sizing disciplined and avoid concentrating too much in one theme.",
        f"{best_sh['ticker'].replace('.NS','')} has the strongest Sharpe ratio ({best_sh['sharpe_ratio']:.2f}) in the selection.",
        f"{worst_dd['ticker'].replace('.NS','')} has the largest historical drawdown ({worst_dd['max_drawdown']:.1%}); stagger entry if you add it.",
        f"{best_ret['ticker'].replace('.NS','')} delivered the highest 2-year return ({best_ret['annualised_return']:+.1%}).",
        f"The current basket is built for {risk} risk and should be reviewed again when the market regime changes."
    ]


def _gemini_summary(selected: List[dict], risk: str) -> str:
    avg_vol = sum(i["annualised_volatility"] * i["suggested_weight"] for i in selected)
    avg_ret = sum(i["annualised_return"] * i["suggested_weight"] for i in selected)
    avg_beta = sum(i["beta"] * i["suggested_weight"] for i in selected)
    avg_sh = sum(i["sharpe_ratio"] * i["suggested_weight"] for i in selected)
    tickers = ", ".join(i["ticker"].replace(".NS", "").replace(".BO", "") for i in selected)

    prompt = (
        "You are writing for an Indian retail investor.\n"
        "Use ONLY the figures below. Do not invent numbers.\n"
        "Write exactly 3 plain-text sentences, no bullets, no markdown.\n\n"
        f"Risk profile: {risk}\n"
        f"Selected instruments: {tickers}\n"
        f"Weighted annualised volatility: {avg_vol:.1%}\n"
        f"Weighted 2-year return: {avg_ret:.1%}\n"
        f"Weighted beta vs NIFTY 50: {avg_beta:.2f}\n"
        f"Weighted Sharpe ratio: {avg_sh:.2f}\n"
        f"All prices are in INR ({CURRENCY})."
    )
    return gemini_request(prompt)


def _gemini_insights(selected: List[dict], risk: str) -> List[str]:
    lines = "\n".join(
        f"{i['ticker'].replace('.NS','')}: vol={i['annualised_volatility']:.1%}, "
        f"return={i['annualised_return']:.1%}, beta={i['beta']:.2f}, "
        f"sharpe={i['sharpe_ratio']:.2f}, max_dd={i['max_drawdown']:.1%}"
        for i in selected
    )

    prompt = (
        "You are an Indian equity market analyst.\n"
        "Use ONLY the metrics below.\n"
        "Return exactly 5 short lines, one insight per line. No bullets, no numbering, no markdown.\n\n"
        f"Risk profile: {risk}\n\n"
        f"{lines}"
    )
    raw = gemini_request(prompt)
    out = [line.strip("-• \t") for line in raw.splitlines() if line.strip()]
    return out[:5] if out else _computed_insights(selected, risk)


def _gemini_reason(item: dict, risk: str) -> str:
    prompt = (
        "Write one plain sentence of at most 30 words.\n"
        "Use ONLY the metrics below. No invented numbers.\n\n"
        f"Ticker: {item['ticker'].replace('.NS','').replace('.BO','')}\n"
        f"Name: {item['name']}\n"
        f"Risk profile: {risk}\n"
        f"2y return: {item['annualised_return']:.1%}\n"
        f"Volatility: {item['annualised_volatility']:.1%}\n"
        f"Beta vs NIFTY 50: {item['beta']:.2f}\n"
        f"Sharpe: {item['sharpe_ratio']:.2f}\n"
    )
    try:
        return gemini_request(prompt)
    except Exception:
        return _computed_reason(item)


def generate_recommendation(risk: str) -> PortfolioRecommendation:
    risk = _validate_risk(risk)
    provider = _auto_provider()
    top_n = int(_POLICY.get("top_n_recommendations", 5))
    period = _POLICY.get("training_period", "2y")

    records = _fetch_etf_metrics(period=period)
    warning = ""

    if not records:
        return PortfolioRecommendation(
            user_risk_profile=risk,
            summary="No ETF data could be fetched.",
            sector_allocations={},
            etf_suggestions=[],
            key_insights=[],
            warning="No ETF universe data available.",
            data_source=provider,
            ai_generated=False,
        )

    selected = _score_etfs(records, risk, top_n=top_n)
    if not selected:
        return PortfolioRecommendation(
            user_risk_profile=risk,
            summary="No ETF selection could be made.",
            sector_allocations={},
            etf_suggestions=[],
            key_insights=[],
            warning="Insufficient ETF data.",
            data_source=provider,
            ai_generated=False,
        )

    if _alpha_key():
        selected = _enrich_with_av_quotes(selected)

    sector_allocations = _build_sector_allocation(selected)

    ai_generated = False
    if provider == "gemini":
        try:
            summary = _gemini_summary(selected, risk)
            key_insights = _gemini_insights(selected, risk)
            ai_generated = True
        except Exception as exc:
            warning = f"Gemini fallback used: {exc}"
            summary = _computed_summary(selected, risk)
            key_insights = _computed_insights(selected, risk)
    else:
        summary = _computed_summary(selected, risk)
        key_insights = _computed_insights(selected, risk)

    etf_suggestions = []
    for item in selected:
        reason = _gemini_reason(item, risk) if provider == "gemini" else _computed_reason(item)
        etf_suggestions.append({
            "ticker": item["ticker"],
            "full_ticker": item.get("full_ticker", item["ticker"]),
            "name": item["name"],
            "category": item.get("category", "Other"),
            "risk_bucket": item.get("risk_bucket", "Medium"),
            "annualised_volatility": item["annualised_volatility"],
            "annualised_return": item["annualised_return"],
            "sharpe_ratio": item["sharpe_ratio"],
            "beta": item["beta"],
            "max_drawdown": item["max_drawdown"],
            "var_95": item["var_95"],
            "live_price": item.get("live_price"),
            "day_change_pct": item.get("day_change_pct"),
            "volume": item.get("volume"),
            "suggested_weight": item["suggested_weight"],
            "reason": reason,
        })

    data_source = "gemini" if provider == "gemini" else ("alpha_vantage" if provider == "alpha_vantage" else "static")

    return PortfolioRecommendation(
        user_risk_profile=risk,
        summary=summary,
        sector_allocations=sector_allocations,
        etf_suggestions=etf_suggestions,
        key_insights=key_insights,
        warning=warning,
        data_source=data_source,
        ai_generated=ai_generated,
    )