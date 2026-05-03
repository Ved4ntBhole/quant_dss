# src/api_client.py
from __future__ import annotations

from functools import lru_cache
from typing import Dict, Any

import requests

from src.config_loader import get_policy

_POLICY = get_policy()
_AV_TIMEOUT = int(_POLICY.get("api", {}).get("alpha_vantage_timeout", 15))


def _validate_alpha_vantage_response(data: dict, symbol: str) -> dict:
    if not isinstance(data, dict):
        raise RuntimeError(f"Alpha Vantage returned invalid data for {symbol}.")
    if "Note" in data:
        raise RuntimeError(f"Alpha Vantage rate limit notice: {data['Note']}")
    if "Error Message" in data:
        raise RuntimeError(f"Alpha Vantage error: {data['Error Message']}")
    return data


@lru_cache(maxsize=64)
def fetch_alpha_vantage_json(
    function: str,
    symbol: str,
    api_key: str,
    api_url: str = "https://www.alphavantage.co/query",
    **params,
) -> dict:
    if not api_key:
        raise ValueError("Alpha Vantage API key is required.")

    payload: Dict[str, Any] = {
        "function": function,
        "symbol": symbol,
        "apikey": api_key,
        **params,
    }

    response = requests.get(api_url, params=payload, timeout=_AV_TIMEOUT)
    response.raise_for_status()
    data = response.json()
    return _validate_alpha_vantage_response(data, symbol)


@lru_cache(maxsize=64)
def fetch_alpha_vantage_quote(
    symbol: str,
    api_key: str,
    api_url: str = "https://www.alphavantage.co/query",
) -> dict:
    raw = fetch_alpha_vantage_json("GLOBAL_QUOTE", symbol, api_key, api_url)
    quote = raw.get("Global Quote") or raw.get("Global Quote ")
    if not quote or not isinstance(quote, dict):
        raise RuntimeError(f"Alpha Vantage quote response for {symbol} is missing or malformed.")

    def _float(key: str):
        value = quote.get(key, "")
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _percent(key: str):
        value = quote.get(key, "")
        if not isinstance(value, str):
            return None
        value = value.strip()
        if not value:
            return None
        return float(value.replace("%", "")) / 100.0

    return {
        "symbol": symbol,
        "price": _float("05. price"),
        "previous_close": _float("08. previous close"),
        "price_change": _float("09. change"),
        "price_change_pct": _percent("10. change percent"),
        "volume": _float("06. volume"),
    }