# src/config_loader.py
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIG_DIR = PROJECT_ROOT / "config"


@lru_cache(maxsize=32)
def load_json_config(filename: str) -> Dict[str, Any]:
    path = CONFIG_DIR / filename
    if not path.exists():
        raise FileNotFoundError(f"Missing config file: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@lru_cache(maxsize=1)
def get_policy() -> Dict[str, Any]:
    return load_json_config("investment_policy.json")


@lru_cache(maxsize=1)
def get_ticker_universe() -> Dict[str, Any]:
    return load_json_config("ticker_universe.json")


@lru_cache(maxsize=1)
def get_questionnaire_scoring() -> Dict[str, Any]:
    return load_json_config("questionnaire_scoring.json")


@lru_cache(maxsize=1)
def get_etf_universe() -> Dict[str, Any]:
    return load_json_config("etf_universe.json")