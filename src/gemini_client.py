# src/gemini_client.py
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import requests

from src.config_loader import get_policy

_POLICY = get_policy()
_GEMINI_TIMEOUT        = int(_POLICY.get("api", {}).get("gemini_timeout", 30))
_GEMINI_TEMPERATURE    = float(_POLICY.get("api", {}).get("gemini_temperature", 0.35))
_GEMINI_MAX_OUTPUT_TOKENS = int(_POLICY.get("api", {}).get("gemini_max_output_tokens", 900))


def gemini_request(
    prompt: str,
    *,
    temperature: Optional[float] = None,
    max_output_tokens: Optional[int] = None,
    model: Optional[str] = None,
) -> str:
    # Read key and model at call time so .env loaded in main.py takes effect
    api_key       = os.getenv("GEMINI_API_KEY", "").strip()
    default_model = os.getenv("GEMINI_MODEL", "gemini-2.5-flash").strip()

    if not api_key:
        raise RuntimeError("GEMINI_API_KEY is missing.")

    chosen_model = (model or default_model).strip()
    url = (
        f"https://generativelanguage.googleapis.com/v1beta"
        f"/models/{chosen_model}:generateContent"
    )

    payload: Dict[str, Any] = {
        "contents": [{"parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature":    _GEMINI_TEMPERATURE if temperature is None else temperature,
            "maxOutputTokens": _GEMINI_MAX_OUTPUT_TOKENS if max_output_tokens is None else max_output_tokens,
        },
    }

    response = requests.post(
        url,
        params={"key": api_key},
        json=payload,
        timeout=_GEMINI_TIMEOUT,
    )
    response.raise_for_status()

    data = response.json()
    try:
        candidates = data.get("candidates", [])
        if not candidates:
            raise RuntimeError("Gemini returned no candidates.")
        content = candidates[0]["content"]
        parts   = content.get("parts", [])
        text    = "".join(part.get("text", "") for part in parts).strip()
        if not text:
            raise RuntimeError("Gemini returned empty text.")
        return text
    except Exception as exc:
        raise RuntimeError(f"Unexpected Gemini response format: {exc}") from exc