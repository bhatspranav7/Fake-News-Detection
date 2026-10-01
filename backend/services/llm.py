"""Provider-agnostic LLM client (Ollama locally, Groq / OpenAI-compatible in the cloud).

Every agent talks to the model through `chat_json`, which asks for strict JSON
and repairs the common failure modes (code fences, trailing prose).
"""
from __future__ import annotations

import json
import re
import time
from typing import Any

import requests

from backend.core import config


class LLMError(RuntimeError):
    pass


def describe() -> dict[str, Any]:
    if config.LLM_PROVIDER == "groq":
        return {"provider": "groq", "model": config.GROQ_MODEL}
    if config.LLM_PROVIDER == "openai":
        return {"provider": "openai", "model": config.OPENAI_MODEL}
    if config.LLM_PROVIDER == "none":
        return {"provider": "none", "model": ""}
    return {"provider": "ollama", "model": config.OLLAMA_MODEL}


def available(timeout: float = 10.0) -> bool:
    """Cheap liveness probe used by /health and to decide deep-mode fallback."""
    try:
        if config.LLM_PROVIDER == "ollama":
            r = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=timeout)
            names = [m.get("name", "") for m in r.json().get("models", [])]
            return r.ok and any(n.split(":")[0] == config.OLLAMA_MODEL.split(":")[0] for n in names)
        if config.LLM_PROVIDER == "groq":
            return bool(config.GROQ_API_KEY)
        if config.LLM_PROVIDER == "openai":
            return bool(config.OPENAI_API_KEY)
    except Exception:  # noqa: BLE001
        return False
    return False


def chat(system: str, user: str, temperature: float = 0.1, max_tokens: int = 900) -> str:
    if config.LLM_PROVIDER == "ollama":
        return _ollama(system, user, temperature, max_tokens)
    if config.LLM_PROVIDER == "groq":
        return _openai_compat(config.GROQ_BASE_URL, config.GROQ_API_KEY, config.GROQ_MODEL,
                              system, user, temperature, max_tokens)
    if config.LLM_PROVIDER == "openai":
        return _openai_compat(config.OPENAI_BASE_URL, config.OPENAI_API_KEY, config.OPENAI_MODEL,
                              system, user, temperature, max_tokens)
    raise LLMError("LLM_PROVIDER is 'none'")


def chat_json(system: str, user: str, schema_hint: str, retries: int = 2) -> dict[str, Any]:
    """Ask for JSON only; parse robustly; retry once with a stricter nudge."""
    sys_prompt = (f"{system}\n\nRespond with ONLY a single JSON object matching this shape, "
                  f"no markdown, no commentary:\n{schema_hint}")
    last_err: Exception | None = None
    for attempt in range(retries + 1):
        text = chat(sys_prompt if attempt == 0 else sys_prompt + "\nSTRICT: output raw JSON only.",
                    user)
        try:
            return _extract_json(text)
        except ValueError as exc:
            last_err = exc
            time.sleep(0.3)
    raise LLMError(f"model did not return valid JSON: {last_err}")


# ---------------------------------------------------------------- providers
def _ollama(system: str, user: str, temperature: float, max_tokens: int) -> str:
    try:
        r = requests.post(
            f"{config.OLLAMA_HOST}/api/chat",
            json={
                "model": config.OLLAMA_MODEL,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "stream": False,
                "format": "json",
                "options": {"temperature": temperature, "num_predict": max_tokens},
                "keep_alive": "10m",
            },
            timeout=config.LLM_TIMEOUT,
        )
        r.raise_for_status()
        data = r.json()
    except requests.RequestException as exc:
        raise LLMError(f"Ollama request failed: {exc}") from exc
    if "error" in data:
        raise LLMError(f"Ollama error: {data['error']}")
    return data.get("message", {}).get("content", "")


def _openai_compat(base: str, key: str, model: str, system: str, user: str,
                   temperature: float, max_tokens: int) -> str:
    if not key:
        raise LLMError(f"{config.LLM_PROVIDER}: no API key configured")
    try:
        r = requests.post(
            f"{base}/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": model,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}],
                "temperature": temperature,
                "max_tokens": max_tokens,
                "response_format": {"type": "json_object"},
            },
            timeout=config.LLM_TIMEOUT,
        )
        if not r.ok:
            raise LLMError(f"Chat API {r.status_code} for model '{model}': {r.text[:300]}")
        data = r.json()
    except requests.RequestException as exc:
        raise LLMError(f"Chat API request failed: {exc}") from exc
    return data["choices"][0]["message"]["content"] or ""


# ------------------------------------------------------------------ parsing
_FENCE = re.compile(r"```(?:json)?\s*(.*?)```", re.S)


def _extract_json(text: str) -> dict[str, Any]:
    text = text.strip()
    m = _FENCE.search(text)
    if m:
        text = m.group(1).strip()
    try:
        obj = json.loads(text)
        if isinstance(obj, dict):
            return obj
    except json.JSONDecodeError:
        pass
    # Find the outermost {...}
    start, end = text.find("{"), text.rfind("}")
    if start != -1 and end > start:
        obj = json.loads(text[start:end + 1])
        if isinstance(obj, dict):
            return obj
    raise ValueError(f"no JSON object in: {text[:120]!r}")
