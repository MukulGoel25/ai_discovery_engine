"""Provider-agnostic LLM layer. Default is Google Gemini's FREE tier (no card needed).

Set in .env:
  LLM_PROVIDER = gemini | groq | ollama | anthropic
  GEMINI_API_KEY   (free: aistudio.google.com -> Get API key)
  GROQ_API_KEY     (free: console.groq.com -> API Keys)
  OLLAMA_MODEL     (local, free: install ollama, then `ollama pull llama3.1`)
  ANTHROPIC_API_KEY (paid, optional)
Models are picked automatically from what your key can use; override with EXTRACT_MODEL / SYNTH_MODEL.

Every call asks for JSON, the reply is parsed and then coerced to the schema
(unknown enum values -> "unclear"/"none", lists filtered, numbers clamped), so
downstream code always gets valid records whatever the model.
"""
from __future__ import annotations

import json
import re
import threading
import time

import requests

from .settings import env


class LLMUnavailable(RuntimeError):
    """No key / provider configured."""


class QuotaExhausted(RuntimeError):
    """Free daily quota used up. Rerun later; finished work is cached."""


def provider() -> str:
    return (env("LLM_PROVIDER", "gemini") or "gemini").strip().lower()


def available() -> bool:
    p = provider()
    if p == "gemini":
        return bool(env("GEMINI_API_KEY"))
    if p == "groq":
        return bool(env("GROQ_API_KEY"))
    if p == "anthropic":
        return bool(env("ANTHROPIC_API_KEY"))
    if p == "ollama":
        try:
            return requests.get(_ollama_url() + "/api/tags", timeout=3).ok
        except Exception:  # noqa: BLE001
            return False
    return False


def _need(key: str) -> str:
    v = env(key)
    if not v:
        raise LLMUnavailable(f"{key} is not set. Add it to .env (local) or Streamlit secrets (deployed). See README.")
    return v


# ------------------------------------------------------------------ rate limiting
_lock = threading.Lock()
_last = [0.0]
DEFAULT_RPM = {"gemini": "8", "groq": "25", "ollama": "0", "anthropic": "50"}


def _throttle():
    rpm = float(env("LLM_RPM", DEFAULT_RPM.get(provider(), "10")) or 0)
    if rpm <= 0:
        return
    gap = 60.0 / rpm
    with _lock:
        wait = _last[0] + gap - time.time()
        if wait > 0:
            time.sleep(wait)
        _last[0] = time.time()


# ------------------------------------------------------------------ model discovery
_models_cache: dict[str, list[str]] = {}


def list_models() -> list[str]:
    p = provider()
    if p in _models_cache:
        return _models_cache[p]
    names: list[str] = []
    try:
        if p == "gemini":
            r = requests.get("https://generativelanguage.googleapis.com/v1beta/models",
                             headers={"x-goog-api-key": _need("GEMINI_API_KEY")}, params={"pageSize": 200}, timeout=30)
            r.raise_for_status()
            names = [m["name"].split("/", 1)[1] for m in r.json().get("models", [])
                     if "generateContent" in m.get("supportedGenerationMethods", [])]
        elif p == "groq":
            r = requests.get("https://api.groq.com/openai/v1/models",
                             headers={"Authorization": f"Bearer {_need('GROQ_API_KEY')}"}, timeout=30)
            r.raise_for_status()
            names = [m["id"] for m in r.json().get("data", [])]
        elif p == "ollama":
            r = requests.get(_ollama_url() + "/api/tags", timeout=10)
            names = [m["name"] for m in r.json().get("models", [])]
    except LLMUnavailable:
        raise
    except Exception as e:  # noqa: BLE001
        print(f"  ! could not list models: {e}")
    _models_cache[p] = names
    return names


_SKIP = re.compile(r"image|tts|audio|live|embed|vision|guard|whisper|learnlm|aqa|robotics|computer|native|banana|veo|imagen|gemma",
                   re.I)


def _version(name: str) -> tuple:
    nums = re.findall(r"\d+(?:\.\d+)?", name)
    return tuple(float(n) for n in nums[:2]) if nums else (0.0,)


def auto_model(role: str) -> str:
    """Pick a default from the models this key can use. role = 'extract' (fast) or 'synth' (smarter)."""
    p = provider()
    if p == "anthropic":
        return "claude-haiku-5-5" if role == "extract" else "claude-sonnet-5-5"
    if p == "ollama":
        return env("OLLAMA_MODEL", "llama3.1")
    models = [m for m in list_models() if not _SKIP.search(m)]
    if p == "gemini":
        stable = [m for m in models if "gemini" in m and not re.search(r"exp|preview|latest", m)] or \
                 [m for m in models if "gemini" in m]
        lite = [m for m in stable if "flash-lite" in m]
        flash = [m for m in stable if "flash" in m and "lite" not in m]
        pool = (lite or flash) if role == "extract" else (flash or lite)
        return sorted(pool, key=_version)[-1] if pool else "gemini-2.5-flash"
    if p == "groq":
        big = [m for m in models if "70b" in m or "versatile" in m]
        small = [m for m in models if "8b" in m or "instant" in m]
        pool = (small or big) if role == "extract" else (big or small)
        return sorted(pool)[-1] if pool else (models[0] if models else "llama-3.3-70b-versatile")
    raise LLMUnavailable(f"Unknown LLM_PROVIDER '{p}'")


# ------------------------------------------------------------------ schema helpers
def describe_schema(schema: dict, indent: str = "") -> str:
    """Compact, model-friendly description of a JSON schema."""
    lines = []
    req = set(schema.get("required", []))
    for k, s in schema.get("properties", {}).items():
        t = s.get("type")
        if t == "array" and s.get("items", {}).get("type") == "object":
            lines.append(f'{indent}"{k}": array of objects, each with:')
            lines.append(describe_schema(s["items"], indent + "    "))
            continue
        if t == "array":
            en = s.get("items", {}).get("enum")
            desc = f"array of strings from {en}" if en else "array of strings"
            if s.get("maxItems"):
                desc += f" (max {s['maxItems']})"
        elif "enum" in s:
            desc = f"one of {s['enum']}"
        else:
            desc = t or "string"
            if "minimum" in s:
                desc += f" {s['minimum']}-{s.get('maximum')}"
        if s.get("description"):
            desc += f" -- {s['description']}"
        lines.append(f'{indent}"{k}"{"" if k in req else " (optional)"}: {desc}')
    return "\n".join(lines)


def coerce(schema: dict, obj):
    t = schema.get("type")
    if t == "object":
        obj = obj if isinstance(obj, dict) else {}
        out = {k: coerce(s, obj.get(k)) for k, s in schema.get("properties", {}).items()}
        for k, v in obj.items():  # keep extra keys such as an 'id' the model echoed
            out.setdefault(k, v)
        return out
    if t == "array":
        obj = obj if isinstance(obj, list) else ([] if obj in (None, "") else [obj])
        items = schema.get("items", {})
        if "enum" in items:  # drop unknown values instead of mapping them to a fallback
            low = {e.lower(): e for e in items["enum"]}
            res = list(dict.fromkeys(low[str(v).strip().lower()] for v in obj if str(v).strip().lower() in low))
        else:
            res = [coerce(items, v) for v in obj]
        return res[: schema["maxItems"]] if schema.get("maxItems") else res
    if "enum" in schema:
        en = schema["enum"]
        v = str(obj).strip() if obj is not None else ""
        if v in en:
            return v
        low = {e.lower(): e for e in en}
        if v.lower() in low:
            return low[v.lower()]
        for fb in ("unclear", "none", "unknown", "neutral"):
            if fb in en:
                return fb
        return en[0]
    if t == "boolean":
        return obj if isinstance(obj, bool) else str(obj).strip().lower() in ("true", "yes", "1")
    if t in ("integer", "number"):
        try:
            v = float(obj)
        except (TypeError, ValueError):
            v = float(schema.get("minimum", 0))
        if "minimum" in schema:
            v = max(v, schema["minimum"])
        if "maximum" in schema:
            v = min(v, schema["maximum"])
        return int(round(v)) if t == "integer" else v
    return "" if obj is None else str(obj)


def _parse_json(text: str):
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", (text or "").strip())
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r"\{.*\}", text, re.S)
        if m:
            return json.loads(m.group(0))
        raise


# ------------------------------------------------------------------ provider calls
def _ollama_url() -> str:
    return (env("OLLAMA_URL", "http://localhost:11434") or "").rstrip("/")


class _Retry(Exception):
    def __init__(self, wait: float):
        super().__init__(f"retry in {wait}s")
        self.wait = wait


def _retry_after(r: requests.Response) -> float:
    try:
        for d in r.json().get("error", {}).get("details", []):
            if "retryDelay" in d:
                return float(str(d["retryDelay"]).rstrip("s")) + 1
    except Exception:  # noqa: BLE001
        pass
    try:
        return float(r.headers.get("retry-after", 0)) or 20
    except ValueError:
        return 20


def _check(r: requests.Response):
    if r.status_code == 429:
        if re.search(r"PerDay|per day|daily|RPD|TPD", r.text, re.I):
            raise QuotaExhausted("Free daily quota reached for this model. Rerun later (it resets daily); "
                                 "finished items are cached.")
        raise _Retry(_retry_after(r))
    if r.status_code in (500, 502, 503, 504):
        raise _Retry(10)
    if r.status_code in (401, 403):
        raise LLMUnavailable(f"API key rejected ({r.status_code}): {r.text[:200]}")
    if r.status_code >= 400:
        raise RuntimeError(f"{r.status_code}: {r.text[:300]}")


def _raw(system: str, user: str, model: str, max_tokens: int, temperature: float, want_json: bool) -> str:
    p = provider()
    if p == "gemini":
        cfg = {"temperature": temperature, "maxOutputTokens": max(max_tokens, 8192)}
        if want_json:
            cfg["responseMimeType"] = "application/json"
        r = requests.post(f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                          headers={"x-goog-api-key": _need("GEMINI_API_KEY"), "Content-Type": "application/json"},
                          json={"systemInstruction": {"parts": [{"text": system}]},
                                "contents": [{"role": "user", "parts": [{"text": user}]}],
                                "generationConfig": cfg}, timeout=180)
        _check(r)
        cands = r.json().get("candidates", [])
        if not cands:
            raise RuntimeError(f"Gemini returned no answer: {r.text[:300]}")
        return "".join(part.get("text", "") for part in cands[0].get("content", {}).get("parts", [])
                       if not part.get("thought"))
    if p == "groq":
        body = {"model": model, "temperature": temperature, "max_tokens": max_tokens,
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if want_json:
            body["response_format"] = {"type": "json_object"}
        r = requests.post("https://api.groq.com/openai/v1/chat/completions",
                          headers={"Authorization": f"Bearer {_need('GROQ_API_KEY')}"}, json=body, timeout=180)
        _check(r)
        return r.json()["choices"][0]["message"]["content"]
    if p == "ollama":
        body = {"model": model, "stream": False, "options": {"temperature": temperature},
                "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if want_json:
            body["format"] = "json"
        r = requests.post(_ollama_url() + "/api/chat", json=body, timeout=600)
        _check(r)
        return r.json()["message"]["content"]
    if p == "anthropic":
        import anthropic

        c = anthropic.Anthropic(api_key=_need("ANTHROPIC_API_KEY"), max_retries=4)
        resp = c.messages.create(model=model, max_tokens=max_tokens, temperature=temperature, system=system,
                                 messages=[{"role": "user", "content": user}])
        return "".join(getattr(b, "text", "") for b in resp.content)
    raise LLMUnavailable(f"Unknown LLM_PROVIDER '{p}'")


def _call(system, user, model, max_tokens, temperature, want_json, attempts=6) -> str:
    last = None
    for _ in range(attempts):
        _throttle()
        try:
            return _raw(system, user, model, max_tokens, temperature, want_json)
        except _Retry as rt:
            last = f"rate limited (waited {rt.wait:.0f}s)"
            time.sleep(min(rt.wait, 90))
        except (LLMUnavailable, QuotaExhausted):
            raise
        except requests.RequestException as e:
            last = e
            time.sleep(5)
    raise RuntimeError(f"LLM call failed after {attempts} attempts: {last}")


# ------------------------------------------------------------------ public API
def call_json(system: str, user: str, schema: dict, model: str | None = None, max_tokens: int = 2000,
              temperature: float = 0.0):
    model = model or auto_model("extract")
    sys2 = (system + "\n\nReply with ONLY one JSON object (no prose, no code fences) with these fields:\n"
            + describe_schema(schema))
    last = None
    for _ in range(2):
        txt = _call(sys2, user, model, max_tokens, temperature, True)
        try:
            return coerce(schema, _parse_json(txt))
        except (json.JSONDecodeError, ValueError) as e:
            last = e
    raise RuntimeError(f"Model did not return valid JSON: {last}")


def call_tool(system: str, user: str, tool: dict, model: str | None = None, max_tokens: int = 2000,
              temperature: float = 0.0) -> dict:
    """A 'tool' is just a named JSON schema."""
    return call_json(system, user, tool["input_schema"], model=model, max_tokens=max_tokens, temperature=temperature)


def call_text(system: str, user: str, model: str | None = None, max_tokens: int = 1500, temperature: float = 0.2) -> str:
    return _call(system, user, model or auto_model("synth"), max_tokens, temperature, False).strip()
