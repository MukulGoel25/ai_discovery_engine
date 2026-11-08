"""Clean raw items: scrub PII, detect language, dedupe, pre-filter for SIP relevance.

The pre-filter is deliberately loose (keywords); the LLM makes the final relevance call.
"""
from __future__ import annotations

import json
import re

import pandas as pd

from .settings import CLEAN_FILE, FUNNEL_FILE, RAW, sources_cfg

PII = [
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), "[email]"),
    (re.compile(r"\b[A-Z]{5}\d{4}[A-Z]\b"), "[pan]"),
    (re.compile(r"(?:\+?91[\s-]?)?\b[6-9]\d{9}\b"), "[phone]"),
    (re.compile(r"\b\d{9,18}\b"), "[number]"),
]
HINGLISH = set("hai hain nahi nahin kya kyu kyun karu karo kar raha rahi tha thi mera meri mujhe aap "
               "paisa paise bhai yaar kab kaise ab bhi sab accha acha band chalu".split())


def scrub(text: str) -> str:
    for pat, repl in PII:
        text = pat.sub(repl, text)
    return text


def detect_lang(text: str) -> str:
    if re.search(r"[\u0900-\u097F]", text):
        return "hi"
    words = re.findall(r"[a-z]+", text.lower())
    if words and sum(w in HINGLISH for w in words) / len(words) > 0.08:
        return "hinglish"
    return "en"


def _kw_regex(keywords: list[str]) -> re.Pattern:
    parts = []
    for k in keywords:
        k = k.strip().lower()
        if not k:
            continue
        esc = re.escape(k)
        parts.append(rf"\b{esc}s?\b" if re.fullmatch(r"[a-z]{2,4}", k) else esc)
    return re.compile("|".join(parts), re.IGNORECASE)


def _norm_key(text: str) -> str:
    t = re.sub(r"[^a-z\u0900-\u097F ]", " ", text.lower())
    return re.sub(r"\s+", " ", t).strip()[:160]


def load_raw() -> pd.DataFrame:
    rows = []
    for f in RAW.glob("*.jsonl"):
        with f.open(encoding="utf-8") as fh:
            rows += [json.loads(line) for line in fh if line.strip()]
    if not rows:
        raise SystemExit("No raw data. Run `python pipeline.py collect` first.")
    return pd.DataFrame(rows)


def run() -> pd.DataFrame:
    cfg = sources_cfg()
    df = load_raw()
    funnel = {"raw": len(df)}
    raw_by_source = df["source"].value_counts().to_dict()
    df["text"] = df["text"].fillna("").astype(str)
    df["title"] = df["title"].fillna("").astype(str)
    df["full_text"] = (df["title"].where(df["title"].str.len() > 0, "") + "\n" + df["text"]).str.strip()

    # 1. drop empty / too short, PII scrub
    df = df[df["full_text"].str.len() >= int(cfg["limits"].get("min_chars", 25))].copy()
    df["full_text"] = df["full_text"].map(scrub)
    funnel["after_length_filter"] = len(df)

    # 2. date filter
    date_from = str(cfg.get("date_from", "2000-01-01"))
    has_date = df["date"].fillna("").astype(str).str.len() >= 10
    df = df[~has_date | (df["date"].astype(str) >= date_from)].copy()
    funnel["after_date_filter"] = len(df)

    # 3. dedupe (exact + near-duplicate on normalised prefix)
    df["dedupe_key"] = df["full_text"].map(_norm_key)
    df = df.drop_duplicates("id").drop_duplicates("dedupe_key").copy()
    funnel["after_dedupe"] = len(df)

    # 4. keyword pre-filter. Reddit/TradingQ&A results were already query-matched,
    #    so their comments only need to be substantive; reviews/YouTube need a keyword.
    kw = _kw_regex(cfg.get("keywords", []))
    df["kw_hit"] = df["full_text"].str.contains(kw) | df["context"].fillna("").str.contains(kw)
    threaded = df["source"].isin(["reddit", "tradingqna", "survey"])
    df = df[df["kw_hit"] | (threaded & (df["full_text"].str.len() >= 40))].copy()
    funnel["after_keyword_filter"] = len(df)

    # 5. language + cap
    df["lang"] = df["full_text"].map(detect_lang)
    cap = int(cfg["limits"].get("max_items_for_llm", 4000))
    if len(df) > cap:  # keep a source-balanced sample
        quota = (df["source"].value_counts() * cap / len(df)).clip(lower=1).astype(int).to_dict()
        df = df.sample(frac=1, random_state=7)
        df = df[df.groupby("source").cumcount() < df["source"].map(quota)]
    funnel["sent_to_llm"] = len(df)
    df = df.drop(columns=["dedupe_key"]).reset_index(drop=True)
    df.to_parquet(CLEAN_FILE, index=False)
    by_source = df["source"].value_counts().to_dict()
    FUNNEL_FILE.write_text(json.dumps({"funnel": funnel, "raw_by_source": raw_by_source, "by_source": by_source}, indent=2), encoding="utf-8")
    print(json.dumps(funnel, indent=2))
    print("by source:", by_source)
    return df
