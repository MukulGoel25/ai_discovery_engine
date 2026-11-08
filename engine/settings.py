"""Paths, environment and config loading shared by the pipeline and the app."""
from __future__ import annotations

import hashlib
import os
from functools import lru_cache
from pathlib import Path

import yaml

try:  # .env is optional (Streamlit Cloud uses st.secrets instead)
    from dotenv import load_dotenv

    load_dotenv()
except Exception:  # pragma: no cover
    pass

ROOT = Path(__file__).resolve().parent.parent
CONFIG = ROOT / "config"
DATA = ROOT / "data"
RAW = DATA / "raw"
PROCESSED = DATA / "processed"
GOLDEN = DATA / "golden"
SURVEY = DATA / "survey"
CACHE = DATA / "cache"
EVALS = ROOT / "evals"
for _p in (RAW, PROCESSED, GOLDEN, SURVEY, CACHE, EVALS):
    _p.mkdir(parents=True, exist_ok=True)

CLEAN_FILE = PROCESSED / "clean.parquet"
ITEMS_FILE = PROCESSED / "items.parquet"          # clean + extraction, what the app reads
CLUSTERS_FILE = PROCESSED / "clusters.json"
INSIGHTS_FILE = PROCESSED / "insights.json"
FUNNEL_FILE = PROCESSED / "funnel.json"
DEMO_FLAG = PROCESSED / "DEMO_DATA"                # present => data is synthetic


def env(name: str, default: str | None = None) -> str | None:
    """Read from environment, then Streamlit secrets if available."""
    val = os.getenv(name)
    if val:
        return val
    try:
        import streamlit as st  # noqa: WPS433

        if name in st.secrets:
            return str(st.secrets[name])
    except Exception:
        pass
    return default


def extract_model() -> str:
    """EXTRACT_MODEL from .env, else picked automatically from what the free key can use."""
    if env("EXTRACT_MODEL"):
        return env("EXTRACT_MODEL")
    from . import llm

    return llm.auto_model("extract")


def synth_model() -> str:
    if env("SYNTH_MODEL"):
        return env("SYNTH_MODEL")
    from . import llm

    return llm.auto_model("synth")


@lru_cache(maxsize=1)
def sources_cfg() -> dict:
    return yaml.safe_load((CONFIG / "sources.yaml").read_text(encoding="utf-8"))


@lru_cache(maxsize=1)
def taxonomy_cfg() -> dict:
    return yaml.safe_load((CONFIG / "taxonomy.yaml").read_text(encoding="utf-8"))


def taxonomy_hash() -> str:
    return hashlib.sha1((CONFIG / "taxonomy.yaml").read_bytes()).hexdigest()[:8]


def theme_ids() -> list[str]:
    return [t["id"] for t in taxonomy_cfg()["themes"]]


def theme_labels() -> dict[str, str]:
    return {t["id"]: t["label"] for t in taxonomy_cfg()["themes"]}


def is_demo() -> bool:
    return DEMO_FLAG.exists()
