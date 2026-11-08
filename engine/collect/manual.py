"""Manual / other public sources and survey free text.

Drop any CSV into data/raw/manual/ with at least a `text` column (optional: date, url,
platform, source). Use it for X posts, Quora answers, LinkedIn comments, etc.
Survey open-text answers are imported from data/survey/responses.csv.
"""
from __future__ import annotations

import pandas as pd

from ..settings import RAW, SURVEY
from .common import make_item


def collect(cfg: dict) -> list[dict]:
    items: list[dict] = []
    folder = RAW / "manual"
    folder.mkdir(exist_ok=True)
    for f in folder.glob("*.csv"):
        df = pd.read_csv(f)
        if "text" not in df.columns:
            print(f"  manual: {f.name} has no 'text' column, skipped")
            continue
        for _, r in df.iterrows():
            items.append(make_item(str(r.get("source", "manual") or "manual"), str(r["text"]),
                                   platform=str(r.get("platform", "generic") or "generic"),
                                   date=str(r.get("date", "") or ""), url=str(r.get("url", "") or "")))
        print(f"  manual: {f.name} -> {len(df)} rows")
    survey = SURVEY / "responses.csv"
    if survey.exists():
        df = pd.read_csv(survey)
        text_cols = [c for c in df.columns if "anything else" in c.lower()]
        for c in text_cols:
            for v in df[c].dropna().astype(str):
                if len(v.strip()) > 10:
                    items.append(make_item("survey", v, platform="generic"))
        print(f"  survey free text: {sum(df[c].notna().sum() for c in text_cols)} answers")
    return items
