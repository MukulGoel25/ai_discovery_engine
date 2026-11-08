"""Write short, grounded findings for each research question in the brief.

The model only sees computed numbers + a sample of real quotes, must cite item IDs, and every
cited ID is verified against the sample. Unverifiable citations are dropped and counted.
"""
from __future__ import annotations

import json

import pandas as pd

from . import llm
from .analyze import (continuers_vs_drop, dist, driver_compare, has_theme, is_drop, journey_leaks,
                      opportunity_table, relevant, theme_by)
from .settings import INSIGHTS_FILE, ITEMS_FILE, synth_model, theme_labels

QUESTIONS = {
    "why_start": "Why do people start a SIP, and what do they expect to happen?",
    "what_changes": "What changes in their behaviour in the months after starting?",
    "why_stop": "What leads investors to pause, reduce or stop a SIP?",
    "what_check": "What do investors check, and how often, once they have started?",
    "confusion": "What confuses or worries them inside the app?",
    "outside_app": "What do they do outside the app when they feel unsure?",
    "continuers": "What separates investors who continue from those who quietly drop off?",
    "segments": "How do these patterns differ across user segments?",
    "unmet_needs": "What unmet needs emerge consistently across conversations?",
}

TOOL = {
    "name": "write_findings",
    "description": "Write 2-4 evidence-backed findings.",
    "input_schema": {"type": "object", "properties": {"findings": {"type": "array", "items": {
        "type": "object", "properties": {
            "headline": {"type": "string", "description": "One sentence stating the finding. Include a number from the stats."},
            "detail": {"type": "string", "description": "1-2 sentences of nuance or caveat."},
            "evidence_ids": {"type": "array", "items": {"type": "string"}, "description": "2-4 item IDs from the sample"},
        }, "required": ["headline", "detail", "evidence_ids"]}}},
        "required": ["findings"]},
}


def _fmt(d: pd.DataFrame, col: str, k: int = 8) -> str:
    return "; ".join(f"{r[col]}: {r['n']} ({r['share']:.0%})" for _, r in d.head(k).iterrows())


def _stats(key: str, df: pd.DataFrame) -> tuple[str, pd.DataFrame]:
    labels = theme_labels()
    drop = df[is_drop(df)]
    if key == "why_start":
        return (f"why_started: {_fmt(dist(df, 'why_started'), 'why_started')}\n"
                f"expectation: {_fmt(dist(df, 'expectation'), 'expectation')}\n"
                f"influencer: {_fmt(dist(df, 'influencer'), 'influencer')}"), df[df["why_started"] != "unclear"]
    if key == "what_changes":
        jl = journey_leaks(df)
        return ("drop events by stage: " + "; ".join(f"{r.label}: {r.drops} (top theme {labels.get(r.top_theme, '-')})"
                                                     for r in jl.itertuples()) +
                f"\nevents: {_fmt(dist(df, 'event'), 'event')}"), drop
    if key == "why_stop":
        opp = opportunity_table(df)
        return ("among drop events, primary themes: " + _fmt(dist(drop, "primary_theme"), "primary_theme") +
                "\nopportunity scores: " + "; ".join(f"{r.label}={r.score:.0f}" for r in opp.head(8).itertuples())), drop
    if key == "what_check":
        return (f"checks: {_fmt(dist(df, 'checks', True), 'checks')}\n"
                f"check_frequency: {_fmt(dist(df, 'check_frequency'), 'check_frequency')}"), df[df["checks"].map(len) > 0]
    if key == "confusion":
        return (f"confusions: {_fmt(dist(df, 'confusions', True), 'confusions')}\n"
                f"emotions: {_fmt(dist(df, 'emotion'), 'emotion')}"), df[df["confusions"].map(len) > 0]
    if key == "outside_app":
        return f"outside_app_actions: {_fmt(dist(df, 'outside_app_actions', True), 'outside_app_actions')}", \
            df[df["outside_app_actions"].map(len) > 0]
    if key == "continuers":
        cvd, nc, nd = continuers_vs_drop(df)
        dc = driver_compare(df)
        return (f"continuers n={nc}, drop-offs n={nd}\ntheme share gap (drop minus continuers): " +
                "; ".join(f"{r.theme}: {r.drop_offs:.0%} vs {r.continuers:.0%}" for r in cvd.head(6).itertuples()) +
                "\ndrivers (continuers vs drop-offs): " +
                "; ".join(f"{r.driver}: {r.continuers:.0%} vs {r.drop_offs:.0%}" for r in dc.itertuples())), \
            df[df["event"].isin(["continuing", "restarted"]) | is_drop(df)]
    if key == "segments":
        parts = []
        for col in ["income", "age_band", "influencer", "money_source"]:
            tb = theme_by(df, col)
            if len(tb):
                top = tb.sort_values("share", ascending=False).groupby(col).head(2)
                parts.append(f"{col}: " + "; ".join(f"{r[col]} -> {r['theme']} {r['share']:.0%} (n={r['n']})"
                                                    for _, r in top.iterrows()))
        return "\n".join(parts) or "too few segment signals", df[(df["income"] != "unclear") | (df["age_band"] != "unclear")]
    if key == "unmet_needs":
        return ("primary themes overall: " + _fmt(dist(df, "primary_theme"), "primary_theme")), \
            df[df["unmet_need"].fillna("").str.len() > 12]
    raise KeyError(key)


def _sample_block(pool: pd.DataFrame, n: int = 18) -> tuple[str, set]:
    pool = pool[pool["quote_grounded"].astype(bool)] if "quote_grounded" in pool else pool
    s = pool.sample(min(n, len(pool)), random_state=7) if len(pool) else pool
    lines = [f"[{r['id']}] ({r['source']}, {r['event']}, {r['primary_theme']}) quote: \"{r['evidence_quote']}\" "
             f"| need: {r['unmet_need']}" for _, r in s.iterrows()]
    return "\n".join(lines), set(s["id"])


def run() -> dict:
    df = relevant(pd.read_parquet(ITEMS_FILE))
    out, total, bad = {}, 0, 0
    for key, q in QUESTIONS.items():
        stats, pool = _stats(key, df)
        sample, ids = _sample_block(pool)
        try:
            res = llm.call_tool(
                "You are a senior product researcher. Write findings ONLY from the statistics and quotes given. "
                "Every headline must contain a number from the stats. Cite 2-4 IDs from the sample. Do not give "
                "investment advice, do not recommend funds, do not speculate beyond the data. Note small samples.",
                f"Research question: {q}\nTotal relevant items: {len(df)}\n\nSTATS:\n{stats}\n\nSAMPLE:\n{sample}",
                TOOL, model=synth_model(), max_tokens=1200)
        except Exception as e:  # noqa: BLE001
            print(f"  ! {key}: {e}")
            continue
        findings = []
        for f in res.get("findings", []):
            cited = [i for i in f.get("evidence_ids", [])]
            ok = [i for i in cited if i in ids]
            total += len(cited)
            bad += len(cited) - len(ok)
            findings.append({**f, "evidence_ids": ok, "dropped_citations": len(cited) - len(ok)})
        out[key] = {"question": q, "stats": stats, "findings": findings}
        print(f"  {key}: {len(findings)} findings")
    out["_meta"] = {"citations_total": total, "citations_invalid": bad, "model": synth_model()}
    INSIGHTS_FILE.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  citations verified: {total - bad}/{total}")
    return out
