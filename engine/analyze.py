"""Aggregations that turn coded items into evidence: counts, shares, opportunity scores."""
from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import DROP_EVENTS
from .settings import taxonomy_cfg

STAGE_ORDER = ["before_start", "start", "first_debit", "months_1_3", "months_4_6", "months_7_12", "after_12_months"]
STAGE_LABEL = {"before_start": "Before start", "start": "Setting up", "first_debit": "First debit",
               "months_1_3": "Months 1–3", "months_4_6": "Months 4–6", "months_7_12": "Months 7–12",
               "after_12_months": "After a year", "unclear": "Unclear"}
BRANCH_LABEL = {"A": "A: Activation quality", "B": "B: Payment reliability", "C": "C: Belief & understanding",
                "D": "D: Recovery", "X": "X: Trigger or out of scope", "none": "Positive signal"}


def as_list(v) -> list:
    if v is None:
        return []
    if isinstance(v, (list, tuple)):
        return list(v)
    if isinstance(v, np.ndarray):
        return v.tolist()
    if isinstance(v, str):
        return [v] if v else []
    return []


def relevant(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    out = df[df["is_relevant"].astype(bool)].copy()
    return out


def mf_only(df: pd.DataFrame) -> pd.DataFrame:
    return df[df["asset_type"].isin(["mutual_fund_sip", "unclear"])]


def has_theme(df: pd.DataFrame, theme: str) -> pd.Series:
    return df["themes"].map(lambda v: theme in as_list(v))


def is_drop(df: pd.DataFrame) -> pd.Series:
    return df["event"].isin(DROP_EVENTS)


def theme_table(df: pd.DataFrame) -> pd.DataFrame:
    tax = taxonomy_cfg()["themes"]
    n = max(len(df), 1)
    rows = []
    for t in tax:
        m = has_theme(df, t["id"]) if len(df) else pd.Series(dtype=bool)
        sub = df[m] if len(df) else df
        rows.append({
            "theme": t["id"], "label": t["label"], "branch": t["branch"],
            "addressability": t["addressability"], "rationale": t["rationale"],
            "items": int(m.sum()) if len(df) else 0,
            "primary": int((df["primary_theme"] == t["id"]).sum()) if len(df) else 0,
            "share": float(m.sum() / n) if len(df) else 0.0,
            "severity": float(sub["severity"].mean()) if len(sub) else 0.0,
            "drop_link": float(is_drop(sub).mean()) if len(sub) else 0.0,
        })
    return pd.DataFrame(rows)


def opportunity_table(df: pd.DataFrame, addr_override: dict | None = None) -> pd.DataFrame:
    """Score = Frequency(0-5) x Severity(1-5) x Metric linkage(0-3) x Addressability(0-3), scaled to 0-100."""
    t = theme_table(df)
    t = t[t["branch"] != "none"].copy()
    if addr_override:
        t["addressability"] = t["theme"].map(lambda k: addr_override.get(k)).fillna(t["addressability"])
    max_share = t["share"].max() or 1
    t["F"] = 5 * t["share"] / max_share
    t["S"] = t["severity"]
    t["L"] = 3 * t["drop_link"]
    t["A"] = t["addressability"].astype(float)
    t["score"] = (t["F"] * t["S"] * t["L"] * t["A"]) / (5 * 5 * 3 * 3) * 100
    return t.sort_values("score", ascending=False).reset_index(drop=True)


def journey_leaks(df: pd.DataFrame) -> pd.DataFrame:
    d = df[is_drop(df)]
    counts = d["journey_stage"].value_counts()
    all_counts = df["journey_stage"].value_counts()
    rows = []
    for s in STAGE_ORDER:
        sub = d[d["journey_stage"] == s]
        top = ""
        if len(sub):
            pt = sub["primary_theme"].value_counts()
            pt = pt[pt.index != "none"]
            if (pt.index != "market_fall_trigger").any():  # a market fall is a trigger, not a root cause
                pt = pt[pt.index != "market_fall_trigger"]
            top = pt.index[0] if len(pt) else ""
        rows.append({"stage": s, "label": STAGE_LABEL[s], "drops": int(counts.get(s, 0)),
                     "mentions": int(all_counts.get(s, 0)), "top_theme": top})
    return pd.DataFrame(rows)


def dist(df: pd.DataFrame, col: str, multi: bool = False, drop=("unclear", "none", "")) -> pd.DataFrame:
    if df.empty:
        return pd.DataFrame(columns=[col, "n", "share"])
    s = df[col].map(as_list).explode() if multi else df[col]
    s = s.dropna()
    s = s[~s.isin(drop)]
    vc = s.value_counts()
    return pd.DataFrame({col: vc.index, "n": vc.values, "share": vc.values / max(len(df), 1)})


def theme_by(df: pd.DataFrame, col: str, min_n: int = 5) -> pd.DataFrame:
    """Share of items in each segment value that mention each theme (heatmap input)."""
    tax = [t for t in taxonomy_cfg()["themes"] if t["branch"] != "none"]
    vals = [v for v, n in df[col].value_counts().items() if v not in ("unclear", "none") and n >= min_n]
    out = []
    for v in vals:
        sub = df[df[col] == v]
        for t in tax:
            out.append({col: v, "theme": t["label"], "share": float(has_theme(sub, t["id"]).mean()), "n": len(sub)})
    return pd.DataFrame(out)


def continuers_vs_drop(df: pd.DataFrame) -> pd.DataFrame:
    cont = df[df["event"].isin(["continuing", "restarted"])]
    drop = df[is_drop(df)]
    rows = []
    for t in taxonomy_cfg()["themes"]:
        c = float(has_theme(cont, t["id"]).mean()) if len(cont) else 0.0
        d = float(has_theme(drop, t["id"]).mean()) if len(drop) else 0.0
        rows.append({"theme": t["label"], "continuers": c, "drop_offs": d, "gap": d - c})
    return pd.DataFrame(rows).sort_values("gap", ascending=False), len(cont), len(drop)


def driver_compare(df: pd.DataFrame) -> pd.DataFrame:
    cont = df[df["event"].isin(["continuing", "restarted"])]
    drop = df[is_drop(df)]
    drivers = taxonomy_cfg().get("continuation_drivers", [])

    def share(sub, d):
        return float(sub["continuation_drivers"].map(lambda v: d in as_list(v)).mean()) if len(sub) else 0.0

    return pd.DataFrame([{"driver": d.replace("_", " "), "continuers": share(cont, d), "drop_offs": share(drop, d)}
                         for d in drivers])


def quotes(df: pd.DataFrame, n: int = 6, theme: str | None = None, mask=None) -> pd.DataFrame:
    d = df
    if theme:
        d = d[d["primary_theme"] == theme]
    if mask is not None:
        d = d[mask.reindex(d.index, fill_value=False)]
    d = d[d["quote_grounded"].astype(bool) & (d["evidence_quote"].str.len() > 10)]
    d = d.assign(_likes=pd.to_numeric(d.get("likes"), errors="coerce").fillna(0))
    return d.sort_values(["severity", "_likes"], ascending=False).head(n)


def monthly_trend(df: pd.DataFrame, top_themes: list[str]) -> pd.DataFrame:
    d = df[df["date"].astype(str).str.len() >= 7].copy()
    if d.empty:
        return pd.DataFrame()
    d["month"] = d["date"].astype(str).str[:7]
    rows = []
    for m, sub in d.groupby("month"):
        for t in top_themes:
            rows.append({"month": m, "theme": t, "share": float(has_theme(sub, t).mean()), "n": len(sub)})
    return pd.DataFrame(rows)
