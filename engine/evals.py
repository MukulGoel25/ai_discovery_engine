"""Evaluation of the discovery engine against human labels.

Workflow
1. `python pipeline.py golden-sample`  -> data/golden/golden_to_label.csv (stratified ~20 items)
2. Label the gold_* columns YOURSELF before looking at AI output, save as data/golden/golden_set.csv
3. `python pipeline.py eval`           -> evals/report.json + evals/report.md (shown in the app)
Optional
- `python pipeline.py theme-audit` -> data/golden/theme_audit.csv; fill is_correct (y/n) to get
  per-theme precision on the full dataset.
- `python pipeline.py eval --compare-model claude-sonnet-5-5` -> agreement between two models.
- data/golden/corrections.csv -> the places you rejected or corrected AI output (Evidence Log).
"""
from __future__ import annotations

import json
from datetime import datetime

import pandas as pd

from .extract import PROMPT_VERSION, extract_frame
from .schema import EVENTS, FIRST_TIME, STAGES
from .settings import CLEAN_FILE, EVALS, GOLDEN, ITEMS_FILE, extract_model, theme_ids

FIELDS = {  # gold column -> model field
    "gold_is_relevant": "is_relevant",
    "gold_event": "event",
    "gold_primary_theme": "primary_theme",
    "gold_journey_stage": "journey_stage",
    "gold_first_time": "first_time_investor",
}
REPORT_JSON = EVALS / "report.json"
REPORT_MD = EVALS / "report.md"


def golden_sample(n: int = 20, seed: int = 11) -> None:
    df = pd.read_parquet(CLEAN_FILE)
    per = max(1, n // max(df["source"].nunique(), 1))
    s = df.sample(frac=1, random_state=seed).groupby("source").head(per)
    if len(s) < n:
        s = pd.concat([s, df.drop(s.index).sample(min(n - len(s), len(df) - len(s)), random_state=seed)])
    s = s.head(n)
    out = s[["id", "source", "platform", "lang", "full_text"]].copy()
    for c in FIELDS:
        out[c] = ""
    out["gold_notes"] = ""
    path = GOLDEN / "golden_to_label.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    guide = GOLDEN / "LABEL_GUIDE.md"
    guide.write_text(
        "# How to label the golden set\n\nLabel each row BEFORE looking at any AI output.\n\n"
        "- gold_is_relevant: yes / no (is it about a person's own SIP / mutual-fund experience?)\n"
        f"- gold_event: one of {', '.join(EVENTS)}\n"
        f"- gold_primary_theme: one of {', '.join(theme_ids())}, none\n"
        f"- gold_journey_stage: one of {', '.join(STAGES)}\n"
        f"- gold_first_time: one of {', '.join(FIRST_TIME)}\n"
        "- gold_notes: anything ambiguous (sarcasm, Hinglish, mixed reasons)\n\n"
        "Save the labelled file as data/golden/golden_set.csv\n", encoding="utf-8")
    print(f"  wrote {len(out)} rows -> {path.relative_to(GOLDEN.parent.parent)} (+ LABEL_GUIDE.md)")


def _norm(field: str, v):
    if field == "is_relevant":
        return str(v).strip().lower() in ("yes", "y", "true", "1")
    v = "" if pd.isna(v) else str(v).strip().lower()
    return v or "unclear"


def _kappa(a: list, b: list) -> float | None:
    try:
        from sklearn.metrics import cohen_kappa_score

        if len(set(a) | set(b)) < 2:
            return None
        return float(cohen_kappa_score([str(x) for x in a], [str(x) for x in b]))
    except Exception:  # noqa: BLE001
        return None


def _compare(gold: pd.DataFrame, pred: pd.DataFrame) -> tuple[dict, list]:
    m = gold.merge(pred, on="id", how="inner", suffixes=("", "_p"))
    gold_rel = m["gold_is_relevant"].map(lambda v: _norm("is_relevant", v))
    fields, disagreements = {}, []
    for gcol, pcol in FIELDS.items():
        sub = m[m[gcol].astype(str).str.strip() != ""]
        if pcol != "is_relevant":  # other fields only mean something for posts the human marked relevant
            sub = sub[gold_rel.reindex(sub.index).astype(bool)]
        if not len(sub):
            continue
        g = [_norm(pcol, v) for v in sub[gcol]]
        p = [_norm(pcol, v) for v in sub[pcol]]
        agree = [x == y for x, y in zip(g, p)]
        res = {"n": len(sub), "agreement": sum(agree) / len(sub), "kappa": _kappa(g, p),
               "scored_on": "all items" if pcol == "is_relevant" else "items I marked relevant"}
        if pcol == "is_relevant":
            tp = sum(x and y for x, y in zip(g, p)); fp = sum((not x) and y for x, y in zip(g, p))
            fn = sum(x and (not y) for x, y in zip(g, p))
            res["precision"] = tp / (tp + fp) if tp + fp else None
            res["recall"] = tp / (tp + fn) if tp + fn else None
        if pcol == "primary_theme":  # lenient: gold theme appears anywhere in predicted themes
            lenient = [x == y or x in [str(t) for t in list(ts)] for x, y, ts in zip(g, p, sub["themes"])]
            res["lenient_agreement"] = sum(lenient) / len(sub)
            res["confusion"] = pd.crosstab(pd.Series(g, name="gold"), pd.Series(p, name="ai")).to_dict()
        fields[pcol] = res
        for (_, r), x, y, ok in zip(sub.iterrows(), g, p, agree):
            if not ok:
                disagreements.append({"id": r["id"], "field": pcol, "gold": x, "ai": y,
                                      "text": str(r["full_text"])[:300], "note": r.get("gold_notes", "")})
    return fields, disagreements


def rescore_cache(cache_file: str, label: str) -> dict:
    """Score an older prompt version's cached predictions with the current scoring rules (fair before/after)."""
    from .extract import _key

    gold = pd.read_csv(GOLDEN / "golden_set.csv").fillna("")
    recs = {}
    for line in open(cache_file, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            recs[r["_key"]] = r
    rows = []
    for _, g in gold.iterrows():
        rec = recs.get(_key({"id": g["id"], "full_text": g["full_text"]}))
        if rec:
            rows.append({"id": g["id"], **{k: v for k, v in rec.items() if k != "_key"}})
    pred = pd.DataFrame(rows)
    fields, dis = _compare(gold, pred)
    rep = {"generated": datetime.now().isoformat(timespec="minutes"), "prompt_label": label, "model": "cached",
           "golden_n": int(len(gold)), "matched": len(pred), "fields": fields, "disagreements": dis}
    (EVALS / f"report_{label}.json").write_text(json.dumps(rep, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    for f, r in fields.items():
        print(f"  {label} {f:<22} agreement {r['agreement']:.0%} (n={r['n']})")
    return rep


def run(fresh: bool = False, compare_model: str | None = None) -> dict:
    path = GOLDEN / "golden_set.csv"
    if not path.exists():
        raise SystemExit("No data/golden/golden_set.csv. Run `python pipeline.py golden-sample`, label it, save as golden_set.csv.")
    gold = pd.read_csv(path).fillna("")
    if "full_text" not in gold:
        raise SystemExit("golden_set.csv must keep the full_text column.")
    base = gold[["id", "full_text"]].assign(source=gold.get("source", ""), platform=gold.get("platform", ""),
                                            date="", context="", rating=None)
    pred = extract_frame(base, use_cache=not fresh, progress=False)
    fields, dis = _compare(gold, pred)
    if REPORT_JSON.exists():  # keep earlier prompt versions for the before/after story
        try:
            old = json.loads(REPORT_JSON.read_text(encoding="utf-8"))
            if not old.get("demo") and old.get("prompt_version") != PROMPT_VERSION:
                tag = old.get("prompt_label", "v1")
                (EVALS / f"report_{tag}.json").write_text(json.dumps(old, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001
            pass
    report = {
        "generated": datetime.now().isoformat(timespec="minutes"), "prompt_version": PROMPT_VERSION,
        "prompt_label": {"v4": "v1", "v5": "v2"}.get(PROMPT_VERSION, PROMPT_VERSION),
        "model": extract_model(), "golden_n": int(len(gold)), "fields": fields, "disagreements": dis,
    }
    # grounding across the whole coded dataset
    if ITEMS_FILE.exists():
        items = pd.read_parquet(ITEMS_FILE)
        rel = items[items["is_relevant"].astype(bool)]
        report["grounding"] = {"all": float(items["quote_grounded"].mean()) if len(items) else None,
                               "relevant": float(rel["quote_grounded"].mean()) if len(rel) else None,
                               "n": int(len(items))}
    # theme precision audit
    audit = GOLDEN / "theme_audit.csv"
    if audit.exists():
        a = pd.read_csv(audit).fillna("")
        a = a[a["is_correct"].astype(str).str.strip() != ""]
        if len(a):
            a["ok"] = a["is_correct"].astype(str).str.lower().str.startswith("y")
            report["theme_precision"] = (a.groupby("primary_theme")["ok"].agg(["mean", "count"])
                                         .rename(columns={"mean": "precision", "count": "n"}).reset_index()
                                         .to_dict("records"))
    # model-vs-model agreement
    if compare_model:
        alt = extract_frame(base, model=compare_model, use_cache=True, progress=False)
        mm = pred.merge(alt, on="id", suffixes=("", "_alt"))
        report["model_agreement"] = {"other_model": compare_model, "fields": {
            f: float((mm[f].astype(str) == mm[f + "_alt"].astype(str)).mean()) for f in FIELDS.values()}}
    corr = GOLDEN / "corrections.csv"
    if corr.exists():
        report["corrections"] = pd.read_csv(corr).fillna("").to_dict("records")
    REPORT_JSON.write_text(json.dumps(report, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    _write_md(report)
    for f, r in fields.items():
        k = f"{r['kappa']:.2f}" if r.get("kappa") is not None else "n/a"
        extra = f"  precision {r['precision']:.0%} recall {r['recall']:.0%}" if r.get("precision") is not None else ""
        print(f"  {f:<22} agreement {r['agreement']:.0%}  kappa {k}  (n={r['n']}){extra}")
    print(f"  {len(dis)} disagreements -> {REPORT_MD.relative_to(EVALS.parent)}")
    return report


def theme_audit(per_theme: int = 10) -> None:
    items = pd.read_parquet(ITEMS_FILE)
    items = items[items["is_relevant"].astype(bool) & (items["primary_theme"] != "none")]
    s = items.sample(frac=1, random_state=3).groupby("primary_theme").head(per_theme)
    out = s[["id", "source", "primary_theme", "evidence_quote", "full_text"]].assign(is_correct="", note="")
    path = GOLDEN / "theme_audit.csv"
    out.to_csv(path, index=False, encoding="utf-8-sig")
    print(f"  wrote {len(out)} rows -> {path.relative_to(GOLDEN.parent.parent)}; fill is_correct with y/n, then run eval")


def _write_md(r: dict) -> None:
    lines = [f"# Discovery engine evaluation\n\nGenerated {r['generated']} with `{r['model']}` on {r['golden_n']} golden items.\n",
             "| Field | n | Agreement | Cohen's kappa |", "|---|---|---|---|"]
    for f, v in r["fields"].items():
        k = f"{v['kappa']:.2f}" if v.get("kappa") is not None else "n/a"
        extra = f" (lenient {v['lenient_agreement']:.0%})" if "lenient_agreement" in v else ""
        lines.append(f"| {f} | {v['n']} | {v['agreement']:.0%}{extra} | {k} |")
    if r.get("grounding"):
        lines.append(f"\nEvidence quotes found verbatim in source text (relevant posts): {r['grounding']['relevant']:.0%}.\n")
    lines.append("\n## Disagreements\n")
    for d in r["disagreements"]:
        lines.append(f"- **{d['id']}** {d['field']}: gold `{d['gold']}` vs AI `{d['ai']}` — {d['text'][:160]}")
    REPORT_MD.write_text("\n".join(lines), encoding="utf-8")
