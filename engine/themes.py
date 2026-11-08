"""Two complementary ways to find themes.

1. Emergent clusters: TF-IDF + k-means on the 'unmet_need' sentences, each cluster named by the LLM.
   This is bottom-up and can surface needs the fixed taxonomy misses.
2. Taxonomy proposal (open coding): the LLM reads a sample of unmet needs and proposes themes,
   written to config/taxonomy_proposed.yaml for a human to review and merge.
"""
from __future__ import annotations

import json
import random

import pandas as pd
import yaml

from . import llm
from .analyze import relevant
from .settings import CLUSTERS_FILE, CONFIG, ITEMS_FILE, synth_model, taxonomy_cfg

NAME_TOOL = {
    "name": "name_cluster",
    "description": "Name a cluster of user needs.",
    "input_schema": {"type": "object", "properties": {
        "name": {"type": "string", "description": "3-7 words, plain language, from the user's point of view"},
        "description": {"type": "string", "description": "One sentence describing the shared need"},
        "maps_to_theme": {"type": "string", "description": "Closest existing taxonomy theme id, or 'new'"},
    }, "required": ["name", "description", "maps_to_theme"]},
}


def clusters(k: int | None = None) -> dict:
    from sklearn.cluster import KMeans
    from sklearn.feature_extraction.text import TfidfVectorizer

    df = relevant(pd.read_parquet(ITEMS_FILE))
    df = df[df["unmet_need"].fillna("").str.len() > 12].reset_index(drop=True)
    if len(df) < 30:
        print("  too few unmet needs to cluster (need 30+)")
        return {}
    k = k or max(4, min(14, len(df) // 25))
    vec = TfidfVectorizer(ngram_range=(1, 2), min_df=2, stop_words="english", max_features=4000)
    X = vec.fit_transform(df["unmet_need"])
    km = KMeans(n_clusters=k, n_init=10, random_state=7).fit(X)
    df["cluster"] = km.labels_
    terms = vec.get_feature_names_out()
    theme_ids = [t["id"] for t in taxonomy_cfg()["themes"]]
    out = []
    for c in range(k):
        sub = df[df["cluster"] == c]
        top_terms = [terms[i] for i in km.cluster_centers_[c].argsort()[::-1][:8]]
        examples = sub["unmet_need"].sample(min(12, len(sub)), random_state=7).tolist()
        named = {"name": ", ".join(top_terms[:3]), "description": "", "maps_to_theme": "new"}
        if llm.available():
            try:
                named = llm.call_tool(
                    "You name clusters of user needs from an investing-app research study. Be concrete and plain.",
                    f"Existing theme ids: {theme_ids}\nTop terms: {top_terms}\nExample needs:\n- " + "\n- ".join(examples),
                    NAME_TOOL, model=synth_model(), max_tokens=300)
            except Exception as e:  # noqa: BLE001
                print(f"    ! naming cluster {c}: {e}")
        out.append({"cluster": c, **named, "items": int(len(sub)), "share": float(len(sub) / len(df)),
                    "top_terms": top_terms, "examples": examples[:5], "ids": sub["id"].tolist()})
    out.sort(key=lambda r: r["items"], reverse=True)
    result = {"k": k, "n": int(len(df)), "clusters": out}
    CLUSTERS_FILE.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  {k} clusters over {len(df):,} unmet needs -> {CLUSTERS_FILE.name}")
    return result


PROPOSE_TOOL = {
    "name": "propose_taxonomy",
    "description": "Propose a theme taxonomy from coded user needs.",
    "input_schema": {"type": "object", "properties": {"themes": {"type": "array", "items": {
        "type": "object", "properties": {
            "id": {"type": "string"}, "label": {"type": "string"}, "definition": {"type": "string"},
            "evidence_count_in_sample": {"type": "integer"},
            "is_new_vs_existing": {"type": "boolean"}},
        "required": ["id", "label", "definition", "evidence_count_in_sample", "is_new_vs_existing"]}}},
        "required": ["themes"]},
}


def propose_taxonomy(sample: int = 200) -> None:
    df = relevant(pd.read_parquet(ITEMS_FILE))
    needs = df["unmet_need"].dropna()
    needs = needs[needs.str.len() > 12].tolist()
    random.Random(7).shuffle(needs)
    existing = [{"id": t["id"], "definition": t["definition"]} for t in taxonomy_cfg()["themes"]]
    res = llm.call_tool(
        "You are a qualitative researcher doing open coding. Propose 8-16 mutually exclusive themes that explain "
        "why first-time SIP investors struggle to continue. Mark which are new compared with the existing list. "
        "Do not propose themes that would require recommending specific funds.",
        f"Existing themes: {json.dumps(existing)}\n\nUnmet needs (sample of {min(sample, len(needs))}):\n- "
        + "\n- ".join(needs[:sample]),
        PROPOSE_TOOL, model=synth_model(), max_tokens=3000)
    path = CONFIG / "taxonomy_proposed.yaml"
    path.write_text(yaml.safe_dump(res, allow_unicode=True, sort_keys=False), encoding="utf-8")
    print(f"  proposed {len(res.get('themes', []))} themes -> {path.relative_to(CONFIG.parent)} (review, then merge)")
