"""'Ask the data': retrieval over coded items + an answer that must cite them.

Safety and quality gates (visible in the app):
1. Advice-seeking questions are declined before any model call.
2. Retrieval must find enough matching evidence, otherwise we say so.
3. The model must cite IDs; citations not in the retrieved set are removed.
4. If the model flags insufficient evidence, or no valid citation survives, we show a fallback.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

from . import llm
from .settings import synth_model

ADVICE = re.compile(
    r"\bbest\s+(mutual\s+)?(fund|stock|scheme|sip)s?\b"
    r"|\b(which|what)\s+(mutual\s+)?(fund|stock|scheme)s?\s+(should|to|can|do)\b"
    r"|\bshould\s+i\s+(buy|invest|switch|sell|hold|stop|continue|pause|redeem)\b"
    r"|\brecommend\s+(me\s+)?(a\s+)?(mutual\s+)?(fund|stock|scheme)s?\b", re.I)

TOOL = {
    "name": "answer",
    "description": "Answer the research question from the evidence only.",
    "input_schema": {"type": "object", "properties": {
        "answer": {"type": "string", "description": "3-6 sentences. Quantify using the counts given. Plain language."},
        "cited_ids": {"type": "array", "items": {"type": "string"}},
        "confidence": {"type": "string", "enum": ["low", "medium", "high"]},
        "insufficient_evidence": {"type": "boolean"},
        "based_on": {"type": "string", "enum": ["posts", "stats", "both"],
                     "description": "posts = answered from the evidence items; stats = from the dataset statistics only"},
    }, "required": ["answer", "cited_ids", "confidence", "insufficient_evidence", "based_on"]},
}


class Index:
    def __init__(self, df: pd.DataFrame):
        from sklearn.feature_extraction.text import TfidfVectorizer

        self.df = df.reset_index(drop=True)
        corpus = (self.df["full_text"].fillna("") + " " + self.df["unmet_need"].fillna("") + " " +
                  self.df["primary_theme"].fillna("").str.replace("_", " ") + " " +
                  self.df["event"].fillna(""))
        self.vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1, max_features=60000)
        self.X = self.vec.fit_transform(corpus) if len(corpus) else None

    def search(self, q: str, k: int = 14) -> pd.DataFrame:
        if self.X is None or self.X.shape[0] == 0:
            return self.df.head(0)
        sims = (self.X @ self.vec.transform([q]).T).toarray().ravel()
        idx = np.argsort(-sims)[:k]
        res = self.df.iloc[idx].copy()
        res["score"] = sims[idx]
        return res[res["score"] > 0.02]


def ask(index: Index, question: str, k: int = 14, stats: str = "") -> dict:
    if ADVICE.search(question):
        return {"status": "declined", "answer": "This tool studies investor experiences. It doesn't recommend funds, "
                "stocks or what to buy, sell or switch. For investment decisions, a SEBI-registered investment "
                "adviser is the right source. Try asking what investors say about a topic instead.",
                "hits": index.df.head(0), "cited": []}
    hits = index.search(question, k)
    if len(hits) < 3 and not stats:
        return {"status": "no_evidence", "answer": "There isn't enough matching evidence in the dataset to answer "
                "that. Try different words (e.g., 'mandate failed', 'stopped after loss', 'XIRR').",
                "hits": hits, "cited": []}
    ctx = "\n".join(f"[{r['id']}] source={r['source']} event={r['event']} theme={r['primary_theme']} "
                    f"stage={r['journey_stage']} | {str(r['full_text'])[:500]}" for _, r in hits.iterrows())
    res = llm.call_tool(
        "You are FinBot, a research assistant for a product team studying why first-time investors stop their SIPs. "
        "Answer using ONLY the dataset statistics and evidence items provided. Cite item IDs for claims taken from posts. "
        "Quote numbers from the statistics when they answer the question. Keep it to 3-6 sentences, plain language. "
        "If neither source answers the question, set insufficient_evidence=true. Never give investment advice or "
        "recommend funds. No fear or urgency language.",
        f"Question: {question}\n\nDATASET STATISTICS:\n{stats or 'n/a'}\n\nEVIDENCE ({len(hits)} retrieved items):\n{ctx}",
        TOOL, model=synth_model(), max_tokens=900)
    valid = [i for i in res.get("cited_ids", []) if i in set(hits["id"])]
    dropped = len(res.get("cited_ids", [])) - len(valid)
    stats_only = res.get("based_on") == "stats" and bool(stats)
    if res.get("insufficient_evidence") or (not valid and not stats_only):
        return {"status": "fallback", "answer": "The retrieved evidence doesn't clearly answer this, so I won't guess. "
                "Browse the matching items below or rephrase the question.", "hits": hits, "cited": [],
                "model_answer": res.get("answer", ""), "dropped": dropped}
    return {"status": "ok", "answer": res["answer"], "confidence": res.get("confidence", "low"),
            "hits": hits, "cited": valid, "dropped": dropped}
