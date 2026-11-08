"""Structured extraction: one LLM call per item, forced into the schema via tool use.

Quality controls built in:
- 'unclear' is always allowed, and the prompt tells the model to prefer it over guessing
- evidence_quote must be copied verbatim; we verify it is a substring of the input (grounding)
- results are cached per (item, model, prompt version, taxonomy) so reruns are cheap and resumable
"""
from __future__ import annotations

import hashlib
import json
import re
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import pandas as pd

from . import llm
from .schema import EMPTY_EXTRACTION, tool_schema
from .settings import CACHE, CLEAN_FILE, ITEMS_FILE, env, extract_model, taxonomy_cfg, taxonomy_hash

PROMPT_VERSION = "v5"  # v2 of the coding rules (stricter relevance and intent)


def system_prompt() -> str:
    tax = taxonomy_cfg()
    themes = "\n".join(f"- {t['id']}: {t['definition']}" for t in tax["themes"])
    return f"""You are a careful qualitative researcher coding public posts, app reviews and comments
about SIPs (Systematic Investment Plans) in mutual funds in India. Your output feeds a product team
studying why first-time investors stop or keep their SIPs in the first 12 months.

Code ONE post by calling the tool. Rules:
1. Prefer "unclear" / empty lists over guessing. Only code what the text supports.
2. is_relevant = true only if the text describes the writer's OWN (or a close person's) SIP or mutual-fund
   experience, decision, question or problem. Mark is_relevant = false for:
   - generic app praise or complaints where mutual funds/SIPs are only listed as a feature
     ("great app for stocks and mutual funds"),
   - a plain list of holdings or amounts with no experience, decision or problem,
   - a passing SIP mention in a post that is really about something else (stock screener, watchlist, UPI),
   - other assets: stocks, US stocks/ETFs, stock SIPs, F&O, IPOs,
   - general market opinions, news, newsletters, off-topic or unintelligible text.
   When is_relevant = false, set event = "none", themes = [], primary_theme = "none", journey_stage = "unclear",
   first_time_investor = "unclear".
3. asset_type: a "stock SIP" (SIP in shares/ETFs, e.g. INDstocks SIP) is stock_sip, not mutual_fund_sip.
4. themes (max 3) from this taxonomy; primary_theme = the ROOT cause, not the trigger:
{themes}
   If both a personal cause (job loss, expense, failed debit, confusion) and a market fall are
   mentioned, the personal cause is primary and market_fall_trigger is secondary.
   Use primary_theme "none" if no theme applies. positive_habit ONLY when the writer describes continuing their SIP
   through a difficulty (a loss, a dip, a cash crunch) or a deliberate habit; never for app praise.
5. event = what actually happened to the SIP in this text. Use "considering_stop" ONLY when the writer says they
   want to stop, pause, exit or redeem (e.g. "should I stop my SIP?"). Questions about how SIP features work
   (lowering the amount, cancel vs redeem, delaying autopay) are event = "none". Use "continuing" only when the
   writer describes keeping their SIP going; praise of an app is not "continuing".
6. journey_stage = months since the person's FIRST SIP when the described event happened, only if the text states
   or clearly implies it. first_time_investor = "yes"/"no" only if the text says so; otherwise "unclear".
7. Read sarcasm correctly ("great app, SIP failed 3 times" is negative). Hinglish/Hindi is common;
   understand it, but write unmet_need and workaround in English.
8. evidence_quote: copy up to 25 words EXACTLY as written in the text (original language). No paraphrase.
9. severity: 1 mild annoyance ... 5 stopped investing, lost money or gave up on the platform.
10. Never add investment opinions. You are only coding what the person said."""


def user_prompt(row: dict) -> str:
    meta = f"source: {row.get('source')} | app/platform hint: {row.get('platform')} | date: {row.get('date')}"
    ctx = f"\nthread/video context: {row['context']}" if row.get("context") else ""
    rating = f"\nstar rating: {row['rating']}" if row.get("rating") not in (None, "") and not pd.isna(row.get("rating")) else ""
    return f"{meta}{ctx}{rating}\n\n<post>\n{row['full_text'][:4000]}\n</post>"


def _norm(s: str) -> str:
    return re.sub(r"[^\w\u0900-\u097F]+", " ", (s or "").lower()).strip()


def grounded(quote: str, text: str) -> bool:
    q = _norm(quote)
    return bool(q) and q in _norm(text)


def _cache_path(model: str):
    return CACHE / f"extract_{re.sub(r'[^a-z0-9]+', '-', model.lower())}_{PROMPT_VERSION}_{taxonomy_hash()}.jsonl"


_lock = threading.Lock()


def _load_cache(path) -> dict:
    out = {}
    if path.exists():
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                out[rec["_key"]] = rec
    return out


def _key(row: dict) -> str:
    return hashlib.sha1((row["id"] + row["full_text"]).encode("utf-8")).hexdigest()


def _finish(rec: dict, row: dict) -> dict:
    rec = {**EMPTY_EXTRACTION, **{k: v for k, v in rec.items() if k != "id"}}
    rec["themes"] = list(dict.fromkeys(rec.get("themes") or []))[:3]
    if rec["primary_theme"] not in rec["themes"] and rec["primary_theme"] != "none":
        rec["themes"] = [rec["primary_theme"]] + rec["themes"][:2]
    rec["quote_grounded"] = grounded(rec.get("evidence_quote", ""), row["full_text"])
    return rec


def extract_one(row: dict, model: str | None = None) -> dict:
    out = llm.call_tool(system_prompt(), user_prompt(row), tool_schema(), model=model or extract_model())
    return _finish(out, row)


def _batch_schema() -> dict:
    item = json.loads(json.dumps(tool_schema()["input_schema"]))
    item["properties"] = {"id": {"type": "string", "description": "copy the post id exactly"}, **item["properties"]}
    item["required"] = ["id"] + item["required"]
    return {"type": "object", "properties": {"items": {"type": "array", "items": item}}, "required": ["items"]}


def extract_batch(rows: list[dict], model: str | None = None) -> dict[str, dict]:
    """Code several posts in one request (free tiers limit requests, not posts). Returns {id: record}."""
    model = model or extract_model()
    if len(rows) == 1:
        return {rows[0]["id"]: extract_one(rows[0], model)}
    user = (f"Code each of these {len(rows)} posts independently. Return {{\"items\": [...]}} with exactly one "
            "object per post, each with its id copied exactly.\n\n" +
            "\n\n".join(user_prompt(r).replace("<post>", f'<post id="{r["id"]}">') for r in rows))
    out = llm.call_json(system_prompt(), user, _batch_schema(), model=model, max_tokens=900 * len(rows))
    by_id = {str(x.get("id", "")).strip(): x for x in out.get("items", [])}
    res = {r["id"]: _finish(by_id[r["id"]], r) for r in rows if r["id"] in by_id}
    for r in rows:  # anything the model skipped gets its own call
        if r["id"] not in res:
            res[r["id"]] = extract_one(r, model)
    return res


def extract_frame(df: pd.DataFrame, model: str | None = None, workers: int | None = None,
                  use_cache: bool = True, progress=True) -> pd.DataFrame:
    model = model or extract_model()
    workers = workers or int(env("EXTRACT_WORKERS", "2"))
    bsize = max(1, int(env("EXTRACT_BATCH", "8")))
    cpath = _cache_path(model)
    cache = _load_cache(cpath) if use_cache else {}
    rows = df.to_dict("records")
    todo = [r for r in rows if _key(r) not in cache]
    batches = [todo[i:i + bsize] for i in range(0, len(todo), bsize)]
    print(f"  provider={llm.provider()} model={model}  items={len(rows):,}  cached={len(rows) - len(todo):,}  "
          f"to code={len(todo):,} in {len(batches):,} requests")
    failures, done, stopped = 0, 0, None
    if batches:
        with ThreadPoolExecutor(max_workers=workers) as ex, cpath.open("a", encoding="utf-8") as fh:
            futs = {ex.submit(extract_batch, b, model): b for b in batches}
            for fut in as_completed(futs):
                b = futs[fut]
                try:
                    recs = fut.result()
                except (llm.LLMUnavailable, llm.QuotaExhausted) as e:
                    stopped = e
                    for f in futs:
                        f.cancel()
                    break
                except Exception as e:  # noqa: BLE001
                    failures += len(b)
                    print(f"    ! batch of {len(b)} failed: {e}")
                    continue
                with _lock:
                    for r in b:
                        rec = recs[r["id"]]
                        rec["_key"] = _key(r)
                        cache[rec["_key"]] = rec
                        if use_cache:
                            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
                    fh.flush()
                done += len(b)
                if progress:
                    print(f"    {done:,}/{len(todo):,} coded")
    if stopped:
        print(f"\n  STOPPED: {stopped}\n  Progress is saved. Run the same command again later to continue.\n")
        if isinstance(stopped, llm.LLMUnavailable):
            raise stopped
    recs = []
    for r in rows:
        rec = cache.get(_key(r))
        if rec is not None:
            recs.append({**r, **{("x_platform" if k == "platform" else k): v for k, v in rec.items() if k != "_key"}})
    if failures:
        print(f"  {failures} items failed; rerun extract to retry them")
    return pd.DataFrame(recs)


def run(limit: int | None = None) -> pd.DataFrame:
    df = pd.read_parquet(CLEAN_FILE)
    if limit:
        df = df.sample(min(limit, len(df)), random_state=7)
    out = extract_frame(df)
    if out.empty:
        print("  nothing coded yet; existing items.parquet left untouched")
        return out
    out.to_parquet(ITEMS_FILE, index=False)
    rel = out["is_relevant"].sum() if len(out) else 0
    print(f"  saved {len(out):,} coded items ({rel:,} relevant) -> {ITEMS_FILE.name}")
    if len(out):
        print(f"  evidence quotes grounded: {out['quote_grounded'].mean():.1%}")
    return out
