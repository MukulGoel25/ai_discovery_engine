"""TradingQ&A (Zerodha's Discourse forum) via public Discourse JSON endpoints."""
from __future__ import annotations

from .common import get_json, make_item, strip_html


def collect(cfg: dict) -> list[dict]:
    tcfg = cfg.get("tradingqna", {})
    base = tcfg.get("base_url", "https://tradingqna.com").rstrip("/")
    seen: set[int] = set()
    items: list[dict] = []
    for q in tcfg.get("queries", []):
        data = get_json(f"{base}/search.json", params={"q": q}, pause=1.0)
        topics = (data or {}).get("topics", [])[: int(tcfg.get("topics_per_query", 25))]
        print(f"  TradingQ&A '{q}': {len(topics)} topics")
        for t in topics:
            if t["id"] in seen:
                continue
            seen.add(t["id"])
            topic = get_json(f"{base}/t/{t['id']}.json", pause=1.0)
            if not topic:
                continue
            title = topic.get("title", t.get("title", ""))
            for p in topic.get("post_stream", {}).get("posts", []):
                text = strip_html(p.get("cooked", ""))
                if text:
                    items.append(make_item(
                        "tradingqna", text, platform="Zerodha", date=p.get("created_at"),
                        url=f"{base}/t/{t['id']}/{p.get('post_number', 1)}", title=title if p.get("post_number") == 1 else "",
                        context=title, likes=p.get("reads"), native_id=str(p.get("id"))))
    return items
