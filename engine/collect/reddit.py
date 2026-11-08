"""Reddit posts + comments.

Uses the official OAuth API when REDDIT_CLIENT_ID/SECRET are set (recommended, create a
'script' app at https://www.reddit.com/prefs/apps), else the public .json endpoints.
"""
from __future__ import annotations

import requests

from ..settings import env
from .common import get_json, make_item


def _auth_headers() -> tuple[str, dict]:
    cid, secret = env("REDDIT_CLIENT_ID"), env("REDDIT_CLIENT_SECRET")
    ua = env("REDDIT_USER_AGENT", "sip-discovery-research/1.0")
    if cid and secret:
        r = requests.post("https://www.reddit.com/api/v1/access_token",
                          auth=(cid, secret), data={"grant_type": "client_credentials"},
                          headers={"User-Agent": ua}, timeout=30)
        if r.ok and "access_token" in r.json():
            print("  Reddit: using OAuth API")
            return "https://oauth.reddit.com", {"Authorization": f"bearer {r.json()['access_token']}", "User-Agent": ua}
        print(f"  Reddit: OAuth failed ({r.status_code}); falling back to public JSON")
    return "https://www.reddit.com", {"User-Agent": ua}


def _walk_comments(children, post, out, limit):
    for c in children or []:
        if len(out) >= limit or c.get("kind") != "t1":
            continue
        d = c["data"]
        body = d.get("body", "")
        if body and body not in ("[deleted]", "[removed]"):
            out.append(make_item(
                "reddit", body, date=d.get("created_utc"),
                url=f"https://www.reddit.com{d.get('permalink', '')}",
                context=post["title"], likes=d.get("score"), native_id=d.get("id", "")))
        replies = d.get("replies")
        if isinstance(replies, dict):
            _walk_comments(replies.get("data", {}).get("children"), post, out, limit)


def collect(cfg: dict) -> list[dict]:
    rcfg = cfg.get("reddit", {})
    base, headers = _auth_headers()
    suffix = "" if "oauth" in base else ".json"
    seen: set[str] = set()
    items: list[dict] = []
    for sub in rcfg.get("subreddits", []):
        for q in rcfg.get("queries", []):
            data = get_json(f"{base}/r/{sub}/search{suffix}", headers=headers, pause=1.2, params={
                "q": q, "restrict_sr": 1, "sort": "relevance", "t": rcfg.get("time_filter", "year"),
                "limit": int(rcfg.get("posts_per_query", 50)), "raw_json": 1})
            posts = [c["data"] for c in (data or {}).get("data", {}).get("children", [])]
            new = [p for p in posts if p["id"] not in seen]
            print(f"  Reddit r/{sub} '{q}': {len(posts)} posts ({len(new)} new)")
            for p in new:
                seen.add(p["id"])
                post = {"title": p.get("title", "")}
                items.append(make_item(
                    "reddit", p.get("selftext", "") or p.get("title", ""), title=p.get("title", ""),
                    date=p.get("created_utc"), url=f"https://www.reddit.com{p.get('permalink', '')}",
                    context=f"r/{sub}", likes=p.get("score"), native_id=p["id"]))
                if p.get("num_comments", 0) > 0:
                    tree = get_json(f"{base}/comments/{p['id']}{suffix}", headers=headers, pause=1.2,
                                    params={"limit": 200, "depth": 3, "raw_json": 1})
                    if isinstance(tree, list) and len(tree) > 1:
                        out: list[dict] = []
                        _walk_comments(tree[1].get("data", {}).get("children"), post, out,
                                       int(rcfg.get("comments_per_post", 60)))
                        items.extend(out)
    return items
