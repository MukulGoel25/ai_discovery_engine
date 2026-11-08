"""Apple App Store reviews via the public iTunes RSS feed (no API key needed)."""
from __future__ import annotations

from .common import get_json, make_item


def resolve_id(term: str, country: str) -> tuple[str, str] | None:
    data = get_json("https://itunes.apple.com/search",
                    params={"term": term, "country": country, "entity": "software", "limit": 5})
    if not data or not data.get("results"):
        return None
    top = data["results"][0]
    return str(top["trackId"]), top.get("trackName", term)


def collect(cfg: dict) -> list[dict]:
    acfg = cfg.get("appstore", {})
    country = acfg.get("country", "in")
    items: list[dict] = []
    for app in cfg.get("apps", []):
        app_id = str(app.get("appstore_id") or "")
        if not app_id and app.get("appstore_term"):
            found = resolve_id(app["appstore_term"], country)
            if not found:
                print(f"  App Store: could not resolve '{app['appstore_term']}'")
                continue
            app_id, track = found
            print(f"  App Store: {app['name']} -> {track} (id {app_id})")
        n = 0
        for page in range(1, int(acfg.get("pages", 10)) + 1):
            url = f"https://itunes.apple.com/{country}/rss/customerreviews/page={page}/id={app_id}/sortby=mostrecent/json"
            data = get_json(url)
            entries = (data or {}).get("feed", {}).get("entry", [])
            if isinstance(entries, dict):
                entries = [entries]
            entries = [e for e in entries if "im:rating" in e]
            if not entries:
                break
            for e in entries:
                text = e.get("content", {}).get("label", "")
                title = e.get("title", {}).get("label", "")
                items.append(make_item(
                    "appstore", text, platform=app["name"], date=e.get("updated", {}).get("label", ""),
                    url=f"https://apps.apple.com/{country}/app/id{app_id}", title=title,
                    rating=int(e["im:rating"]["label"]), native_id=e.get("id", {}).get("label", "")))
                n += 1
        print(f"    {n:,} reviews")
    return items
