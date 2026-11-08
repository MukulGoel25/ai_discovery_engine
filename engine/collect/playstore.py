"""Google Play reviews via the google-play-scraper package (no API key needed)."""
from __future__ import annotations

from .common import make_item


def collect(cfg: dict) -> list[dict]:
    from google_play_scraper import Sort, reviews

    pcfg = cfg.get("playstore", {})
    items: list[dict] = []
    for app in cfg.get("apps", []):
        if not app.get("play_id"):
            continue
        for lang in pcfg.get("langs", ["en"]):
            target = int(pcfg.get("reviews_per_app_per_lang", 2000))
            got, token = 0, None
            print(f"  Play Store: {app['name']} [{lang}]")
            while got < target:
                try:
                    batch, token = reviews(app["play_id"], lang=lang, country=pcfg.get("country", "in"),
                                           sort=Sort.NEWEST, count=min(200, target - got),
                                           continuation_token=token)
                except Exception as e:  # noqa: BLE001
                    print(f"    ! {e}")
                    break
                if not batch:
                    break
                for r in batch:
                    items.append(make_item(
                        "playstore", r.get("content") or "", platform=app["name"], date=r.get("at"),
                        url=f"https://play.google.com/store/apps/details?id={app['play_id']}",
                        rating=r.get("score"), likes=r.get("thumbsUpCount"), native_id=r.get("reviewId", "")))
                got += len(batch)
                if token is None:
                    break
            print(f"    {got:,} reviews")
    return items
