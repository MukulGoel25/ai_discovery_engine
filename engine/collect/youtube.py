"""YouTube comments via the YouTube Data API v3 (needs YOUTUBE_API_KEY)."""
from __future__ import annotations

from ..settings import env
from .common import get_json, make_item

API = "https://www.googleapis.com/youtube/v3"


def collect(cfg: dict) -> list[dict]:
    key = env("YOUTUBE_API_KEY")
    if not key:
        print("  YouTube: skipped (set YOUTUBE_API_KEY in .env)")
        return []
    ycfg = cfg.get("youtube", {})
    videos: dict[str, str] = {vid: "" for vid in ycfg.get("video_ids", [])}
    for q in ycfg.get("queries", []):
        data = get_json(f"{API}/search", params={
            "part": "snippet", "q": q, "type": "video", "regionCode": "IN",
            "maxResults": int(ycfg.get("videos_per_query", 6)), "key": key})
        for it in (data or {}).get("items", []):
            videos[it["id"]["videoId"]] = it["snippet"]["title"]
    items: list[dict] = []
    for vid, title in videos.items():
        token, n = None, 0
        while n < int(ycfg.get("comments_per_video", 300)):
            params = {"part": "snippet", "videoId": vid, "maxResults": 100, "order": "relevance",
                      "textFormat": "plainText", "key": key}
            if token:
                params["pageToken"] = token
            data = get_json(f"{API}/commentThreads", params=params, pause=0.2)
            if not data:
                break
            for th in data.get("items", []):
                s = th["snippet"]["topLevelComment"]["snippet"]
                items.append(make_item(
                    "youtube", s.get("textOriginal", ""), date=s.get("publishedAt"),
                    url=f"https://www.youtube.com/watch?v={vid}", context=title,
                    likes=s.get("likeCount"), native_id=th.get("id", "")))
                n += 1
            token = data.get("nextPageToken")
            if not token:
                break
        print(f"  YouTube {vid} '{title[:50]}': {n} comments")
    return items
