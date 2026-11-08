"""Shared helpers for collectors."""
from __future__ import annotations

import hashlib
import json
import re
import time
from datetime import datetime, timezone
from html import unescape
from pathlib import Path

import requests

from ..settings import RAW

UA = "sip-discovery-research/1.0 (product research; contact via GitHub)"
_session = requests.Session()
_session.headers.update({"User-Agent": UA})


def get_json(url: str, params: dict | None = None, headers: dict | None = None,
             tries: int = 4, pause: float = 1.0) -> dict | list | None:
    for i in range(tries):
        try:
            r = _session.get(url, params=params, headers=headers, timeout=30)
            if r.status_code == 429:
                time.sleep(5 * (i + 1))
                continue
            if r.status_code >= 400:
                print(f"    ! {r.status_code} {url}")
                return None
            time.sleep(pause)
            return r.json()
        except Exception as e:  # noqa: BLE001
            print(f"    ! {type(e).__name__} {url}: {e}")
            time.sleep(2 * (i + 1))
    return None


def strip_html(s: str) -> str:
    s = re.sub(r"<(br|p|/p|li)[^>]*>", "\n", s or "")
    s = re.sub(r"<[^>]+>", " ", s)
    return re.sub(r"[ \t]+", " ", unescape(s)).strip()


def iso(dt) -> str:
    if dt is None or dt == "":
        return ""
    if isinstance(dt, (int, float)):
        return datetime.fromtimestamp(dt, tz=timezone.utc).date().isoformat()
    if isinstance(dt, datetime):
        return dt.date().isoformat()
    return str(dt)[:10]


def make_item(source: str, text: str, *, platform: str = "generic", date="", url: str = "",
              title: str = "", context: str = "", rating=None, likes=None, native_id: str = "") -> dict:
    key = f"{source}|{native_id or url}|{text[:200]}"
    return {
        "id": f"{source[:2].upper()}-{hashlib.sha1(key.encode('utf-8')).hexdigest()[:10]}",
        "source": source,
        "platform": platform,
        "date": iso(date),
        "url": url,
        "title": title or "",
        "context": context or "",
        "text": (text or "").strip(),
        "rating": rating,
        "likes": likes,
    }


def save_raw(name: str, items: list[dict]) -> Path:
    path = RAW / f"{name}.jsonl"
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"  saved {len(items):,} items -> {path.relative_to(RAW.parent.parent)}")
    return path
