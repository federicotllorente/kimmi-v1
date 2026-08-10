"""General web search via a local SearXNG instance."""
from __future__ import annotations

import hashlib
import json
import os
import time
from typing import Any, Dict, List, Optional
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from tools.common import ensure_cache_dir, truncate
from tools.connectivity import error_json, require_internet

SEARCH_TIMEOUT_SEC = 12
CACHE_TTL_SEC = 60 * 60
MAX_SNIPPET_CHARS = 280
USER_AGENT = "KimmiSearchBot/1.0 (+local-assistant)"


def searxng_url() -> str:
    raw = (os.environ.get("KIMMI_SEARXNG_URL") or "").strip()
    return raw.rstrip("/") if raw else "http://127.0.0.1:8080"


def web_search(
    query: str,
    max_results: int = 5,
    language: str = "",
    allowed_root: str = ".",
) -> str:
    """Search the web through SearXNG. Returns JSON string for the LLM."""
    q = (query or "").strip()
    if not q:
        return error_json("invalid_args", "query is required")

    offline = require_internet()
    if offline is not None:
        return json.dumps(offline, ensure_ascii=False)

    try:
        n = int(max_results)
    except (TypeError, ValueError):
        n = 5
    n = max(1, min(10, n))

    cache_key = hashlib.sha256(
        f"{q}|{n}|{language}|{searxng_url()}".encode("utf-8")
    ).hexdigest()[:32]
    cached = _read_cache(allowed_root, cache_key)
    if cached is not None:
        return cached

    params: Dict[str, str] = {"q": q, "format": "json"}
    if language:
        params["language"] = language

    url = f"{searxng_url()}/search?{urlencode(params)}"
    req = Request(url, headers={"User-Agent": USER_AGENT, "Accept": "application/json"})

    try:
        with urlopen(req, timeout=SEARCH_TIMEOUT_SEC) as resp:
            raw = resp.read()
    except HTTPError as exc:
        return error_json(
            "searxng_unavailable",
            f"SearXNG HTTP {exc.code}",
        )
    except URLError as exc:
        return error_json(
            "searxng_unavailable",
            f"SearXNG unreachable: {exc.reason}",
        )
    except TimeoutError:
        return error_json("timeout", "SearXNG request timed out")
    except Exception as exc:  # noqa: BLE001
        return error_json("searxng_unavailable", f"SearXNG error: {exc}")

    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except json.JSONDecodeError:
        return error_json("searxng_unavailable", "SearXNG returned invalid JSON")

    results_in = data.get("results") or []
    results: List[Dict[str, str]] = []
    for item in results_in[:n]:
        title = truncate(str(item.get("title") or ""), 120)
        link = str(item.get("url") or item.get("href") or "")
        snippet = truncate(
            str(item.get("content") or item.get("snippet") or ""),
            MAX_SNIPPET_CHARS,
        )
        if not title and not link:
            continue
        results.append({"title": title, "url": link, "snippet": snippet})

    payload = {
        "query": q,
        "engine": "searxng",
        "count": len(results),
        "results": results,
    }
    out = json.dumps(payload, ensure_ascii=False)
    _write_cache(allowed_root, cache_key, out)
    return out


def _cache_path(allowed_root: str, key: str) -> str:
    folder = ensure_cache_dir(allowed_root, "search")
    return os.path.join(folder, f"{key}.json")


def _read_cache(allowed_root: str, key: str) -> Optional[str]:
    path = _cache_path(allowed_root, key)
    try:
        with open(path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
        if time.time() - float(blob.get("ts", 0)) > CACHE_TTL_SEC:
            return None
        return str(blob.get("body") or "")
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return None


def _write_cache(allowed_root: str, key: str, body: str) -> None:
    path = _cache_path(allowed_root, key)
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"ts": time.time(), "body": body}, fh)
    except OSError:
        pass
