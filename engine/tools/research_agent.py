"""Background research agent: SearXNG retrieval + local/cloud synthesis."""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List

from tools.connectivity import error_json, require_internet
from tools.llm_cloud import (
    VALID_PROVIDERS,
    check_provider_ready,
    fast_model,
    ollama_chat,
    synthesize_with_provider,
)
from tools.web_fetch import fetch_url_text
from tools.web_search import web_search

MAX_QUERIES = 4
MAX_URLS = 6
SEARCH_DELAY_SEC = 0.4
FETCH_DELAY_SEC = 0.4


def start_research(
    topic: str = "",
    provider: str = "",
    allowed_root: str = ".",
) -> str:
    """Enqueue a research job. Returns JSON for the conversational LLM."""
    from jobs import get_job_manager

    topic_s = (topic or "").strip()
    if not topic_s:
        return error_json("invalid_args", "topic is required")

    provider_s = (provider or "").strip().lower()
    if not provider_s:
        return json.dumps(
            {
                "error": "need_provider_choice",
                "message": (
                    "Ask the user whether to research with local (Ollama + SearXNG), "
                    "Claude, or ChatGPT, then call start_research again with provider."
                ),
                "providers": sorted(VALID_PROVIDERS),
            },
            ensure_ascii=False,
        )

    provider_err = check_provider_ready(provider_s)
    if provider_err is not None:
        return provider_err

    offline = require_internet()
    if offline is not None:
        return json.dumps(offline, ensure_ascii=False)

    title = f"Research: {topic_s[:60]}"
    mgr = get_job_manager()

    def _fn(ctx):
        return run_research_job(
            ctx,
            topic=topic_s,
            provider=provider_s,
            allowed_root=allowed_root,
        )

    job_id = mgr.start(
        "research",
        title,
        _fn,
        topic=topic_s,
        provider=provider_s,
    )
    return json.dumps(
        {
            "job_id": job_id,
            "status": "queued",
            "kind": "research",
            "title": title,
            "provider": provider_s,
            "topic": topic_s,
            "message": (
                "Research started in the background. Tell the user you will notify "
                "them when it finishes; do not invent the final summary yet."
            ),
        },
        ensure_ascii=False,
    )


def run_research_job(
    ctx,
    *,
    topic: str,
    provider: str,
    allowed_root: str,
) -> Dict[str, Any]:
    if ctx.stopping():
        raise RuntimeError("cancelled")

    queries = _expand_queries(topic)
    if ctx.stopping():
        raise RuntimeError("cancelled")

    urls: List[str] = []
    search_hits: List[Dict[str, str]] = []
    for q in queries:
        if ctx.stopping():
            raise RuntimeError("cancelled")
        raw = web_search(q, max_results=5, allowed_root=allowed_root)
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            continue
        if data.get("error"):
            continue
        for item in data.get("results") or []:
            link = str(item.get("url") or "").strip()
            if not link or link in urls:
                continue
            urls.append(link)
            search_hits.append(
                {
                    "title": str(item.get("title") or ""),
                    "url": link,
                    "snippet": str(item.get("snippet") or ""),
                    "query": q,
                }
            )
        time.sleep(SEARCH_DELAY_SEC)

    urls = urls[:MAX_URLS]
    pages: List[Dict[str, Any]] = []
    for link in urls:
        if ctx.stopping():
            raise RuntimeError("cancelled")
        page = fetch_url_text(link)
        if page.get("ok"):
            pages.append(page)
        time.sleep(FETCH_DELAY_SEC)

    if not pages and not search_hits:
        raise RuntimeError("No usable sources found (search/fetch empty)")

    if ctx.stopping():
        raise RuntimeError("cancelled")

    summary = _synthesize(
        topic=topic,
        provider=provider,
        pages=pages,
        search_hits=search_hits,
    )

    sources = []
    for p in pages:
        sources.append(
            {
                "title": p.get("title") or "",
                "url": p.get("final_url") or p.get("url") or "",
            }
        )
    if not sources:
        for h in search_hits[:MAX_URLS]:
            sources.append({"title": h.get("title") or "", "url": h.get("url") or ""})

    sources_path = os.path.join(ctx.job_dir, "sources.json")
    with open(sources_path, "w", encoding="utf-8") as fh:
        json.dump(
            {"topic": topic, "provider": provider, "queries": queries, "sources": sources},
            fh,
            ensure_ascii=False,
            indent=2,
        )

    ctx.write_summary(summary)
    return {
        "summary": summary,
        "topic": topic,
        "provider": provider,
        "queries": queries,
        "sources": sources,
        "pages_fetched": len(pages),
    }


def _expand_queries(topic: str) -> List[str]:
    """Ask the fast local model for 3–4 search queries; fallback to topic."""
    prompt = (
        "You help plan web research. Given a topic, reply with ONLY a JSON array "
        "of 3 or 4 short web search queries (strings). No markdown.\n"
        f"Topic: {topic}"
    )
    text = ollama_chat(fast_model(), prompt, num_predict=200)
    queries = _parse_query_list(text)
    if not queries:
        queries = [
            topic,
            f"{topic} overview",
            f"{topic} latest",
        ]
    return queries[:MAX_QUERIES]


def _parse_query_list(raw: str) -> List[str]:
    text = (raw or "").strip()
    if not text:
        return []
    fence = re.search(r"```(?:json)?\s*(\[.*?\])\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        bracket = re.search(r"\[.*\]", text, re.DOTALL)
        if bracket:
            text = bracket.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return []
    out: List[str] = []
    if isinstance(data, list):
        for item in data:
            s = str(item).strip()
            if s:
                out.append(s)
    return out


def _synthesize(
    *,
    topic: str,
    provider: str,
    pages: List[Dict[str, Any]],
    search_hits: List[Dict[str, str]],
) -> str:
    corpus_parts: List[str] = []
    for p in pages:
        title = p.get("title") or "Untitled"
        url = p.get("final_url") or p.get("url") or ""
        text = p.get("text") or ""
        corpus_parts.append(f"### {title}\nURL: {url}\n{text}\n")
    if not corpus_parts:
        for h in search_hits[:8]:
            corpus_parts.append(
                f"### {h.get('title')}\nURL: {h.get('url')}\n{h.get('snippet')}\n"
            )

    corpus = "\n".join(corpus_parts)[:14000]
    prompt = (
        "Write a concise spoken-friendly research brief on the topic below. "
        "Use short paragraphs (no markdown bullets or asterisks). "
        "End with a 'Sources:' section listing the titles and URLs you used. "
        "If evidence is thin, say so. Do not invent facts or URLs.\n\n"
        f"Topic: {topic}\n\nSource material:\n{corpus}"
    )

    out = (synthesize_with_provider(provider, prompt, num_predict=900) or "").strip()
    if not out:
        bits = [f"Research on {topic}."]
        for h in search_hits[:5]:
            bits.append(f"{h.get('title')}: {h.get('snippet')}")
        bits.append("Sources:")
        for h in search_hits[:5]:
            bits.append(f"{h.get('title')} — {h.get('url')}")
        out = " ".join(bits)
    return out
