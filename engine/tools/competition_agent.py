"""Background competition analysis: SearXNG + fetch + structured brief."""
from __future__ import annotations

import json
import os
import re
import time
from typing import Any, Dict, List, Tuple
from urllib.parse import urlparse

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

MAX_QUERIES = 5
MAX_URLS = 10
SEARCH_DELAY_SEC = 0.5
FETCH_DELAY_SEC = 0.5

_PRIORITY_RE = re.compile(
    r"competitor|competition|vs\.?|versus|alternative|pricing|price|review|compare",
    re.IGNORECASE,
)


def start_competition_analysis(
    target: str = "",
    provider: str = "",
    allowed_root: str = ".",
) -> str:
    """Enqueue a competition-analysis job. Returns JSON for the LLM."""
    from jobs import get_job_manager

    target_s = (target or "").strip()
    if not target_s:
        return error_json("invalid_args", "target (company or product) is required")

    provider_s = (provider or "").strip().lower()
    if not provider_s:
        return json.dumps(
            {
                "error": "need_provider_choice",
                "message": (
                    "Ask the user whether to analyze competition with local "
                    "(Ollama + SearXNG), Claude, or ChatGPT, then call "
                    "start_competition_analysis again with provider."
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

    title = f"Competition: {target_s[:60]}"
    mgr = get_job_manager()

    def _fn(ctx):
        return run_competition_job(
            ctx,
            target=target_s,
            provider=provider_s,
            allowed_root=allowed_root,
        )

    job_id = mgr.start(
        "competition",
        title,
        _fn,
        target=target_s,
        provider=provider_s,
    )
    return json.dumps(
        {
            "job_id": job_id,
            "status": "queued",
            "kind": "competition",
            "title": title,
            "provider": provider_s,
            "target": target_s,
            "message": (
                "Competition analysis started in the background; it may take several "
                "minutes. Tell the user you will notify them when it finishes; "
                "do not invent the final report yet."
            ),
        },
        ensure_ascii=False,
    )


def run_competition_job(
    ctx,
    *,
    target: str,
    provider: str,
    allowed_root: str,
) -> Dict[str, Any]:
    if ctx.stopping():
        raise RuntimeError("cancelled")

    plan = _plan_competition(target)
    queries = plan.get("queries") or []
    company = plan.get("company") or target
    market = plan.get("market") or ""

    if ctx.stopping():
        raise RuntimeError("cancelled")

    urls: List[str] = []
    search_hits: List[Dict[str, str]] = []
    for q in queries:
        if ctx.stopping():
            raise RuntimeError("cancelled")
        raw = web_search(q, max_results=6, allowed_root=allowed_root)
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

    ranked = _rank_urls(urls, search_hits)[:MAX_URLS]
    pages: List[Dict[str, Any]] = []
    fetch_errors: List[Dict[str, str]] = []
    for link in ranked:
        if ctx.stopping():
            raise RuntimeError("cancelled")
        page = fetch_url_text(link)
        if page.get("ok"):
            pages.append(page)
        else:
            fetch_errors.append(
                {"url": link, "error": str(page.get("error") or "fetch_failed")}
            )
        time.sleep(FETCH_DELAY_SEC)

    if not pages and not search_hits:
        raise RuntimeError("No usable sources found (search/fetch empty)")

    if ctx.stopping():
        raise RuntimeError("cancelled")

    summary = _synthesize_competition(
        target=target,
        company=company,
        market=market,
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
                "status": "ok",
            }
        )
    for err in fetch_errors:
        sources.append(
            {
                "title": "",
                "url": err.get("url") or "",
                "status": "error",
                "error": err.get("error") or "",
            }
        )
    if not any(s.get("status") == "ok" for s in sources):
        for h in search_hits[:MAX_URLS]:
            sources.append(
                {
                    "title": h.get("title") or "",
                    "url": h.get("url") or "",
                    "status": "snippet_only",
                }
            )

    sources_path = os.path.join(ctx.job_dir, "sources.json")
    with open(sources_path, "w", encoding="utf-8") as fh:
        json.dump(
            {
                "target": target,
                "company": company,
                "market": market,
                "provider": provider,
                "queries": queries,
                "sources": sources,
                "fetch_errors": fetch_errors,
            },
            fh,
            ensure_ascii=False,
            indent=2,
        )

    ctx.write_summary(summary)
    return {
        "summary": summary,
        "target": target,
        "company": company,
        "market": market,
        "provider": provider,
        "queries": queries,
        "sources": sources,
        "pages_fetched": len(pages),
        "fetch_errors": fetch_errors,
    }


def _plan_competition(target: str) -> Dict[str, Any]:
    prompt = (
        "You plan a competitive-analysis web search. Reply ONLY with JSON object:\n"
        '{"company": string, "market": string, "queries": string[4-5]}\n'
        "Queries should cover competitors, comparisons, pricing, and reviews. "
        "No markdown.\n"
        f"Target: {target}"
    )
    text = ollama_chat(fast_model(), prompt, num_predict=250)
    plan = _parse_plan_json(text)
    if plan:
        queries = [str(q).strip() for q in (plan.get("queries") or []) if str(q).strip()]
        if queries:
            return {
                "company": str(plan.get("company") or target).strip() or target,
                "market": str(plan.get("market") or "").strip(),
                "queries": queries[:MAX_QUERIES],
            }

    return {
        "company": target,
        "market": "",
        "queries": [
            f"{target} competitors",
            f"{target} vs alternatives",
            f"{target} pricing",
            f"{target} reviews comparison",
            f"best alternatives to {target}",
        ][:MAX_QUERIES],
    }


def _parse_plan_json(raw: str) -> Dict[str, Any]:
    text = (raw or "").strip()
    if not text:
        return {}
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)
    else:
        brace = re.search(r"\{.*\}", text, re.DOTALL)
        if brace:
            text = brace.group(0)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        return {}
    return data if isinstance(data, dict) else {}


def _rank_urls(
    urls: List[str],
    search_hits: List[Dict[str, str]],
) -> List[str]:
    """Prefer diverse domains and competitor/pricing/review signals."""
    hit_by_url = {h["url"]: h for h in search_hits if h.get("url")}
    scored: List[Tuple[float, str]] = []
    seen_hosts: set[str] = set()

    for url in urls:
        host = (urlparse(url).hostname or "").lower()
        hit = hit_by_url.get(url) or {}
        blob = f"{hit.get('title', '')} {hit.get('snippet', '')} {url}"
        score = 1.0
        if _PRIORITY_RE.search(blob):
            score += 3.0
        if host and host not in seen_hosts:
            score += 1.5
        scored.append((score, url))
        if host:
            seen_hosts.add(host)

    scored.sort(key=lambda x: x[0], reverse=True)

    # Re-order for domain diversity while keeping score preference.
    final: List[str] = []
    used_hosts: set[str] = set()
    deferred: List[str] = []
    for _, url in scored:
        host = (urlparse(url).hostname or "").lower()
        if host and host in used_hosts:
            deferred.append(url)
            continue
        final.append(url)
        if host:
            used_hosts.add(host)
    for url in deferred:
        if url not in final:
            final.append(url)
    return final


def _synthesize_competition(
    *,
    target: str,
    company: str,
    market: str,
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
        for h in search_hits[:10]:
            corpus_parts.append(
                f"### {h.get('title')}\nURL: {h.get('url')}\n{h.get('snippet')}\n"
            )

    corpus = "\n".join(corpus_parts)[:16000]
    market_line = f" Market focus: {market}." if market else ""
    prompt = (
        "Write a spoken-friendly competitive analysis brief. "
        "Use short paragraphs (no markdown bullets or asterisks). "
        "Cover: main competitors/players, value propositions, pricing only if "
        "explicitly present in the sources, cautious strengths/weaknesses, and "
        "clearly label speculation vs evidence. "
        "Do not invent prices, market shares, or URLs. "
        "End with a 'Sources:' section listing titles and URLs used.\n\n"
        f"Target: {target} (normalized: {company}).{market_line}\n\n"
        f"Source material:\n{corpus}"
    )

    out = (synthesize_with_provider(provider, prompt, num_predict=1100) or "").strip()
    if not out:
        bits = [f"Competition brief for {target}."]
        for h in search_hits[:6]:
            bits.append(f"{h.get('title')}: {h.get('snippet')}")
        bits.append("Sources:")
        for h in search_hits[:6]:
            bits.append(f"{h.get('title')} — {h.get('url')}")
        out = " ".join(bits)
    return out
