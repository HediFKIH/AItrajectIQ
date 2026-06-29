"""
Lightweight web search for fact-checking claims.
Supports: Tavily, Serper (Google), Brave Search.
Configured via FACT_CHECK_PROVIDER + FACT_CHECK_API_KEY in .env.

NOT a RAG system: no vector store, no embeddings, no document ingestion.
Just a live search API call per claim when needed.
"""

import logging
from typing import Optional
from dataclasses import dataclass

import httpx

log = logging.getLogger("llminsight.eval.search")


@dataclass
class SearchConfig:
    provider: str = "none"       # none | tavily | serper | brave
    api_key: str = ""


_search_config: Optional[SearchConfig] = None


def configure_search(cfg: SearchConfig):
    global _search_config
    _search_config = cfg
    if cfg.provider != "none":
        log.info("Fact-check search configured: provider=%s", cfg.provider)


def is_search_enabled() -> bool:
    return _search_config is not None and _search_config.provider != "none" and bool(_search_config.api_key)


async def search_claim(claim: str, max_results: int = 3) -> str:
    """
    Search the web for evidence about a claim.
    Returns concatenated snippets as context for the LLM judge.
    """
    if not is_search_enabled():
        return ""

    try:
        if _search_config.provider == "tavily":
            return await _tavily_search(claim, max_results)
        elif _search_config.provider == "serper":
            return await _serper_search(claim, max_results)
        elif _search_config.provider == "brave":
            return await _brave_search(claim, max_results)
    except Exception:
        log.exception("Search failed for claim: %s", claim[:80])

    return ""


async def _tavily_search(query: str, max_results: int) -> str:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": _search_config.api_key,
                "query": query,
                "max_results": max_results,
                "search_depth": "basic",
                "include_answer": True,
            },
        )
        resp.raise_for_status()
        data = resp.json()

    parts = []
    if data.get("answer"):
        parts.append(f"Summary: {data['answer']}")
    for r in data.get("results", [])[:max_results]:
        parts.append(f"[{r.get('title', '')}] {r.get('content', '')}")
    return "\n\n".join(parts)


async def _serper_search(query: str, max_results: int) -> str:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.post(
            "https://google.serper.dev/search",
            json={"q": query, "num": max_results},
            headers={"X-API-KEY": _search_config.api_key},
        )
        resp.raise_for_status()
        data = resp.json()

    parts = []
    if data.get("answerBox", {}).get("answer"):
        parts.append(f"Answer: {data['answerBox']['answer']}")
    for r in data.get("organic", [])[:max_results]:
        parts.append(f"[{r.get('title', '')}] {r.get('snippet', '')}")
    return "\n\n".join(parts)


async def _brave_search(query: str, max_results: int) -> str:
    async with httpx.AsyncClient(timeout=10) as client:
        resp = await client.get(
            "https://api.search.brave.com/res/v1/web/search",
            params={"q": query, "count": max_results},
            headers={
                "X-Subscription-Token": _search_config.api_key,
                "Accept": "application/json",
            },
        )
        resp.raise_for_status()
        data = resp.json()

    parts = []
    for r in data.get("web", {}).get("results", [])[:max_results]:
        parts.append(f"[{r.get('title', '')}] {r.get('description', '')}")
    return "\n\n".join(parts)
