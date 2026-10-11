"""
AI rewrite of a recommended prompt (services.recommendations.recommended_prompts).

The deterministic recommendation appends the instructions of the fired rules; the rewrite
asks the evaluation LLM for a clean version that fixes the problems seen in the traffic (the
judge's reasons included) while keeping what the original prompt requires. Runs on demand
only, and results are cached: the same prompt and findings are not paid for twice.
"""

import hashlib
import json
import time
from typing import Optional

from evaluation import runner
from services.recommendations import instructions_fix

CACHE_TTL_S = 24 * 3600
CACHE_MAX = 200
MAX_TOKENS = 3_000
MAX_RULES = 12

_cache: dict[str, tuple[float, dict]] = {}


class RewriteUnavailable(Exception):
    """No evaluation LLM configured, or it failed or answered something unusable."""


_SYSTEM_PROMPT_TASK = (
    "You are a senior prompt engineer. Rewrite the system prompt below so that it fixes the problems "
    "observed in production. Keep every requirement of the original (role, output format, constraints, "
    "tools, tone), remove redundant or contradictory instructions, and keep it at most 20% longer than the "
    "original. Put stable instructions first and variable parts last, so the provider can cache the prefix. "
    "Write in the language of the original prompt. "
    'Return JSON: {"prompt": "the full rewritten prompt", "changes": ["one short sentence per change"]}'
)

_RULES_TASK = (
    "You write the project rules read by the coding agent {tool} before every request. From the problems "
    "observed in its runs (with the reviewer's reasons), write 4 to {max_rules} short imperative rules that "
    "would have prevented them. Be concrete and specific to these failures; no generic advice. Write in the "
    "language of the problems (English when unsure). "
    'Return JSON: {{"rules": ["rule", ...], "changes": ["one short sentence per problem addressed"]}}'
)

_TOOLS = {"cursor": "Cursor", "claude-code": "Claude Code"}


def _key(item: dict, findings: list[str]) -> str:
    raw = json.dumps([item.get("target"), item.get("current"), item.get("content"), findings], ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _cached(key: str) -> Optional[dict]:
    hit = _cache.get(key)
    if hit and time.time() - hit[0] < CACHE_TTL_S:
        return hit[1]
    _cache.pop(key, None)
    return None


def _store(key: str, value: dict) -> None:
    if len(_cache) >= CACHE_MAX:
        _cache.pop(min(_cache, key=lambda k: _cache[k][0]))
    _cache[key] = (time.time(), value)


def _strings(value) -> list[str]:
    return [str(v).strip() for v in value if str(v).strip()] if isinstance(value, list) else []


async def _ask(system: str, user: str) -> dict:
    if runner.judge_tag() is None:
        raise RewriteUnavailable(
            "Rewrite unavailable: no evaluation model is configured (EVAL_LLM_API_KEY)."
        )
    try:
        return await runner.complete_json(system, user, MAX_TOKENS)
    except Exception as exc:
        raise RewriteUnavailable(f"The evaluation model could not rewrite the prompt ({type(exc).__name__}).") from exc


async def rewrite(item: dict, findings: list[str]) -> dict:
    """The recommended prompt `item` rewritten by the evaluation LLM, same shape with
    source="ai"."""
    key = _key(item, findings)
    cached = _cached(key)
    if cached is not None:
        return cached

    problems = "\n".join(f"- {f}" for f in findings) or "- (no problem measured: make the prompt clearer and shorter)"
    target = item.get("target")
    if target in _TOOLS:
        result = await _ask(
            _RULES_TASK.format(tool=_TOOLS[target], max_rules=MAX_RULES),
            f"Problems observed:\n{problems}\n\nCurrent proposal (to improve):\n{item.get('content', '')}",
        )
        rules = _strings(result.get("rules"))[:MAX_RULES]
        if not rules:
            raise RewriteUnavailable("The evaluation model returned no rules.")
        content = instructions_fix(rules, target)["content"]
    else:
        original = item.get("current") or ""
        result = await _ask(
            _SYSTEM_PROMPT_TASK,
            f"Problems observed:\n{problems}\n\nOriginal system prompt:\n{original or '(none: write one from the instructions below)'}"
            f"\n\nInstructions already proposed:\n{item.get('content', '')}",
        )
        content = str(result.get("prompt") or "").strip()
        if not content:
            raise RewriteUnavailable("The evaluation model returned no prompt.")

    rewritten = {**item, "content": content, "changes": _strings(result.get("changes")) or item.get("changes", []),
                 "source": "ai"}
    _store(key, rewritten)
    return rewritten
