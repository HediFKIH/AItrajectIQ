"""SDK → backend ingestion for conversational logs.

Design goals:
- SDK captures request/response and sends raw data.
- Backend computes prompt-injection blocking + anonymization.
- Backend computes metrics and saves them into Log.metrics.

This route intentionally replaces the legacy proxy /v1/chat/completions behavior.

Migration note:
- The legacy conversation-metrics computation (app/proxy/metrics.py) has been removed from the route.
- Metrics computation is now performed by app/evaluation/log_runner.py.
"""

import asyncio
import logging
from datetime import datetime
from typing import Any, Optional, List

from fastapi import APIRouter, Request
from pydantic import BaseModel

from models.log import Log
from security.api_keys import INGEST, authenticate
from utils.time import now_utc_plus_one

from evaluation.log_runner import evaluate_log_metrics

logger = logging.getLogger("aitrajectiq.ingest")

router = APIRouter(prefix="/ingest", tags=["ingest-logs"])

_llm_guard: Optional[dict] = None


def _guard() -> Optional[dict]:
    """llm_guard scanners, or None when the package isn't installed (it doesn't support
    every Python version); logs are then stored without injection/PII scanning."""
    global _llm_guard
    if _llm_guard is None:
        try:
            from llm_guard.input_scanners import Anonymize
            from llm_guard.input_scanners.anonymize_helpers import BERT_LARGE_NER_CONF
            from llm_guard.input_scanners.prompt_injection import PromptInjection, MatchType
            from llm_guard.vault import Vault
            _llm_guard = {
                "Anonymize": Anonymize,
                "NER_CONF": BERT_LARGE_NER_CONF,
                "PromptInjection": PromptInjection,
                "MatchType": MatchType,
                "Vault": Vault,
            }
        except ImportError:
            logger.warning("llm_guard is not installed: /ingest/log skips prompt-injection and PII scanning")
            _llm_guard = {}
    return _llm_guard or None


# ---------------------------------------------------------------------------
# Payload schema from SDK
# ---------------------------------------------------------------------------


class LogIngestRequest(BaseModel):
    project_id: Optional[str] = None   # optional: the API key identifies the project

    # User-facing prompt/logging
    prompt: Optional[str] = None
    history: Optional[list] = None  # list[{role, content}]
    retrieved_contexts: Optional[List[str]] = None

    # Provider/gen params
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    model: Optional[str] = None

    # Captured timing/token usage
    latency_ms: Optional[int] = None
    n_tokens: Optional[int] = None

    # Assistant output
    output: Optional[str] = None


class LogIngestResponse(BaseModel):
    log_id: str


def _extract_prompt_from_history(history: list | None) -> str:
    if not history:
        return ""
    for m in reversed(history):
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content", "")
            return str(content)
    return ""


@router.post("/log", response_model=LogIngestResponse)
async def ingest_log(body: LogIngestRequest, request: Request):
    auth = await authenticate(request, INGEST, project_id=body.project_id, allow_legacy=True)
    project = auth.project

    history = body.history or []
    prompt = body.prompt or _extract_prompt_from_history(history)

    blocked = False
    blocked_reason = None
    contains_pii = False
    anonymized = False
    guard = _guard()

    # PI scanning + anonymization
    if guard and prompt:
        pi_scanner = guard["PromptInjection"](threshold=0.5, match_type=guard["MatchType"].FULL)
        _, is_valid_pi, risk_score_pi = pi_scanner.scan(prompt)
        if not is_valid_pi:
            blocked = True
            blocked_reason = f"Prompt Injection detected with risk score {risk_score_pi}"

    if guard and history:
        anon_scanner = guard["Anonymize"](
            guard["Vault"](),
            hidden_names=None,
            recognizer_conf=guard["NER_CONF"],
            language="en",
        )
        for i, m in enumerate(history):
            if not isinstance(m, dict):
                continue
            if m.get("role") == "user" and isinstance(m.get("content"), str):
                sanitized_prompt, is_valid, _risk = anon_scanner.scan(m["content"])
                history[i] = {**m, "content": sanitized_prompt}
                contains_pii = contains_pii or (not is_valid)
                anonymized = anonymized or (not is_valid)

    # Create Log
    log = Log(
        project_id=project.id,
        prompt=prompt,
        history=history,
        retrieved_contexts=body.retrieved_contexts,
        Output=body.output,
        Model=body.model,
        timestamp=body.__dict__.get("timestamp") or now_utc_plus_one(),
        temperature=body.temperature,
        latency=body.latency_ms,
        n_tokens=body.n_tokens,
        blocked=blocked,
        blocked_reason=blocked_reason,
        contains_pii=contains_pii,
        anonymized=anonymized,
    )

    await log.insert()

    # Async metrics (never fail ingestion due to metrics)
    if project.enabled_metrics:
        asyncio.create_task(evaluate_log_metrics(log_id=log.id, project=project))

    return LogIngestResponse(log_id=str(log.id))

