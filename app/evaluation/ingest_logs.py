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
from datetime import datetime
from typing import Any, Optional, List

from beanie import PydanticObjectId
from fastapi import APIRouter, Header, HTTPException
from pydantic import BaseModel

from models.log import Log
from models.project import Project
from utils.time import now_utc_plus_one

from llm_guard.input_scanners import Anonymize
from llm_guard.input_scanners.anonymize_helpers import BERT_LARGE_NER_CONF
from llm_guard.input_scanners.prompt_injection import PromptInjection, MatchType
from llm_guard.vault import Vault

from evaluation.log_runner import evaluate_log_metrics

router = APIRouter(prefix="/ingest", tags=["ingest-logs"])


# ---------------------------------------------------------------------------
# Payload schema from SDK
# ---------------------------------------------------------------------------


class LogIngestRequest(BaseModel):
    project_id: str

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


async def _auth_project(project_id: str, api_key: str) -> Project:
    try:
        pid = PydanticObjectId(project_id)
    except Exception:
        raise HTTPException(400, "Invalid project_id")
    project = await Project.get(pid)
    if project is None:
        raise HTTPException(404, "Project not found")

    # Legacy auth is commented elsewhere; keep consistent and just validate project existence.
    # If you want strict auth, uncomment checks.
    # if not project.api_key or not project.api_key.active or project.api_key.key != api_key:
    #     raise HTTPException(401, "Unauthorized")

    return project


def _extract_prompt_from_history(history: list | None) -> str:
    if not history:
        return ""
    for m in reversed(history):
        if isinstance(m, dict) and m.get("role") == "user":
            content = m.get("content", "")
            return str(content)
    return ""


@router.post("/log", response_model=LogIngestResponse)
async def ingest_log(
    body: LogIngestRequest,
    x_api_key: str = Header(..., alias="X-API-Key"),
):
    project = await _auth_project(body.project_id, x_api_key)

    history = body.history or []
    prompt = body.prompt or _extract_prompt_from_history(history)

    # PI scanning + anonymization
    vault = Vault()
    pi_scanner = PromptInjection(threshold=0.5, match_type=MatchType.FULL)

    blocked = False
    blocked_reason = None
    contains_pii = False

    if prompt:
        _, is_valid_pi, risk_score_pi = pi_scanner.scan(prompt)
        if not is_valid_pi:
            blocked = True
            blocked_reason = f"Prompt Injection detected with risk score {risk_score_pi}"

    anonymized = False
    if history:
        anon_scanner = Anonymize(
            vault,
            hidden_names=None,
            recognizer_conf=BERT_LARGE_NER_CONF,
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

