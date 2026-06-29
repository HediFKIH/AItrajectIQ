"""
LLMInsight SDK — Observabilité des applications LLM.

Intégration zéro-code — fonctionne avec N'IMPORTE quelle application LLM :

    import llminsight
    llminsight.init(project_id="...", api_key="...")
    llminsight.patch()

    # C'est tout. Chaque appel LLM est capturé automatiquement.
    # Les traces sont créées automatiquement.
    # Les métriques sont calculées côté backend.
"""

import time
import uuid
import atexit
import functools
import logging
from contextlib import contextmanager
from typing import Optional

from .client import configure, send_span, start_trace, end_trace
from .context import (
    set_trace_id,
    get_trace_id,
    clear_trace_id,
    next_step_index,
    set_project_id,
    get_project_id,
    set_api_key,
    close_auto_trace,
    new_span_id,
    set_agent_context,
    push_span,
    pop_span,
    current_span_id,
    parent_span_id,
)
from .wrappers import patch_all, unpatch_all
from .guardrails import (
    add_guardrail,
    clear_guardrails,
    configure_guardrails,
    load_baseline,
    PromptBlockedError,
)

logger = logging.getLogger("llminsight")

__version__ = "0.1.0"
__all__ = [
    "init", "patch", "unpatch", "trace", "agent_span", "tool_span", "observe",
    "add_guardrail", "clear_guardrails", "configure_guardrails", "load_baseline", "PromptBlockedError",
    "__version__",
]


# ─── init ────────────────────────────────────────────────────────────────────

def init(
    project_id: str,
    api_key: str,
    base_url: str = "http://localhost:8000",
):
    """
    Initialise le SDK. À appeler une seule fois au démarrage de l'application.

    Args:
        project_id: Identifiant du projet LLMInsight.
        api_key: Clé API LLMInsight (PAS la clé du fournisseur LLM).
        base_url: URL du backend LLMInsight.
    """
    configure(base_url=base_url, api_key=api_key, project_id=project_id)
    set_project_id(project_id)
    set_api_key(api_key)
    atexit.register(close_auto_trace)
    logger.info("LLMInsight SDK initialized (project=%s, backend=%s)", project_id, base_url)


# ─── patch / unpatch ─────────────────────────────────────────────────────────

def patch():
    """
    Active le monitoring automatique complet — couvre TOUT :

    Providers : OpenAI, Azure OpenAI, Anthropic, Google Gemini, Mistral,
                Cohere, DeepSeek, Ollama, Groq, Together, OpenRouter, LiteLLM, etc.

    Frameworks : LangChain, LangGraph, CrewAI, AutoGen, LlamaIndex, etc.

    Les traces sont créées automatiquement si l'utilisateur n'en crée pas.
    Après patch(), aucune modification de code n'est nécessaire.
    """
    patch_all()


def unpatch():
    """Désactive le monkey-patching et restaure les fonctions originales."""
    unpatch_all()


# ─── trace ───────────────────────────────────────────────────────────────────

@contextmanager
def trace(name: str, metadata: Optional[dict] = None):
    """
    Context manager pour grouper des spans dans une trace.

    Usage:
        with llminsight.trace("agent-run-42", {"user_id": "u-1"}):
            resp = openai.chat.completions.create(...)
            # le span LLM est automatiquement rattaché à la trace
    """
    trace_id = start_trace(name, metadata)

    if trace_id is None:
        trace_id = f"local-{uuid.uuid4().hex[:12]}"
        logger.warning("LLMInsight: backend unreachable, using local trace_id=%s", trace_id)

    set_trace_id(trace_id)
    try:
        yield trace_id
    except Exception:
        end_trace(trace_id, status="failed")
        clear_trace_id()
        raise
    else:
        end_trace(trace_id, status="completed")
        clear_trace_id()


# ─── agent_span ──────────────────────────────────────────────────────────────

@contextmanager
def agent_span(
    agent_name: str,
    role: str = "worker",
    metadata: Optional[dict] = None,
):
    """
    Context manager pour tracer un agent dans une hiérarchie.

    Séquentiel (Flow AI) :
        with llminsight.trace("pipeline"):
            with llminsight.agent_span("étape-1-analyse", role="worker"):
                resp1 = client.chat.completions.create(...)
            with llminsight.agent_span("étape-2-synthèse", role="worker"):
                resp2 = client.chat.completions.create(...)

    Hiérarchique (manager/workers) :
        with llminsight.trace("delegation"):
            with llminsight.agent_span("manager", role="manager"):
                decision = client.chat.completions.create(...)
                with llminsight.agent_span("worker-recherche", role="worker"):
                    r1 = client.chat.completions.create(...)
                with llminsight.agent_span("worker-rédaction", role="worker"):
                    r2 = client.chat.completions.create(...)

    Le span_id de chaque agent est automatiquement propagé comme parent_span_id
    pour tout appel LLM ou outil exécuté à l'intérieur du bloc.
    """
    sid = new_span_id()
    p_sid = current_span_id()
    push_span(sid)
    t_start = time.time()
    _result = {"status": "completed", "error": None}

    try:
        yield sid
    except Exception as e:
        _result["status"] = "failed"
        _result["error"] = str(e)
        raise
    finally:
        latency_ms = int((time.time() - t_start) * 1000)
        send_span({
            "span_type": "agent",
            "span_id": sid,
            "parent_span_id": p_sid,
            "name": agent_name,
            "agent_name": agent_name,
            "role": role,
            "status": _result["status"],
            "error": _result["error"],
            "metadata": metadata or {},
            "latency_ms": latency_ms,
            "trace_id": get_trace_id() or "",
            "project_id": get_project_id(),
            "step_index": next_step_index(),
        })
        pop_span()


# ─── tool_span ───────────────────────────────────────────────────────────────

@contextmanager
def tool_span(tool_name: str, tool_input: Optional[dict] = None):
    """
    Context manager pour capturer un appel outil (non-LLM).

    Usage:
        with llminsight.tool_span("web_search", {"query": "prix Tesla"}):
            results = search_web("prix Tesla")
    """
    sid = new_span_id()
    p_sid = current_span_id()
    push_span(sid)
    t_start = time.time()
    _result = {"output": None, "error": None}

    class _ToolCapture:
        def set_output(self, output):
            _result["output"] = str(output) if output is not None else None

        def set_error(self, error):
            _result["error"] = str(error) if error is not None else None

    capture = _ToolCapture()

    try:
        yield capture
    except Exception as e:
        _result["error"] = str(e)
        raise
    finally:
        latency_ms = int((time.time() - t_start) * 1000)
        send_span({
            "span_type": "tool",
            "span_id": sid,
            "parent_span_id": p_sid,
            "name": tool_name,
            "tool_name": tool_name,
            "tool_input": tool_input or {},
            "tool_output": _result["output"],
            "error": _result["error"],
            "latency_ms": latency_ms,
            "trace_id": get_trace_id(),
            "project_id": get_project_id(),
            "step_index": next_step_index(),
        })
        pop_span()


# ─── observe (décorateur) ────────────────────────────────────────────────────

def observe(name: Optional[str] = None, metadata: Optional[dict] = None):
    """
    Décorateur qui encapsule une fonction dans une trace automatique.

    Usage:
        @llminsight.observe(name="planning")
        def plan(messages):
            return openai.chat.completions.create(model="gpt-4o", messages=messages)
    """
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            trace_name = name or func.__name__
            with trace(trace_name, metadata):
                return func(*args, **kwargs)

        @functools.wraps(func)
        async def async_wrapper(*args, **kwargs):
            trace_name = name or func.__name__
            with trace(trace_name, metadata):
                return await func(*args, **kwargs)

        import asyncio
        if asyncio.iscoroutinefunction(func):
            return async_wrapper
        return wrapper

    return decorator
