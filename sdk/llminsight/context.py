"""
Propagation automatique du trace_id, span_id et step_index à travers les threads
et coroutines. Utilise contextvars (Python 3.7+) pour une propagation thread-safe
et async-safe.

Hiérarchie des spans :
  trace
  └─ agent_span (manager)          ← parent_span_id = None
     ├─ llm_span (décision)        ← parent_span_id = manager
     ├─ agent_span (worker-1)      ← parent_span_id = manager
     │  └─ llm_span (exécution)    ← parent_span_id = worker-1
     └─ agent_span (worker-2)      ← parent_span_id = manager
        ├─ tool_span (web_search)  ← parent_span_id = worker-2
        └─ llm_span (synthèse)    ← parent_span_id = worker-2
"""

from contextvars import ContextVar
from typing import Optional, List
import threading
import time
import uuid

_current_trace_id: ContextVar[Optional[str]] = ContextVar("llminsight_trace_id", default=None)
_current_project_id: ContextVar[Optional[str]] = ContextVar("llminsight_project_id", default=None)
_current_api_key: ContextVar[Optional[str]] = ContextVar("llminsight_api_key", default=None)

# --- Agent execution context (set by the user application) ---
_agent_context: ContextVar[dict] = ContextVar("llminsight_agent_ctx", default={})

# Pile de span_ids pour le nesting hiérarchique (agent → sous-agent → LLM)
_span_stack: ContextVar[List[str]] = ContextVar("llminsight_span_stack", default=[])

_step_counter = threading.local()

_auto_trace_lock = threading.Lock()
_auto_trace_id: Optional[str] = None
_auto_trace_last_activity: float = 0.0
AUTO_TRACE_TIMEOUT_S = 30.0


def new_span_id() -> str:
    return f"span-{uuid.uuid4().hex[:12]}"


def push_span(span_id: str):
    """Empile un span_id (on entre dans un agent/outil enfant)."""
    stack = _span_stack.get().copy()
    stack.append(span_id)
    _span_stack.set(stack)


def pop_span():
    """Dépile le span_id courant (on sort de l'agent/outil enfant)."""
    stack = _span_stack.get().copy()
    if stack:
        stack.pop()
    _span_stack.set(stack)


def current_span_id() -> Optional[str]:
    """Retourne le span_id courant (sommet de la pile)."""
    stack = _span_stack.get()
    return stack[-1] if stack else None


def parent_span_id() -> Optional[str]:
    """Retourne le span_id parent (avant-dernier de la pile)."""
    stack = _span_stack.get()
    return stack[-2] if len(stack) >= 2 else None


def set_trace_id(trace_id: str):
    _current_trace_id.set(trace_id)
    _step_counter.value = 0


def get_trace_id() -> Optional[str]:
    return _current_trace_id.get()


def clear_trace_id():
    _current_trace_id.set(None)
    _step_counter.value = 0


def next_step_index() -> int:
    current = getattr(_step_counter, "value", 0)
    _step_counter.value = current + 1
    return current


def set_project_id(project_id: str):
    _current_project_id.set(project_id)


def get_project_id() -> Optional[str]:
    return _current_project_id.get()


def set_api_key(api_key: str):
    _current_api_key.set(api_key)


def get_api_key() -> Optional[str]:
    return _current_api_key.get()


def get_or_create_auto_trace() -> str:
    """
    Si aucune trace n'est active (l'utilisateur n'a pas utilisé `with trace(...)`),
    crée automatiquement une trace. Les appels LLM espacés de plus de
    AUTO_TRACE_TIMEOUT_S secondes créent une nouvelle trace.
    """
    global _auto_trace_id, _auto_trace_last_activity

    explicit = _current_trace_id.get()
    if explicit is not None:
        return explicit

    now = time.time()
    with _auto_trace_lock:
        if _auto_trace_id is None or (now - _auto_trace_last_activity > AUTO_TRACE_TIMEOUT_S):
            from .client import start_trace, end_trace

            if _auto_trace_id is not None:
                end_trace(_auto_trace_id, status="completed")

            _auto_trace_id = start_trace("auto-trace", {"auto": True})
            if _auto_trace_id is None:
                import uuid
                _auto_trace_id = f"local-{uuid.uuid4().hex[:12]}"
            set_trace_id(_auto_trace_id)

        _auto_trace_last_activity = now
        return _auto_trace_id


def set_agent_context(**kwargs):
    """Set agent execution metadata for the next LLM span.

    Call this just before your agent makes an LLM call to attach
    runtime context that the SDK cannot detect automatically.
    All fields are optional — use only what is relevant for your agent.

    Args:
        iteration (int):          agent loop iteration number (any loop-based agent)
        action_success (bool):    did the action achieve its goal? (True/False)
        similarity_score (float): optional similarity/confidence metric (0.0 – 1.0)
        feedback (str):           optional reflection or evaluation result (any string)

    Example (generic agentic loop)::

        import llminsight
        llminsight.set_agent_context(
            iteration=step_number,
            action_success=goal_reached,
            feedback=evaluation_result,
        )
        response = llm_client.chat(...)
    """
    ctx = _agent_context.get().copy()
    ctx.update(kwargs)
    _agent_context.set(ctx)


def get_agent_context() -> dict:
    return _agent_context.get()


def clear_agent_context():
    _agent_context.set({})


def close_auto_trace():
    """Ferme la trace automatique si elle existe (appelé à l'arrêt)."""
    global _auto_trace_id
    with _auto_trace_lock:
        if _auto_trace_id is not None:
            from .client import end_trace
            end_trace(_auto_trace_id, status="completed")
            _auto_trace_id = None
