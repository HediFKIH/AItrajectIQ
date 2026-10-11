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
import os
import threading
import time
import uuid

_current_trace_id: ContextVar[Optional[str]] = ContextVar("aitrajectiq_trace_id", default=None)
_current_project_id: ContextVar[Optional[str]] = ContextVar("aitrajectiq_project_id", default=None)
_current_api_key: ContextVar[Optional[str]] = ContextVar("aitrajectiq_api_key", default=None)

# --- Agent execution context (set by the user application) ---
_agent_context: ContextVar[dict] = ContextVar("aitrajectiq_agent_ctx", default={})

# Pile de span_ids pour le nesting hiérarchique (agent → sous-agent → LLM)
_span_stack: ContextVar[List[str]] = ContextVar("aitrajectiq_span_stack", default=[])

_step_counter = threading.local()

AUTO_TRACE_TIMEOUT_S = float(os.environ.get("AITRAJECTIQ_AUTO_TRACE_IDLE_S", "60"))


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


class AutoTrace:
    """A trace opened lazily on the first span and closed after AUTO_TRACE_TIMEOUT_S
    seconds without activity (or at exit). Activity after that opens a new trace.

    One process-wide instance groups the spans of a script; servers get one instance
    per HTTP request (see `request_scope`), shared by the tasks the request spawns."""

    def __init__(self, name: str, metadata: Optional[dict] = None):
        self.name = name
        self.metadata = {"auto": True, **(metadata or {})}
        self.trace_id: Optional[str] = None
        self.last_activity = 0.0
        self._lock = threading.Lock()
        self._timer: Optional[threading.Timer] = None

    def current(self) -> str:
        now = time.time()
        previous = None
        with self._lock:
            if self.trace_id is None or now - self.last_activity > AUTO_TRACE_TIMEOUT_S:
                from .client import start_trace

                previous = (self.trace_id, self.last_activity) if self.trace_id else None
                self.trace_id = start_trace(self.name, self.metadata) or f"local-{uuid.uuid4().hex[:12]}"
                _step_counter.value = 0
                _open_auto_traces.add(self)
            self.last_activity = now
            self._arm()
            trace_id = self.trace_id
        if previous:
            _end(*previous)
        return trace_id

    def _arm(self):
        if self._timer is not None:
            self._timer.cancel()
        self._timer = threading.Timer(AUTO_TRACE_TIMEOUT_S + 0.5, self._close_if_idle)
        self._timer.daemon = True
        self._timer.start()

    def _close_if_idle(self):
        with self._lock:
            if self.trace_id is None or time.time() - self.last_activity < AUTO_TRACE_TIMEOUT_S:
                return
            closing = (self.trace_id, self.last_activity)
            self.trace_id = None
            _open_auto_traces.discard(self)
        _end(*closing)

    def close(self):
        with self._lock:
            if self._timer is not None:
                self._timer.cancel()
            closing = (self.trace_id, self.last_activity) if self.trace_id else None
            self.trace_id = None
            _open_auto_traces.discard(self)
        if closing:
            _end(*closing)


def _end(trace_id: str, last_activity: float):
    """Close an auto-trace at the time of its last span, not when the idle timer fired."""
    from .client import end_trace
    end_trace(trace_id, status="completed", ended_at=last_activity or None)


def _script_name() -> str:
    import sys
    script = os.path.basename(sys.argv[0]) if sys.argv and sys.argv[0] else ""
    return os.path.splitext(script)[0] or "auto-trace"


_open_auto_traces: set = set()
_process_auto_trace: Optional[AutoTrace] = None
_process_auto_trace_lock = threading.Lock()
_request_auto_trace: ContextVar[Optional[AutoTrace]] = ContextVar("aitrajectiq_request_trace", default=None)


def request_scope(name: str, metadata: Optional[dict] = None):
    """Group the spans of an HTTP request, and of the tasks it starts, in their own
    auto-trace. Returns a token for `end_request_scope`."""
    return _request_auto_trace.set(AutoTrace(name, metadata))


def end_request_scope(token) -> None:
    _request_auto_trace.reset(token)


def peek_trace_id() -> Optional[str]:
    """The trace a child process should join. Opens the request's trace if needed, but
    never the process-wide one: a script that only spawns workers stays trace-free."""
    explicit = _current_trace_id.get()
    if explicit is not None:
        return explicit
    scoped = _request_auto_trace.get()
    if scoped is not None:
        return scoped.current()
    return _process_auto_trace.trace_id if _process_auto_trace is not None else None


def get_or_create_auto_trace() -> str:
    """
    Trace des spans émis hors d'un `with trace(...)` : celle de la requête HTTP en
    cours si l'application est un serveur instrumenté, sinon une trace partagée par
    tout le programme. Elle est fermée après AUTO_TRACE_TIMEOUT_S secondes sans
    activité ou à l'arrêt ; l'activité suivante ouvre une nouvelle trace.
    """
    global _process_auto_trace

    explicit = _current_trace_id.get()
    if explicit is not None:
        return explicit

    scoped = _request_auto_trace.get()
    if scoped is not None:
        return scoped.current()

    if _process_auto_trace is None:
        with _process_auto_trace_lock:
            if _process_auto_trace is None:
                _process_auto_trace = AutoTrace(_script_name())
    return _process_auto_trace.current()


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

        import aitrajectiq
        aitrajectiq.set_agent_context(
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
    """Ferme les traces automatiques encore ouvertes (appelé à l'arrêt)."""
    for auto in list(_open_auto_traces):
        auto.close()
