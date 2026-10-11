"""
Auto-instrumentation universelle.
Patche TOUS les providers LLM et TOUS les frameworks d'agents automatiquement.

Providers couverts :
  - OpenAI (+ Azure OpenAI, DeepSeek, Ollama compat, LiteLLM, toute lib basée sur openai SDK)
  - Anthropic
  - Google Generative AI (Gemini)
  - Mistral
  - Cohere

Frameworks couverts (via patch du BaseChatModel LangChain) :
  - LangChain / LangGraph / CrewAI / AutoGen / tout framework utilisant LangChain
  - LlamaIndex (callback manager)
"""

import contextvars
import io
import json
import os
import sys
import time
import inspect
import logging
from contextvars import ContextVar
from typing import Any, Optional
from .context import (
    get_project_id, next_step_index, get_or_create_auto_trace, peek_trace_id,
    new_span_id, push_span, pop_span, current_span_id, parent_span_id,
    get_agent_context, clear_agent_context, request_scope, end_request_scope,
    get_trace_id,
)
from .client import send_span,send_log

logger = logging.getLogger("aitrajectiq")

_originals = {}
_patched = False

# Guard against double-capture when a framework (LangChain) wraps a provider SDK (OpenAI).
# Set to True while a framework-level patch is active so the provider-level patch skips.
_in_framework_call: ContextVar[bool] = ContextVar("_in_framework_call", default=False)

_SKIP_MODULES = frozenset({
    "aitrajectiq", "openai", "anthropic", "google", "mistralai", "cohere",
    "langchain_core", "langchain", "llama_index", "httpx", "httpcore",
    "urllib", "http", "threading", "asyncio", "concurrent",
    "multiprocessing", "unittest", "_pytest", "importlib",
    "uvicorn", "starlette", "fastapi", "pydantic",
    # Python stdlib execution/runner modules
    "runpy", "runpy2", "py_compile", "compileall",
    "_bootstrap", "_bootstrap_external",
})

_SDK_DIR = os.path.dirname(os.path.abspath(__file__)).replace("\\", "/").lower() + "/"
_STDLIB_DIRS = tuple(
    os.path.abspath(p).replace("\\", "/").lower() + "/"
    for p in {getattr(sys, "base_prefix", sys.prefix), sys.prefix}
)

# Entry-point style names that say nothing about which agent runs: the frame is named
# after its module instead (`agent_browser_use.main()` → "agent_browser_use").
_GENERIC_FUNCS = frozenset({
    "main", "run", "arun", "_run", "_arun", "invoke", "ainvoke", "execute", "aexecute",
    "__call__", "call", "start", "process", "handle", "wrapper", "inner", "wrapped",
})
_ANONYMOUS_FUNCS = frozenset({"<module>", "<lambda>", "<listcomp>", "<dictcomp>", "<setcomp>", "<genexpr>"})


def _is_library_frame(filename: str) -> bool:
    fn = filename.replace("\\", "/").lower()
    if fn.startswith(_SDK_DIR) or "site-packages/" in fn or "dist-packages/" in fn:
        return True
    if fn.startswith("<"):                       # <frozen runpy>, <string>
        return True
    # stdlib lives under <prefix>/Lib (Windows) or <prefix>/lib/pythonX.Y
    return any(fn.startswith(d + "lib/") for d in _STDLIB_DIRS)


def _frame_name(fi) -> Optional[str]:
    func = fi.function
    if func in _ANONYMOUS_FUNCS:
        return None
    local_self = fi.frame.f_locals.get("self")
    if local_self is not None and not isinstance(local_self, type):
        return f"{type(local_self).__name__}.{func}"
    if func not in _GENERIC_FUNCS:
        return func
    module = fi.frame.f_globals.get("__name__", "") or ""
    if module == "__main__":
        return None                              # the script's own main(): the trace already names it
    return module.rsplit(".", 1)[-1] or None


def _langgraph_nodes() -> list:
    """LangGraph nodes running the current call, outermost first. Subgraphs show up in
    `checkpoint_ns` as "parent_node:<task id>|child_node:<task id>"."""
    try:
        from langchain_core.runnables.config import var_child_runnable_config
    except ImportError:
        return []
    config = var_child_runnable_config.get() or {}
    metadata = config.get("metadata") or {}
    ns = metadata.get("checkpoint_ns") or ""
    nodes = [part.split(":", 1)[0] for part in ns.split("|") if part]
    node = metadata.get("langgraph_node")
    if node and (not nodes or nodes[-1] != node):
        nodes.append(node)
    return [n for n in nodes if n and not n.startswith("__")]


def _get_caller_chain() -> list:
    """Remonte la pile d'appels et extrait les fonctions du code utilisateur.

    Filtre automatiquement les frames des bibliothèques (openai, aitrajectiq,
    langchain, stdlib, site-packages) pour ne garder que le code de l'application,
    puis ajoute les nœuds LangGraph en cours d'exécution.
    Retourne la chaîne du plus externe (main/manager) au plus interne (worker).
    """
    chain = []
    try:
        for fi in inspect.stack():
            module = fi.frame.f_globals.get("__name__", "") or ""
            if module.split(".")[0] in _SKIP_MODULES or _is_library_frame(fi.filename):
                continue
            name = _frame_name(fi)
            if name and (not chain or chain[-1] != name):
                chain.append(name)
    except Exception:
        pass
    chain.reverse()
    try:
        for node in _langgraph_nodes():
            if node not in chain:
                chain.append(node)
    except Exception:
        pass
    return chain


def _serialize_messages(messages: Any) -> list:
    if not messages:
        return []
    result = []
    for m in messages:
        if isinstance(m, dict):
            result.append(m)
        elif hasattr(m, "model_dump"):
            result.append(m.model_dump())
        elif hasattr(m, "content") and hasattr(m, "role"):
            result.append({"role": getattr(m, "role", "unknown"), "content": getattr(m, "content", str(m))})
        elif hasattr(m, "content") and hasattr(m, "type"):
            result.append({"role": getattr(m, "type", "unknown"), "content": getattr(m, "content", str(m))})
        else:
            result.append({"role": "unknown", "content": str(m)})
    return result


def _extract_provider_from_url(url: str) -> str:
    """Extrait le nom du provider depuis le domaine de l'URL.
    Ex: 'https://api.deepseek.com/v1' → 'deepseek'
        'https://rd-flowai.openai.azure.com/' → 'azure_openai'
        'https://my-res.services.ai.azure.com/' → 'azure_ai'
        'http://localhost:11434' → 'localhost'
    """
    from urllib.parse import urlparse
    try:
        host = urlparse(url).hostname or ""
    except Exception:
        return ""
    if "azure" in host:
        if ".openai.azure." in host:
            return "azure_openai"
        return "azure_ai"
    parts = host.replace("www.", "").split(".")
    if len(parts) >= 2:
        return parts[-2]
    return parts[0] if parts else ""


def _detect_provider_from_client(client_self) -> str:
    """Détecte le provider depuis la classe du client et/ou son URL de base.
    Générique : fonctionne avec n'importe quel provider, présent ou futur,
    utilisant n'importe quel standard SDK (OpenAI, Anthropic, etc.).
    """
    cls_name = (type(client_self).__module__ + "." + type(client_self).__qualname__).lower()
    if "azure" in cls_name:
        return "azure_openai"

    base = getattr(client_self, "base_url", None)
    if base:
        provider = _extract_provider_from_url(str(base))
        if provider and provider not in ("localhost", "127", "0"):
            return provider

    return "openai"


def _resolve_model_name(client_self, kwargs, resp) -> str:
    """Extrait le vrai nom du modèle (pas le nom de classe)."""
    model = kwargs.get("model")
    if model:
        return model
    if hasattr(resp, "model") and resp.model:
        return resp.model
    deployment = getattr(client_self, "_deployment_id", None)
    if deployment:
        return deployment
    base = getattr(getattr(client_self, "_client", client_self), "base_url", None)
    if base:
        parts = str(base).split("/deployments/")
        if len(parts) > 1:
            return parts[1].split("/")[0]
    return None


def _send_llm_span(provider, model, messages_in, response_text, tokens_in, tokens_out, temperature, latency_ms,
                   *, cache_read_tokens=None, cache_write_tokens=None, max_tokens=None, status=None, error=None):
    try:
        sid = new_span_id()
        caller_chain = _get_caller_chain()
        agent_ctx = get_agent_context()
        clear_agent_context()
        push_span(sid)
        try:
            send_span({
                "span_type": "llm",
                "span_id": sid,
                "parent_span_id": parent_span_id(),
                "caller_chain": caller_chain,
                "provider": provider,
                "name": model or f"{provider}-call",
                "model": model,
                "messages_in": messages_in,
                "response": response_text,
                "tokens_in": tokens_in,
                "tokens_out": tokens_out,
                "cache_read_tokens": cache_read_tokens or None,
                "cache_write_tokens": cache_write_tokens or None,
                "temperature": temperature,
                "max_tokens": max_tokens,
                "status": status,
                "error": error,
                "latency_ms": latency_ms,
                "trace_id": get_or_create_auto_trace(),
                "project_id": get_project_id(),
                "step_index": next_step_index(),
                # Generic agent execution context (optional, set by the application)
                "iteration": agent_ctx.get("iteration"),
                "action_success": agent_ctx.get("action_success"),
                "similarity_score": agent_ctx.get("similarity_score"),
                "feedback": agent_ctx.get("feedback"),
            })
        finally:
            pop_span()
    except Exception as e:
        logger.debug("AItrajectIQ: span send failed: %s", e)


def _send_usage_span(provider, model, messages_in, response_text, usage: dict, temperature, latency_ms, **extra):
    _send_llm_span(
        provider, model, messages_in, response_text,
        usage.get("tokens_in", 0), usage.get("tokens_out", 0), temperature, latency_ms,
        cache_read_tokens=usage.get("cache_read_tokens"),
        cache_write_tokens=usage.get("cache_write_tokens"),
        **extra,
    )


def _send_error_span(provider, model, messages, exc, started, temperature=None):
    """Record a failed provider call; the exception still reaches the application."""
    try:
        from .guardrails import PromptBlockedError
        if isinstance(exc, PromptBlockedError):
            return                                   # already recorded as a [BLOCKED] span
        _send_llm_span(
            provider, model, _serialize_messages(messages), None, 0, 0, temperature,
            int((time.time() - started) * 1000),
            status="failed", error=f"{type(exc).__name__}: {exc}"[:2000],
        )
    except Exception as e:
        logger.debug("AItrajectIQ: error span failed: %s", e)


def _field(obj, name):
    return obj.get(name) if isinstance(obj, dict) else getattr(obj, name, None)


def _openai_usage(usage) -> dict:
    """OpenAI's prompt_tokens already includes the cached tokens, as the backend expects."""
    if not usage:
        return {"tokens_in": 0, "tokens_out": 0}
    details = _field(usage, "prompt_tokens_details")
    return {
        "tokens_in": _field(usage, "prompt_tokens") or 0,
        "tokens_out": _field(usage, "completion_tokens") or 0,
        "cache_read_tokens": (_field(details, "cached_tokens") if details else None) or None,
    }


def _anthropic_usage(usage) -> dict:
    """Anthropic's input_tokens leaves out cache reads and writes: they are added back,
    since the backend counts them in tokens_in and bills them at the cache rates."""
    if not usage:
        return {"tokens_in": 0, "tokens_out": 0}
    read = _field(usage, "cache_read_input_tokens") or 0
    write = _field(usage, "cache_creation_input_tokens") or 0
    return {
        "tokens_in": (_field(usage, "input_tokens") or 0) + read + write,
        "tokens_out": _field(usage, "output_tokens") or 0,
        "cache_read_tokens": read or None,
        "cache_write_tokens": write or None,
    }


def _safe_provider(client_self) -> str:
    try:
        return _detect_provider_from_client(client_self._client)
    except Exception:
        return "openai"


class _StreamCollector:
    """Wraps a streamed chat completion: the application reads the same chunks, and the
    span is sent once the stream ends (exhausted, closed or failed). The span is sent in
    the context of the call, so it joins the call's trace and parent span."""

    def __init__(self, stream, finish):
        self._stream = stream
        self._finish = finish
        self._ctx = contextvars.copy_context()
        self._parts: list = []
        self._usage = None
        self._model = None
        self._done = False
        self._it = None

    def _collect(self, chunk):
        try:
            if getattr(chunk, "model", None):
                self._model = chunk.model
            if getattr(chunk, "usage", None):
                self._usage = chunk.usage
            for choice in getattr(chunk, "choices", None) or []:
                delta = getattr(choice, "delta", None)
                if delta is not None and getattr(delta, "content", None):
                    self._parts.append(delta.content)
        except Exception:
            pass

    def _end(self, error=None):
        if self._done:
            return
        self._done = True
        try:
            self._ctx.run(self._finish, "".join(self._parts), self._usage, self._model, error)
        except Exception as e:
            logger.debug("AItrajectIQ stream span: %s", e)

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        return getattr(self._stream, name)

    def __del__(self):
        try:
            self._end()
        except Exception:
            pass


class _TracedStream(_StreamCollector):
    def _chunks(self):
        error = None
        try:
            for chunk in self._stream:
                self._collect(chunk)
                yield chunk
        except Exception as e:
            error = e
            raise
        finally:
            self._end(error)

    def __iter__(self):
        return self

    def __next__(self):
        if self._it is None:
            self._it = self._chunks()
        return next(self._it)

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def close(self):
        try:
            close = getattr(self._stream, "close", None)
            if close:
                close()
        finally:
            self._end()


class _TracedAsyncStream(_StreamCollector):
    async def _chunks(self):
        error = None
        try:
            async for chunk in self._stream:
                self._collect(chunk)
                yield chunk
        except Exception as e:
            error = e
            raise
        finally:
            self._end(error)

    def __aiter__(self):
        return self

    async def __anext__(self):
        if self._it is None:
            self._it = self._chunks()
        return await self._it.__anext__()

    async def __aenter__(self):
        return self

    async def __aexit__(self, *exc):
        await self.close()

    async def close(self):
        try:
            close = getattr(self._stream, "close", None)
            if close:
                result = close()
                if inspect.isawaitable(result):
                    await result
        finally:
            self._end()


def _openai_stream_finisher(client_self, kwargs, started):
    def finish(text, usage, model, error):
        _send_usage_span(
            _safe_provider(client_self), kwargs.get("model") or model,
            _serialize_messages(kwargs.get("messages", [])), text, _openai_usage(usage),
            kwargs.get("temperature"), int((time.time() - started) * 1000),
            max_tokens=kwargs.get("max_tokens"),
            status="failed" if error else None,
            error=f"{type(error).__name__}: {error}"[:2000] if error else None,
        )
    return finish


def _check_input_guards(messages, provider="unknown", model="unknown"):
    """Run input guardrails (prompt injection, cost limit) BEFORE the LLM call.
    Raises PromptBlockedError if any check fails. Returns silently otherwise.
    Reusable across all providers.
    """
    from .guardrails import get_input_guardrails, run_input_guardrails, PromptBlockedError
    guards = get_input_guardrails()
    if not guards:
        return
    results = run_input_guardrails(messages)
    failures = [r for r in results if not r.passed]
    if failures:
        for f in failures:
            logger.warning("AItrajectIQ INPUT guardrail BLOCKED [%s]: %s — %s", provider, f.name, f.reason)
        _send_llm_span(
            provider, model, _serialize_messages(messages),
            f"[BLOCKED] {'; '.join(f.reason for f in failures)}",
            0, 0, None, 0,
        )
        raise PromptBlockedError(failures)


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER 1 : OpenAI SDK (couvre aussi Azure, DeepSeek, Ollama, LiteLLM, etc.)
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_openai():
    try:
        import openai
    except ImportError:
        return
    if "openai_sync" in _originals:
        return

    orig = openai.resources.chat.completions.Completions.create
    _originals["openai_sync"] = orig

    def _self_heal_loop(self_ref, orig_fn, args, kwargs):
        """Input guardrails → call → output guardrails → mutate prompt → retry."""
        from .guardrails import (get_guardrails, run_guardrails, build_healed_messages,
                                 get_max_retries)

        # ── INPUT guardrails (shared helper) ──
        _check_input_guards(
            kwargs.get("messages", []),
            _detect_provider_from_client(self_ref._client),
            kwargs.get("model", "unknown"),
        )

        # ── Call the LLM ──
        t = time.time()
        resp = orig_fn(self_ref, *args, **kwargs)
        ms = int((time.time() - t) * 1000)

        guardrails = get_guardrails()
        if not guardrails:
            return resp, ms, 0

        response_text = resp.choices[0].message.content if resp.choices else ""
        results = run_guardrails(response_text)
        failures = [r for r in results if not r.passed]

        if not failures:
            return resp, ms, 0

        max_retries = get_max_retries()
        for attempt in range(1, max_retries + 1):
            logger.info("AItrajectIQ self-heal: attempt %d/%d (failures: %s)",
                        attempt, max_retries,
                        ", ".join(f.name for f in failures))

            # Trace the failed attempt
            try:
                provider = _detect_provider_from_client(self_ref._client)
                model = _resolve_model_name(self_ref, kwargs, resp)
                _send_usage_span(
                    provider, model,
                    _serialize_messages(kwargs.get("messages", [])),
                    response_text, _openai_usage(resp.usage), kwargs.get("temperature"), ms,
                )
            except Exception:
                pass

            from .context import set_agent_context
            set_agent_context(
                feedback=f"guardrail_retry:{attempt}/" + ",".join(f.name for f in failures),
                action_success=False,
            )

            healed = build_healed_messages(
                kwargs.get("messages", []), response_text, failures
            )
            retry_kwargs = {**kwargs, "messages": healed}
            t = time.time()
            resp = orig_fn(self_ref, *args, **retry_kwargs)
            ms = int((time.time() - t) * 1000)
            response_text = resp.choices[0].message.content if resp.choices else ""

            results = run_guardrails(response_text)
            failures = [r for r in results if not r.passed]
            if not failures:
                from .context import set_agent_context
                set_agent_context(feedback=f"guardrail_healed:attempt_{attempt+1}", action_success=True)
                break

        return resp, ms, 0

    def patched(self, *args, **kwargs):
        if _in_framework_call.get(False):
            return orig(self, *args, **kwargs)
        from .guardrails import get_guardrails, get_input_guardrails
        started = time.time()
        if kwargs.get("stream"):
            # Output guardrails need the whole answer: a stream only gets the input ones.
            _check_input_guards(kwargs.get("messages", []), _safe_provider(self), kwargs.get("model", "unknown"))
            try:
                stream = orig(self, *args, **kwargs)
            except Exception as e:
                _send_error_span(_safe_provider(self), kwargs.get("model"), kwargs.get("messages"), e, started,
                                 kwargs.get("temperature"))
                raise
            return _TracedStream(stream, _openai_stream_finisher(self, kwargs, started))
        try:
            if get_guardrails() or get_input_guardrails():
                resp, ms, _ = _self_heal_loop(self, orig, args, kwargs)
            else:
                resp = orig(self, *args, **kwargs)
                ms = int((time.time() - started) * 1000)
        except Exception as e:
            _send_error_span(_safe_provider(self), kwargs.get("model"), kwargs.get("messages"), e, started,
                             kwargs.get("temperature"))
            raise
        try:
            provider = _detect_provider_from_client(self._client)
            model = _resolve_model_name(self, kwargs, resp)
            _send_usage_span(
                provider, model,
                _serialize_messages(kwargs.get("messages", [])),
                resp.choices[0].message.content if resp.choices else "",
                _openai_usage(resp.usage), kwargs.get("temperature"), ms,
                max_tokens=kwargs.get("max_tokens"),
            )
            # Conversational log ingestion (legacy proxy-metrics migration)
            try:
                send_log({
                    "prompt": next((m.get("content") for m in reversed(_serialize_messages(kwargs.get("messages", []))) if m.get("role") == "user"), ""),
                    "history": _serialize_messages(kwargs.get("messages", [])),
                    "output": resp.choices[0].message.content if resp.choices else "",
                    "model": model,
                    "temperature": kwargs.get("temperature"),
                    "max_tokens": kwargs.get("max_tokens"),
                    "latency_ms": ms,
                    "n_tokens": getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0,
                    "retrieved_contexts": None,
                })
            except Exception:
                pass
        except Exception as e:
            logger.debug("AItrajectIQ openai: %s", e)
        return resp

    openai.resources.chat.completions.Completions.create = patched

    # Async
    try:
        orig_a = openai.resources.chat.completions.AsyncCompletions.create
        _originals["openai_async"] = orig_a

        async def patched_a(self, *args, **kwargs):
            if _in_framework_call.get(False):
                return await orig_a(self, *args, **kwargs)
            started = time.time()
            try:
                resp = await orig_a(self, *args, **kwargs)
            except Exception as e:
                _send_error_span(_safe_provider(self), kwargs.get("model"), kwargs.get("messages"), e, started,
                                 kwargs.get("temperature"))
                raise
            if kwargs.get("stream"):
                return _TracedAsyncStream(resp, _openai_stream_finisher(self, kwargs, started))
            ms = int((time.time() - started) * 1000)
            try:
                provider = _detect_provider_from_client(self._client)
                model = _resolve_model_name(self, kwargs, resp)
                _send_usage_span(
                    provider, model,
                    _serialize_messages(kwargs.get("messages", [])),
                    resp.choices[0].message.content if resp.choices else "",
                    _openai_usage(resp.usage), kwargs.get("temperature"), ms,
                    max_tokens=kwargs.get("max_tokens"),
                )
            except Exception as e:
                logger.debug("AItrajectIQ openai async: %s", e)
            return resp

        openai.resources.chat.completions.AsyncCompletions.create = patched_a
    except AttributeError:
        pass

    logger.info("AItrajectIQ: patched openai (covers Azure, DeepSeek, Ollama, LiteLLM, OpenRouter, Groq, Together)")


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER 2 : Anthropic
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_anthropic():
    try:
        import anthropic
    except ImportError:
        return
    if "anthropic_sync" in _originals:
        return

    orig = anthropic.resources.messages.Messages.create
    _originals["anthropic_sync"] = orig

    def _extract_anthropic(kwargs, resp, ms):
        msgs = _serialize_messages(kwargs.get("messages", []))
        sys = kwargs.get("system")
        if sys:
            msgs.insert(0, {"role": "system", "content": sys})
        txt = ""
        if resp.content:
            txt = "".join(b.text for b in resp.content if hasattr(b, "text"))
        _send_usage_span(
            "anthropic", kwargs.get("model"), msgs, txt, _anthropic_usage(resp.usage),
            kwargs.get("temperature"), ms, max_tokens=kwargs.get("max_tokens"),
        )

    def _anthropic_messages(kwargs):
        try:
            msgs = _serialize_messages(kwargs.get("messages", []))
            if kwargs.get("system"):
                msgs.insert(0, {"role": "system", "content": kwargs["system"]})
            return msgs
        except Exception:
            return []

    def patched(self, *args, **kwargs):
        _check_input_guards(kwargs.get("messages", []), "anthropic", kwargs.get("model", "claude"))
        t = time.time()
        try:
            resp = orig(self, *args, **kwargs)
        except Exception as e:
            _send_error_span("anthropic", kwargs.get("model"), _anthropic_messages(kwargs), e, t, kwargs.get("temperature"))
            raise
        try: _extract_anthropic(kwargs, resp, int((time.time()-t)*1000))
        except Exception as e: logger.debug("AItrajectIQ anthropic: %s", e)
        # Conversational log ingestion
        try:
            messages_in = _serialize_messages(kwargs.get("messages", []))
            response_text = ""
            if resp is not None and getattr(resp, "content", None):
                response_text = "".join(b.text for b in resp.content if hasattr(b, "text"))
                model = kwargs.get("model", None)
                if hasattr(resp, "model") and resp.model:
                    model = resp.model
                # best-effort token counts
                tokens_in = getattr(resp.usage, "input_tokens", 0) if getattr(resp, "usage", None) else 0
                tokens_out = getattr(resp.usage, "output_tokens", 0) if getattr(resp, "usage", None) else 0
                ms = int((time.time() - t) * 1000)
                send_log({
                    "prompt": next((m.get("content") for m in reversed(messages_in) if m.get("role") == "user"), ""),
                    "history": messages_in,
                    "output": response_text,
                    "model": model,
                    "temperature": kwargs.get("temperature"),
                    "max_tokens": kwargs.get("max_tokens"),
                    "latency_ms": ms,
                    "n_tokens": tokens_out,
                    "retrieved_contexts": None,
                }) 
        except Exception:
            pass
        return resp

    anthropic.resources.messages.Messages.create = patched

    try:
        orig_a = anthropic.resources.messages.AsyncMessages.create
        _originals["anthropic_async"] = orig_a

        async def patched_a(self, *args, **kwargs):
            t = time.time()
            try:
                resp = await orig_a(self, *args, **kwargs)
            except Exception as e:
                _send_error_span("anthropic", kwargs.get("model"), _anthropic_messages(kwargs), e, t,
                                 kwargs.get("temperature"))
                raise
            try: _extract_anthropic(kwargs, resp, int((time.time()-t)*1000))
            except Exception as e: logger.debug("AItrajectIQ anthropic async: %s", e)
            # Conversational log ingestion
            try:
                messages_in = _serialize_messages(kwargs.get("messages", []))
                response_text = ""
                if resp is not None and getattr(resp, "content", None):
                    response_text = "".join(b.text for b in resp.content if hasattr(b, "text"))
                model = kwargs.get("model", None)
                if hasattr(resp, "model") and resp.model:
                    model = resp.model
                tokens_out = getattr(resp.usage, "output_tokens", 0) if getattr(resp, "usage", None) else 0
                send_log({
                    "prompt": next((m.get("content") for m in reversed(messages_in) if m.get("role") == "user"), ""),
                    "history": messages_in,
                    "output": response_text,
                    "model": model,
                    "temperature": kwargs.get("temperature"),
                    "max_tokens": kwargs.get("max_tokens"),
                    "latency_ms": int((time.time() - t) * 1000),
                    "n_tokens": tokens_out,
                    "retrieved_contexts": None,
                })
            except Exception:
                pass
            return resp

        anthropic.resources.messages.AsyncMessages.create = patched_a
    except AttributeError:
        pass

    logger.info("AItrajectIQ: patched anthropic")


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER 3 : Google Generative AI (Gemini)
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_google():
    try:
        import google.generativeai as genai
    except ImportError:
        return
    if "google_generate" in _originals:
        return

    orig = genai.GenerativeModel.generate_content
    _originals["google_generate"] = orig

    def patched(self, *args, **kwargs):
        prompt = args[0] if args else kwargs.get("contents", "")
        if isinstance(prompt, list):
            msgs = [{"role": "user", "content": str(p)} for p in prompt]
        else:
            msgs = [{"role": "user", "content": str(prompt)}]
        model_name = getattr(self, "model_name", "gemini")
        _check_input_guards(msgs, "google_gemini", model_name)
        t = time.time()
        try:
            resp = orig(self, *args, **kwargs)
        except Exception as e:
            _send_error_span("google_gemini", model_name, msgs, e, t)
            raise
        ms = int((time.time() - t) * 1000)
        try:
            txt = resp.text if hasattr(resp, "text") else str(resp)
            _send_usage_span("google_gemini", model_name, msgs, txt,
                             _gemini_usage(getattr(resp, "usage_metadata", None)), None, ms)
        except Exception as e:
            logger.debug("AItrajectIQ google: %s", e)
        return resp

    genai.GenerativeModel.generate_content = patched
    logger.info("AItrajectIQ: patched google generativeai (Gemini)")


def _gemini_usage(usage) -> dict:
    """Gemini's prompt_token_count already includes the cached content tokens."""
    if not usage:
        return {"tokens_in": 0, "tokens_out": 0}
    return {
        "tokens_in": _field(usage, "prompt_token_count") or 0,
        "tokens_out": _field(usage, "candidates_token_count") or 0,
        "cache_read_tokens": _field(usage, "cached_content_token_count") or None,
    }


def _gemini_messages(contents) -> list:
    if isinstance(contents, str):
        return [{"role": "user", "content": contents}]
    if isinstance(contents, list):
        msgs = []
        for item in contents:
            if isinstance(item, str):
                msgs.append({"role": "user", "content": item})
            elif hasattr(item, "role") and hasattr(item, "parts"):
                msgs.append({"role": getattr(item, "role", None) or "user", "content": str(item.parts)})
            else:
                msgs.append({"role": "user", "content": str(item)})
        return msgs
    return [{"role": "user", "content": str(contents)}]


def _patch_google_genai():
    """Patch the new google.genai SDK (replaces deprecated google.generativeai)."""
    try:
        from google.genai import models as genai_models
    except (ImportError, AttributeError):
        return
    if "google_genai_generate" in _originals:
        return

    # client.models.generate_content(model=..., contents=..., config=...) and its async
    # twin client.aio.models.generate_content; arguments are keyword-only in recent versions.
    def _call_args(args, kwargs):
        model = kwargs.get("model", args[0] if args else None)
        contents = kwargs.get("contents", args[1] if len(args) > 1 else "")
        return (str(model) if model else "gemini"), _gemini_messages(contents)

    def _record(model_name, msgs, resp, t):
        try:
            text = resp.text if hasattr(resp, "text") else str(resp)
            _send_usage_span("google_gemini", model_name, msgs, text,
                             _gemini_usage(getattr(resp, "usage_metadata", None)), None,
                             int((time.time() - t) * 1000))
        except Exception as e:
            logger.debug("AItrajectIQ google.genai: %s", e)

    orig_generate = genai_models.Models.generate_content
    _originals["google_genai_generate"] = orig_generate

    def patched_generate(self, *args, **kwargs):
        model_name, msgs = _call_args(args, kwargs)
        _check_input_guards(msgs, "google_gemini", model_name)
        t = time.time()
        try:
            resp = orig_generate(self, *args, **kwargs)
        except Exception as e:
            _send_error_span("google_gemini", model_name, msgs, e, t)
            raise
        _record(model_name, msgs, resp, t)
        return resp

    genai_models.Models.generate_content = patched_generate

    async_models = getattr(genai_models, "AsyncModels", None)
    if async_models is not None and hasattr(async_models, "generate_content"):
        orig_async = async_models.generate_content
        _originals["google_genai_generate_async"] = orig_async

        async def patched_generate_async(self, *args, **kwargs):
            model_name, msgs = _call_args(args, kwargs)
            _check_input_guards(msgs, "google_gemini", model_name)
            t = time.time()
            try:
                resp = await orig_async(self, *args, **kwargs)
            except Exception as e:
                _send_error_span("google_gemini", model_name, msgs, e, t)
                raise
            _record(model_name, msgs, resp, t)
            return resp

        async_models.generate_content = patched_generate_async

    logger.info("AItrajectIQ: patched google.genai (new Gemini SDK)")


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER 4 : Mistral
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_mistral():
    try:
        from mistralai import Mistral
    except ImportError:
        return
    if "mistral" in _originals:
        return

    try:
        orig = Mistral.chat.complete
        _originals["mistral"] = orig

        def patched(self, *args, **kwargs):
            _check_input_guards(kwargs.get("messages", []), "mistral", kwargs.get("model", "mistral"))
            t = time.time()
            try:
                resp = orig(self, *args, **kwargs)
            except Exception as e:
                _send_error_span("mistral", kwargs.get("model"), kwargs.get("messages"), e, t, kwargs.get("temperature"))
                raise
            ms = int((time.time() - t) * 1000)
            try:
                _send_llm_span(
                    "mistral", kwargs.get("model"),
                    _serialize_messages(kwargs.get("messages", [])),
                    resp.choices[0].message.content if resp.choices else "",
                    getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0,
                    getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0,
                    kwargs.get("temperature"), ms,
                )
            except Exception as e:
                logger.debug("AItrajectIQ mistral: %s", e)
            return resp

        Mistral.chat.complete = patched
        logger.info("AItrajectIQ: patched mistral")
    except (AttributeError, TypeError):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER 5 : AWS Bedrock (boto3 — invoke_model)
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_bedrock():
    try:
        import boto3
    except ImportError:
        return
    if "boto3_client" in _originals:
        return

    # Save whatever is currently boto3.client (may already be mocked by user code)
    _originals["boto3_client"] = boto3.client

    def _patched_boto3_client(service_name=None, **kwargs):
        """Wrapper around boto3.client that intercepts bedrock-runtime."""
        # Call the original (or already-mocked) boto3.client
        client = _originals["boto3_client"](service_name=service_name, **kwargs)

        if service_name == "bedrock-runtime":
            wrap_bedrock_client(client)
        return client

    # Replace boto3.client with our patched version
    boto3.client = _patched_boto3_client

    logger.info("AItrajectIQ: patched AWS Bedrock (boto3 bedrock-runtime invoke_model)")


def _bedrock_request(invoke_kwargs) -> tuple:
    """Request body as a dict, and its messages (system prompt first)."""
    raw = invoke_kwargs.get("body", "{}")
    try:
        if isinstance(raw, (bytes, bytearray)):
            raw = raw.decode("utf-8")
        body = json.loads(raw) if isinstance(raw, str) else raw
    except (ValueError, TypeError):
        body = {}
    if not isinstance(body, dict):
        body = {}
    messages = [
        {"role": m.get("role", "user"), "content": m.get("content", "")}
        for m in body.get("messages") or [] if isinstance(m, dict)
    ]
    if body.get("system"):
        messages.insert(0, {"role": "system", "content": body["system"]})
    return body, messages


def _replayable_body(data: bytes):
    """A fresh body with the bytes already read, readable by the application like the original."""
    try:
        from botocore.response import StreamingBody
        return StreamingBody(io.BytesIO(data), len(data))
    except ImportError:
        return io.BytesIO(data)


def _read_bedrock_body(resp) -> dict:
    """Parse invoke_model's response body. The body is a one-shot stream: it is replaced by a
    fresh stream of the same bytes, so the application still reads the whole answer."""
    if not isinstance(resp, dict) or resp.get("body") is None:
        return {}
    raw = resp["body"]
    if hasattr(raw, "read"):
        data = raw.read()
        if isinstance(data, str):
            data = data.encode("utf-8")
        resp["body"] = _replayable_body(data)
    else:
        data = raw
    if isinstance(data, (bytes, bytearray)):
        data = data.decode("utf-8")
    parsed = json.loads(data) if isinstance(data, str) else data
    return parsed if isinstance(parsed, dict) else {}


def _bedrock_text(data: dict) -> str:
    """Answer text across the Bedrock model families (Anthropic, Nova, Llama, Titan, Mistral, Cohere)."""
    blocks = data.get("content")
    if isinstance(blocks, list):
        return "".join(b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "") for b in blocks)
    nova = ((data.get("output") or {}).get("message") or {}).get("content")
    if isinstance(nova, list):
        return "".join(b.get("text", "") for b in nova if isinstance(b, dict))
    if isinstance(data.get("generation"), str):
        return data["generation"]
    for key, field in (("results", "outputText"), ("outputs", "text"), ("generations", "text")):
        items = data.get(key)
        if isinstance(items, list) and items and isinstance(items[0], dict):
            return str(items[0].get(field, ""))
    return ""


def _bedrock_usage(data: dict) -> dict:
    usage = data.get("usage") or {}
    if "input_tokens" in usage or "cache_read_input_tokens" in usage:
        return _anthropic_usage(usage)
    if "inputTokens" in usage:      # Nova; its cache reads are not part of inputTokens
        read = usage.get("cacheReadInputTokenCount") or 0
        write = usage.get("cacheWriteInputTokenCount") or 0
        return {"tokens_in": (usage.get("inputTokens") or 0) + read + write,
                "tokens_out": usage.get("outputTokens") or 0,
                "cache_read_tokens": read or None, "cache_write_tokens": write or None}
    return {"tokens_in": data.get("prompt_token_count") or 0,
            "tokens_out": data.get("generation_token_count") or 0}


def _traced_invoke_model(orig_invoke, invoke_kwargs):
    model_id = invoke_kwargs.get("modelId", "unknown")
    body, messages_in = _bedrock_request(invoke_kwargs)
    _check_input_guards(messages_in, "aws_bedrock", model_id)
    temperature = body.get("temperature")

    t = time.time()
    try:
        resp = orig_invoke(**invoke_kwargs)
    except Exception as e:
        _send_error_span("aws_bedrock", model_id, messages_in, e, t, temperature)
        raise
    ms = int((time.time() - t) * 1000)

    text, usage = "", {"tokens_in": 0, "tokens_out": 0}
    try:
        data = _read_bedrock_body(resp)
        text, usage = _bedrock_text(data), _bedrock_usage(data)
        if model_id == "unknown":
            model_id = data.get("model", model_id)
    except Exception as e:
        logger.debug("AItrajectIQ bedrock parse response: %s", e)
    _send_usage_span("aws_bedrock", model_id, messages_in, text, usage, temperature, ms,
                     max_tokens=body.get("max_tokens"))
    return resp


def wrap_bedrock_client(client):
    """
    Instrument an existing AWS Bedrock client (or mock) with AItrajectIQ monitoring.
    
    This is useful when:
      - You have a mock Bedrock client for testing/demos
      - You created a client directly without using boto3.client()
      - You want to add instrumentation to an already-existing client

    Usage:
        client = MockBedrockClient()  # or real boto3 bedrock client
        wrap_bedrock_client(client)
        # Now client.invoke_model is instrumented

    Returns the same client object (for chaining).
    """
    if hasattr(client, "_aitrajectiq_wrapped") and client._aitrajectiq_wrapped:
        return client

    orig_invoke = client.invoke_model

    def _instrumented_invoke(**invoke_kwargs):
        return _traced_invoke_model(orig_invoke, invoke_kwargs)

    client.invoke_model = _instrumented_invoke
    client._aitrajectiq_wrapped = True
    return client


def _wrap_client_invoke_model(client):
    """Internal helper: wrap invoke_model on a real boto3 bedrock client."""
    if hasattr(client, "_aitrajectiq_wrapped") and client._aitrajectiq_wrapped:
        return client
    return wrap_bedrock_client(client)


# ═══════════════════════════════════════════════════════════════════════════════
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_cohere():
    try:
        import cohere
    except ImportError:
        return
    if "cohere" in _originals:
        return

    try:
        orig = cohere.Client.chat
        _originals["cohere"] = orig

        def patched(self, *args, **kwargs):
            msg = kwargs.get("message", args[0] if args else "")
            _check_input_guards([{"role": "user", "content": msg}], "cohere", kwargs.get("model", "command"))
            t = time.time()
            try:
                resp = orig(self, *args, **kwargs)
            except Exception as e:
                _send_error_span("cohere", kwargs.get("model", "command"), [{"role": "user", "content": msg}], e, t,
                                 kwargs.get("temperature"))
                raise
            ms = int((time.time() - t) * 1000)
            try:
                _send_llm_span(
                    "cohere", kwargs.get("model", "command"),
                    [{"role": "user", "content": msg}],
                    resp.text if hasattr(resp, "text") else str(resp),
                    getattr(getattr(resp, "meta", None), "tokens", {}).get("input_tokens", 0),
                    getattr(getattr(resp, "meta", None), "tokens", {}).get("output_tokens", 0),
                    kwargs.get("temperature"), ms,
                )
            except Exception as e:
                logger.debug("AItrajectIQ cohere: %s", e)
            return resp

        cohere.Client.chat = patched
        logger.info("AItrajectIQ: patched cohere")
    except (AttributeError, TypeError):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# FRAMEWORK : LangChain BaseChatModel (couvre TOUT : LangChain, LangGraph,
#             CrewAI, AutoGen, et tout framework qui utilise LangChain)
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_langchain():
    try:
        from langchain_core.language_models.chat_models import BaseChatModel
    except ImportError:
        return
    if "langchain_invoke" in _originals:
        return

    # ── Patch invoke (sync) ──
    orig_invoke = BaseChatModel.invoke
    _originals["langchain_invoke"] = orig_invoke

    def _detect_lc_provider(self) -> str:
        cls = type(self).__name__.lower()
        if "azure" in cls: return "azure_openai"
        if "openai" in cls: return "openai"
        if "anthropic" in cls or "claude" in cls: return "anthropic"
        if "gemini" in cls or "google" in cls: return "google_gemini"
        if "mistral" in cls: return "mistral"
        if "cohere" in cls: return "cohere"
        if "deepseek" in cls: return "deepseek"
        if "ollama" in cls: return "ollama"
        if "groq" in cls: return "groq"
        if "fireworks" in cls: return "fireworks"
        if "together" in cls: return "together"
        if "bedrock" in cls: return "aws_bedrock"
        if "huggingface" in cls or "hf" in cls: return "huggingface"
        return cls

    def _extract_model_name(self) -> str:
        for attr in ("model_name", "model", "model_id"):
            v = getattr(self, attr, None)
            if v:
                return str(v)
        return type(self).__name__

    def _extract_lc_response(result):
        if hasattr(result, "content"):
            return str(result.content)
        return str(result)

    def _extract_lc_usage(result) -> tuple:
        meta = getattr(result, "usage_metadata", None) or getattr(result, "response_metadata", {}).get("token_usage", {})
        if isinstance(meta, dict):
            return meta.get("input_tokens", 0) or meta.get("prompt_tokens", 0), meta.get("output_tokens", 0) or meta.get("completion_tokens", 0)
        if meta:
            return getattr(meta, "input_tokens", 0), getattr(meta, "output_tokens", 0)
        return 0, 0

    def patched_invoke(self, input, config=None, **kwargs):
        if isinstance(input, str):
            msgs = [{"role": "user", "content": input}]
        elif isinstance(input, list):
            msgs = _serialize_messages(input)
        else:
            msgs = [{"role": "user", "content": str(input)}]
        _check_input_guards(msgs, _detect_lc_provider(self), _extract_model_name(self))
        token = _in_framework_call.set(True)
        t = time.time()
        try:
            result = orig_invoke(self, input, config=config, **kwargs)
            ms = int((time.time() - t) * 1000)
        except Exception as e:
            _send_error_span(_detect_lc_provider(self), _extract_model_name(self), msgs, e, t,
                             getattr(self, "temperature", None))
            raise
        finally:
            _in_framework_call.reset(token)
        try:
            tin, tout = _extract_lc_usage(result)
            _send_llm_span(
                _detect_lc_provider(self), _extract_model_name(self),
                msgs, _extract_lc_response(result), tin, tout,
                getattr(self, "temperature", None), ms,
            )
        except Exception as e:
            logger.debug("AItrajectIQ langchain invoke: %s", e)
        return result

    BaseChatModel.invoke = patched_invoke

    # ── Patch ainvoke (async) ──
    orig_ainvoke = BaseChatModel.ainvoke
    _originals["langchain_ainvoke"] = orig_ainvoke

    async def patched_ainvoke(self, input, config=None, **kwargs):
        token = _in_framework_call.set(True)
        t = time.time()
        try:
            result = await orig_ainvoke(self, input, config=config, **kwargs)
            ms = int((time.time() - t) * 1000)
        except Exception as e:
            _send_error_span(_detect_lc_provider(self), _extract_model_name(self),
                             input if isinstance(input, list) else [{"role": "user", "content": str(input)}],
                             e, t, getattr(self, "temperature", None))
            raise
        finally:
            _in_framework_call.reset(token)
        try:
            if isinstance(input, str):
                msgs = [{"role": "user", "content": input}]
            elif isinstance(input, list):
                msgs = _serialize_messages(input)
            else:
                msgs = [{"role": "user", "content": str(input)}]
            tin, tout = _extract_lc_usage(result)
            _send_llm_span(
                _detect_lc_provider(self), _extract_model_name(self),
                msgs, _extract_lc_response(result), tin, tout,
                getattr(self, "temperature", None), ms,
            )
            # Conversational log ingestion
            try:
                provider = _detect_lc_provider(self)
                model = _extract_model_name(self)
                response_text = _extract_lc_response(result)
                send_log({
                    "prompt": next((m.get("content") for m in reversed(msgs) if m.get("role") == "user"), ""),
                    "history": msgs,
                    "output": response_text,
                    "model": model,
                    "temperature": getattr(self, "temperature", None),
                    "max_tokens": getattr(self, "max_tokens", None),
                    "latency_ms": ms,
                    "n_tokens": tout,
                    "retrieved_contexts": None,
                })
            except Exception:
                pass
        except Exception as e:
            logger.debug("AItrajectIQ langchain ainvoke: %s", e)
        return result

    BaseChatModel.ainvoke = patched_ainvoke

    logger.info("AItrajectIQ: patched LangChain BaseChatModel (covers LangChain, LangGraph, CrewAI, AutoGen)")


# ═══════════════════════════════════════════════════════════════════════════════
# FRAMEWORK : LlamaIndex
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_llamaindex():
    try:
        from llama_index.core.llms import LLM as LlamaLLM
    except ImportError:
        return
    if "llamaindex_chat" in _originals:
        return

    orig_chat = LlamaLLM.chat
    _originals["llamaindex_chat"] = orig_chat

    def patched_chat(self, messages, **kwargs):
        t = time.time()
        resp = orig_chat(self, messages, **kwargs)
        ms = int((time.time() - t) * 1000)
        try:
            cls = type(self).__name__.lower()
            provider = "openai" if "openai" in cls else "anthropic" if "anthropic" in cls else cls
            msgs = [{"role": getattr(m, "role", "user").value if hasattr(getattr(m, "role", ""), "value") else str(getattr(m, "role", "user")), "content": str(m.content)} for m in messages]
            _send_llm_span(
                provider, getattr(self, "model", cls),
                msgs, str(resp.message.content) if resp.message else "",
                0, 0, None, ms,
            )
        except Exception as e:
            logger.debug("AItrajectIQ llamaindex: %s", e)
        return resp

    LlamaLLM.chat = patched_chat
    logger.info("AItrajectIQ: patched LlamaIndex LLM.chat")


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def _aitrajectiq_child_wrapper(original_target, args, kwargs, parent_trace_id=None):
    """Exécuté dans le process enfant : init + patch avant d'appeler la vraie target.
    Les spans de l'enfant rejoignent la trace du parent qui a lancé le process."""
    import os
    pid = os.environ.get("AITRAJECTIQ_PROJECT_ID")
    key = os.environ.get("AITRAJECTIQ_API_KEY")
    url = os.environ.get("AITRAJECTIQ_BASE_URL", "http://localhost:8000")
    if not key:
        return original_target(*args, **(kwargs or {}))
    try:
        import aitrajectiq as _li
        from .context import set_trace_id
        _li.init(project_id=pid, api_key=key, base_url=url)
        _li.patch()
        if parent_trace_id:
            set_trace_id(parent_trace_id)
    except Exception:
        pass
    try:
        return original_target(*args, **(kwargs or {}))
    finally:
        from .client import flush
        flush(10)


def _patch_multiprocessing():
    """Wrap la target de chaque Process pour injecter AItrajectIQ dans le child."""
    import multiprocessing
    if "multiprocessing_process_init" in _originals:
        return

    _orig_init = multiprocessing.Process.__init__
    _originals["multiprocessing_process_init"] = _orig_init

    def _patched_init(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        if target is not None:
            _orig_init(
                self, group=group,
                target=_aitrajectiq_child_wrapper,
                name=name,
                args=(target, args, kwargs or {}, peek_trace_id()),
                daemon=daemon,
            )
        else:
            _orig_init(self, group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)

    multiprocessing.Process.__init__ = _patched_init
    logger.info("AItrajectIQ: patched multiprocessing (child processes will be instrumented)")


# ═══════════════════════════════════════════════════════════════════════════════
# SERVEURS ASGI (FastAPI / Starlette) : une trace par requête
# ═══════════════════════════════════════════════════════════════════════════════

def _patch_asgi():
    """Each HTTP request gets its own auto-trace, opened on its first span. Tasks started
    by the request (asyncio.create_task) inherit it, so background runs stay grouped."""
    try:
        from starlette.applications import Starlette
    except ImportError:
        return
    if "starlette_call" in _originals:
        return
    orig_call = Starlette.__call__
    _originals["starlette_call"] = orig_call

    async def patched_call(self, scope, receive, send):
        if scope.get("type") != "http" or get_trace_id() is not None:
            return await orig_call(self, scope, receive, send)
        method, path = scope.get("method", ""), scope.get("path", "")
        token = request_scope(f"{method} {path}", {"http": {"method": method, "path": path}})
        try:
            return await orig_call(self, scope, receive, send)
        finally:
            end_request_scope(token)

    Starlette.__call__ = patched_call
    logger.info("AItrajectIQ: patched Starlette/FastAPI (one trace per HTTP request)")


# ═══════════════════════════════════════════════════════════════════════════════
# OUTILS LangChain / LangGraph (BaseTool.run / arun)
# ═══════════════════════════════════════════════════════════════════════════════

_TOOL_OUTPUT_MAX = 20_000


def _tool_input_dict(tool_input) -> dict:
    if isinstance(tool_input, dict):
        return tool_input
    if tool_input is None:
        return {}
    return {"input": str(tool_input)}


def _send_tool_span(sid, parent, name, tool_input, output, error, ms):
    try:
        send_span({
            "span_type": "tool",
            "span_id": sid,
            "parent_span_id": parent,
            "caller_chain": _get_caller_chain(),
            "name": name,
            "tool_name": name,
            "tool_input": _tool_input_dict(tool_input),
            "tool_output": None if output is None else str(output)[:_TOOL_OUTPUT_MAX],
            "error": error,
            "latency_ms": ms,
            "trace_id": get_or_create_auto_trace(),
            "project_id": get_project_id(),
            "step_index": next_step_index(),
        })
    except Exception as e:
        logger.debug("AItrajectIQ tool span: %s", e)


def _patch_langchain_tools():
    try:
        from langchain_core.tools import BaseTool
    except ImportError:
        return
    if "langchain_tool_run" in _originals:
        return
    orig_run, orig_arun = BaseTool.run, BaseTool.arun
    _originals["langchain_tool_run"] = orig_run
    _originals["langchain_tool_arun"] = orig_arun

    def _input_of(args, kwargs):
        return args[0] if args else kwargs.get("tool_input")

    def patched_run(self, *args, **kwargs):
        sid, parent = new_span_id(), current_span_id()
        push_span(sid)
        t = time.time()
        output, error = None, None
        try:
            output = orig_run(self, *args, **kwargs)
            return output
        except Exception as e:
            error = str(e)
            raise
        finally:
            pop_span()
            _send_tool_span(sid, parent, self.name, _input_of(args, kwargs), output, error,
                            int((time.time() - t) * 1000))

    async def patched_arun(self, *args, **kwargs):
        sid, parent = new_span_id(), current_span_id()
        push_span(sid)
        t = time.time()
        output, error = None, None
        try:
            output = await orig_arun(self, *args, **kwargs)
            return output
        except Exception as e:
            error = str(e)
            raise
        finally:
            pop_span()
            _send_tool_span(sid, parent, self.name, _input_of(args, kwargs), output, error,
                            int((time.time() - t) * 1000))

    BaseTool.run = patched_run
    BaseTool.arun = patched_arun
    logger.info("AItrajectIQ: patched LangChain BaseTool (tool calls of agents and LangGraph)")


def patch_all():
    global _patched
    if _patched:
        return

    _patch_openai()       # OpenAI, Azure, DeepSeek, Ollama, LiteLLM, OpenRouter, Groq, Together, Fireworks, Perplexity
    _patch_anthropic()    # Anthropic Claude
    _patch_google()       # Google Gemini (google.generativeai, deprecated SDK)
    _patch_google_genai() # Google Gemini (google.genai)
    _patch_mistral()      # Mistral
    _patch_bedrock()      # AWS Bedrock (boto3 bedrock-runtime)
    _patch_cohere()       # Cohere
    _patch_langchain()    # LangChain, LangGraph, CrewAI, AutoGen (tout BaseChatModel)
    _patch_langchain_tools()  # Outils LangChain / LangGraph
    _patch_llamaindex()   # LlamaIndex
    _patch_multiprocessing()  # Process enfants
    _patch_asgi()         # FastAPI / Starlette : une trace par requête

    _patched = True


def unpatch_all():
    global _patched

    if "openai_sync" in _originals:
        import openai
        openai.resources.chat.completions.Completions.create = _originals.pop("openai_sync")
        if "openai_async" in _originals:
            openai.resources.chat.completions.AsyncCompletions.create = _originals.pop("openai_async")

    if "anthropic_sync" in _originals:
        import anthropic
        anthropic.resources.messages.Messages.create = _originals.pop("anthropic_sync")
        if "anthropic_async" in _originals:
            anthropic.resources.messages.AsyncMessages.create = _originals.pop("anthropic_async")

    if "google_generate" in _originals:
        import google.generativeai as genai
        genai.GenerativeModel.generate_content = _originals.pop("google_generate")

    if "google_genai_generate" in _originals:
        from google.genai import models as genai_models
        genai_models.Models.generate_content = _originals.pop("google_genai_generate")
        if "google_genai_generate_async" in _originals:
            genai_models.AsyncModels.generate_content = _originals.pop("google_genai_generate_async")

    if "mistral" in _originals:
        from mistralai import Mistral
        try: Mistral.chat.complete = _originals.pop("mistral")
        except: pass

    if "cohere" in _originals:
        import cohere
        cohere.Client.chat = _originals.pop("cohere")

    if "boto3_client" in _originals:
        import boto3
        boto3.client = _originals.pop("boto3_client")

    if "langchain_invoke" in _originals:
        from langchain_core.language_models.chat_models import BaseChatModel
        BaseChatModel.invoke = _originals.pop("langchain_invoke")
        if "langchain_ainvoke" in _originals:
            BaseChatModel.ainvoke = _originals.pop("langchain_ainvoke")

    if "llamaindex_chat" in _originals:
        from llama_index.core.llms import LLM as LlamaLLM
        LlamaLLM.chat = _originals.pop("llamaindex_chat")

    if "langchain_tool_run" in _originals:
        from langchain_core.tools import BaseTool
        BaseTool.run = _originals.pop("langchain_tool_run")
        BaseTool.arun = _originals.pop("langchain_tool_arun")

    if "starlette_call" in _originals:
        from starlette.applications import Starlette
        Starlette.__call__ = _originals.pop("starlette_call")

    if "multiprocessing_process_init" in _originals:
        import multiprocessing
        multiprocessing.Process.__init__ = _originals.pop("multiprocessing_process_init")

    _patched = False
