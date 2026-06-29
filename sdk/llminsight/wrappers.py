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

import time
import inspect
import logging
from contextvars import ContextVar
from typing import Any
from .context import (
    get_project_id, next_step_index, get_or_create_auto_trace,
    new_span_id, push_span, pop_span, current_span_id, parent_span_id,
    get_agent_context, clear_agent_context,
)
from .client import send_span

logger = logging.getLogger("llminsight")

_originals = {}
_patched = False

# Guard against double-capture when a framework (LangChain) wraps a provider SDK (OpenAI).
# Set to True while a framework-level patch is active so the provider-level patch skips.
_in_framework_call: ContextVar[bool] = ContextVar("_in_framework_call", default=False)

_SKIP_MODULES = frozenset({
    "llminsight", "openai", "anthropic", "google", "mistralai", "cohere",
    "langchain_core", "langchain", "llama_index", "httpx", "httpcore",
    "urllib", "http", "threading", "asyncio", "concurrent",
    "multiprocessing", "unittest", "_pytest", "importlib",
    "uvicorn", "starlette", "fastapi", "pydantic",
    # Python stdlib execution/runner modules
    "runpy", "runpy2", "py_compile", "compileall",
    "_bootstrap", "_bootstrap_external",
})

# Additional stdlib path fragments (Windows uses \Lib\ not /lib/python)
_STDLIB_PATH_MARKERS = ("/lib/python", "/lib\\python", "\\lib\\runpy", "/lib/runpy",
                        "\\python3", "\\python2", "\\lib\\site")


def _get_caller_chain() -> list:
    """Remonte la pile d'appels et extrait les fonctions du code utilisateur.

    Filtre automatiquement les frames des bibliothèques (openai, llminsight,
    langchain, etc.) pour ne garder que le code de l'application.
    Retourne la chaîne du plus externe (main/manager) au plus interne (worker).
    """
    chain = []
    try:
        for fi in inspect.stack():
            module = fi.frame.f_globals.get("__name__", "") or ""
            top = module.split(".")[0]
            if top in _SKIP_MODULES:
                continue
            fn = fi.filename.replace("\\", "/")
            fn_lower = fn.lower()
            if "site-packages" in fn_lower:
                continue
            # Filter Python stdlib on all platforms (Linux: /lib/python3.x, Windows: /Lib/)
            if ("/lib/python" in fn_lower
                    or "/lib/runpy" in fn_lower
                    or fn_lower.endswith("runpy.py")
                    or fn_lower.endswith("runpy2.py")):
                continue
            func = fi.function
            if func in ("<module>", "<lambda>", "<listcomp>", "<dictcomp>", "<genexpr>"):
                continue
            if func in ("run_path", "run_module", "run_code", "main"):
                continue
            local_self = fi.frame.f_locals.get("self")
            cls = type(local_self).__name__ if local_self and not isinstance(local_self, type) else None
            name = f"{cls}.{func}" if cls else func
            if not chain or chain[-1] != name:
                chain.append(name)
    except Exception:
        pass
    chain.reverse()
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


def _send_llm_span(provider, model, messages_in, response_text, tokens_in, tokens_out, temperature, latency_ms):
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
                "temperature": temperature,
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
        logger.debug("LLMInsight: span send failed: %s", e)


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
            logger.warning("LLMInsight INPUT guardrail BLOCKED [%s]: %s — %s", provider, f.name, f.reason)
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
            logger.info("LLMInsight self-heal: attempt %d/%d (failures: %s)",
                        attempt, max_retries,
                        ", ".join(f.name for f in failures))

            # Trace the failed attempt
            try:
                provider = _detect_provider_from_client(self_ref._client)
                model = _resolve_model_name(self_ref, kwargs, resp)
                _send_llm_span(
                    provider, model,
                    _serialize_messages(kwargs.get("messages", [])),
                    response_text,
                    getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0,
                    getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0,
                    kwargs.get("temperature"), ms,
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
        if get_guardrails() or get_input_guardrails():
            resp, ms, _ = _self_heal_loop(self, orig, args, kwargs)
        else:
            t = time.time()
            resp = orig(self, *args, **kwargs)
            ms = int((time.time() - t) * 1000)
        try:
            provider = _detect_provider_from_client(self._client)
            model = _resolve_model_name(self, kwargs, resp)
            _send_llm_span(
                provider, model,
                _serialize_messages(kwargs.get("messages", [])),
                resp.choices[0].message.content if resp.choices else "",
                getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0,
                getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0,
                kwargs.get("temperature"), ms,
            )
        except Exception as e:
            logger.debug("LLMInsight openai: %s", e)
        return resp

    openai.resources.chat.completions.Completions.create = patched

    # Async
    try:
        orig_a = openai.resources.chat.completions.AsyncCompletions.create
        _originals["openai_async"] = orig_a

        async def patched_a(self, *args, **kwargs):
            if _in_framework_call.get(False):
                return await orig_a(self, *args, **kwargs)
            t = time.time()
            resp = await orig_a(self, *args, **kwargs)
            ms = int((time.time() - t) * 1000)
            try:
                provider = _detect_provider_from_client(self._client)
                model = _resolve_model_name(self, kwargs, resp)
                _send_llm_span(
                    provider, model,
                    _serialize_messages(kwargs.get("messages", [])),
                    resp.choices[0].message.content if resp.choices else "",
                    getattr(resp.usage, "prompt_tokens", 0) if resp.usage else 0,
                    getattr(resp.usage, "completion_tokens", 0) if resp.usage else 0,
                    kwargs.get("temperature"), ms,
                )
            except Exception as e:
                logger.debug("LLMInsight openai async: %s", e)
            return resp

        openai.resources.chat.completions.AsyncCompletions.create = patched_a
    except AttributeError:
        pass

    logger.info("LLMInsight: patched openai (covers Azure, DeepSeek, Ollama, LiteLLM, OpenRouter, Groq, Together)")


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
        _send_llm_span(
            "anthropic", kwargs.get("model"), msgs, txt,
            getattr(resp.usage, "input_tokens", 0) if resp.usage else 0,
            getattr(resp.usage, "output_tokens", 0) if resp.usage else 0,
            kwargs.get("temperature"), ms,
        )

    def patched(self, *args, **kwargs):
        _check_input_guards(kwargs.get("messages", []), "anthropic", kwargs.get("model", "claude"))
        t = time.time()
        resp = orig(self, *args, **kwargs)
        try: _extract_anthropic(kwargs, resp, int((time.time()-t)*1000))
        except Exception as e: logger.debug("LLMInsight anthropic: %s", e)
        return resp

    anthropic.resources.messages.Messages.create = patched

    try:
        orig_a = anthropic.resources.messages.AsyncMessages.create
        _originals["anthropic_async"] = orig_a

        async def patched_a(self, *args, **kwargs):
            t = time.time()
            resp = await orig_a(self, *args, **kwargs)
            try: _extract_anthropic(kwargs, resp, int((time.time()-t)*1000))
            except Exception as e: logger.debug("LLMInsight anthropic async: %s", e)
            return resp

        anthropic.resources.messages.AsyncMessages.create = patched_a
    except AttributeError:
        pass

    logger.info("LLMInsight: patched anthropic")


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
        _check_input_guards(msgs, "google_gemini", getattr(self, "model_name", "gemini"))
        t = time.time()
        resp = orig(self, *args, **kwargs)
        ms = int((time.time() - t) * 1000)
        try:
            txt = resp.text if hasattr(resp, "text") else str(resp)
            tokens_in = getattr(resp, "usage_metadata", None)
            tin = getattr(tokens_in, "prompt_token_count", 0) if tokens_in else 0
            tout = getattr(tokens_in, "candidates_token_count", 0) if tokens_in else 0
            _send_llm_span("google_gemini", self.model_name, msgs, txt, tin, tout, None, ms)
        except Exception as e:
            logger.debug("LLMInsight google: %s", e)
        return resp

    genai.GenerativeModel.generate_content = patched
    logger.info("LLMInsight: patched google generativeai (Gemini)")


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
            resp = orig(self, *args, **kwargs)
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
                logger.debug("LLMInsight mistral: %s", e)
            return resp

        Mistral.chat.complete = patched
        logger.info("LLMInsight: patched mistral")
    except (AttributeError, TypeError):
        pass


# ═══════════════════════════════════════════════════════════════════════════════
# PROVIDER 5 : Cohere
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
            resp = orig(self, *args, **kwargs)
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
                logger.debug("LLMInsight cohere: %s", e)
            return resp

        cohere.Client.chat = patched
        logger.info("LLMInsight: patched cohere")
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
        try:
            t = time.time()
            result = orig_invoke(self, input, config=config, **kwargs)
            ms = int((time.time() - t) * 1000)
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
            logger.debug("LLMInsight langchain invoke: %s", e)
        return result

    BaseChatModel.invoke = patched_invoke

    # ── Patch ainvoke (async) ──
    orig_ainvoke = BaseChatModel.ainvoke
    _originals["langchain_ainvoke"] = orig_ainvoke

    async def patched_ainvoke(self, input, config=None, **kwargs):
        token = _in_framework_call.set(True)
        try:
            t = time.time()
            result = await orig_ainvoke(self, input, config=config, **kwargs)
            ms = int((time.time() - t) * 1000)
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
        except Exception as e:
            logger.debug("LLMInsight langchain ainvoke: %s", e)
        return result

    BaseChatModel.ainvoke = patched_ainvoke

    logger.info("LLMInsight: patched LangChain BaseChatModel (covers LangChain, LangGraph, CrewAI, AutoGen)")


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
            logger.debug("LLMInsight llamaindex: %s", e)
        return resp

    LlamaLLM.chat = patched_chat
    logger.info("LLMInsight: patched LlamaIndex LLM.chat")


# ═══════════════════════════════════════════════════════════════════════════════
# Public API
# ═══════════════════════════════════════════════════════════════════════════════

def _llminsight_child_wrapper(original_target, args, kwargs):
    """Exécuté dans le process enfant : init + patch avant d'appeler la vraie target."""
    import os
    pid = os.environ.get("LLMINSIGHT_PROJECT_ID")
    key = os.environ.get("LLMINSIGHT_API_KEY")
    url = os.environ.get("LLMINSIGHT_BASE_URL", "http://localhost:8085")
    if pid and key:
        try:
            import llminsight as _li
            _li.init(project_id=pid, api_key=key, base_url=url)
            _li.patch()
        except Exception:
            pass
    return original_target(*args, **(kwargs or {}))


def _patch_multiprocessing():
    """Wrap la target de chaque Process pour injecter LLMInsight dans le child."""
    import multiprocessing
    if "multiprocessing_process_init" in _originals:
        return

    _orig_init = multiprocessing.Process.__init__
    _originals["multiprocessing_process_init"] = _orig_init

    def _patched_init(self, group=None, target=None, name=None, args=(), kwargs=None, *, daemon=None):
        if target is not None:
            _orig_init(
                self, group=group,
                target=_llminsight_child_wrapper,
                name=name,
                args=(target, args, kwargs or {}),
                daemon=daemon,
            )
        else:
            _orig_init(self, group=group, target=target, name=name, args=args, kwargs=kwargs, daemon=daemon)

    multiprocessing.Process.__init__ = _patched_init
    logger.info("LLMInsight: patched multiprocessing (child processes will be instrumented)")


def patch_all():
    global _patched
    if _patched:
        return

    _patch_openai()       # OpenAI, Azure, DeepSeek, Ollama, LiteLLM, OpenRouter, Groq, Together, Fireworks, Perplexity
    _patch_anthropic()    # Anthropic Claude
    _patch_google()       # Google Gemini
    _patch_mistral()      # Mistral
    _patch_cohere()       # Cohere
    _patch_langchain()    # LangChain, LangGraph, CrewAI, AutoGen (tout BaseChatModel)
    _patch_llamaindex()   # LlamaIndex
    _patch_multiprocessing()  # Process enfants

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

    if "mistral" in _originals:
        from mistralai import Mistral
        try: Mistral.chat.complete = _originals.pop("mistral")
        except: pass

    if "cohere" in _originals:
        import cohere
        cohere.Client.chat = _originals.pop("cohere")

    if "langchain_invoke" in _originals:
        from langchain_core.language_models.chat_models import BaseChatModel
        BaseChatModel.invoke = _originals.pop("langchain_invoke")
        if "langchain_ainvoke" in _originals:
            BaseChatModel.ainvoke = _originals.pop("langchain_ainvoke")

    if "llamaindex_chat" in _originals:
        from llama_index.core.llms import LLM as LlamaLLM
        LlamaLLM.chat = _originals.pop("llamaindex_chat")

    _patched = False
