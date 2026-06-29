"""
Intégration LangChain / LangGraph via CallbackHandler.

Usage:
    from llminsight.integrations.langchain import LLMInsightCallback

    callback = LLMInsightCallback(trace_name="my-agent")
    chain.invoke({"input": "..."}, config={"callbacks": [callback]})
"""

import time
import uuid
import logging
from typing import Any, Optional, Union

from ..client import send_span, start_trace, end_trace
from ..context import set_trace_id, get_trace_id, clear_trace_id, next_step_index, get_project_id

logger = logging.getLogger("llminsight")

try:
    from langchain_core.callbacks import BaseCallbackHandler
except ImportError:
    try:
        from langchain.callbacks.base import BaseCallbackHandler
    except ImportError:
        class BaseCallbackHandler:
            """Stub when langchain is not installed."""
            pass


class LLMInsightCallback(BaseCallbackHandler):
    """
    LangChain callback qui capture chaque appel LLM et tool comme un span LLMInsight.
    Crée automatiquement une trace au premier événement.
    """

    def __init__(self, trace_name: Optional[str] = None, metadata: Optional[dict] = None):
        super().__init__()
        self.trace_name = trace_name or f"langchain-{uuid.uuid4().hex[:8]}"
        self.metadata = metadata or {}
        self._trace_id: Optional[str] = None
        self._llm_starts: dict[str, dict] = {}
        self._tool_starts: dict[str, dict] = {}

    def _ensure_trace(self):
        if self._trace_id is None:
            self._trace_id = start_trace(self.trace_name, self.metadata)
            if self._trace_id:
                set_trace_id(self._trace_id)
            else:
                self._trace_id = f"local-{uuid.uuid4().hex[:12]}"
                set_trace_id(self._trace_id)

    # ── LLM events ───────────────────────────────────────────────────────

    def on_llm_start(self, serialized: dict, prompts: list[str], *, run_id: Any, **kwargs):
        self._ensure_trace()
        self._llm_starts[str(run_id)] = {
            "t_start": time.time(),
            "model": serialized.get("kwargs", {}).get("model_name")
                     or serialized.get("id", ["unknown"])[-1],
            "messages_in": [{"role": "user", "content": p} for p in prompts],
        }

    def on_chat_model_start(self, serialized: dict, messages: list, *, run_id: Any, **kwargs):
        self._ensure_trace()
        flat_messages = []
        for msg_list in messages:
            if isinstance(msg_list, list):
                for m in msg_list:
                    flat_messages.append({
                        "role": getattr(m, "type", "unknown"),
                        "content": getattr(m, "content", str(m)),
                    })
            else:
                flat_messages.append({
                    "role": getattr(msg_list, "type", "unknown"),
                    "content": getattr(msg_list, "content", str(msg_list)),
                })

        model_name = (
            serialized.get("kwargs", {}).get("model_name")
            or serialized.get("kwargs", {}).get("model")
            or serialized.get("id", ["unknown"])[-1]
        )

        self._llm_starts[str(run_id)] = {
            "t_start": time.time(),
            "model": model_name,
            "messages_in": flat_messages,
        }

    def on_llm_end(self, response: Any, *, run_id: Any, **kwargs):
        start_info = self._llm_starts.pop(str(run_id), None)
        if not start_info:
            return

        latency_ms = int((time.time() - start_info["t_start"]) * 1000)

        response_text = ""
        tokens_in = 0
        tokens_out = 0

        if hasattr(response, "generations") and response.generations:
            gen = response.generations[0]
            if isinstance(gen, list) and gen:
                response_text = gen[0].text if hasattr(gen[0], "text") else str(gen[0])
            elif hasattr(gen, "text"):
                response_text = gen.text

        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage", {})
            tokens_in = usage.get("prompt_tokens", 0)
            tokens_out = usage.get("completion_tokens", 0)

        provider = "langchain"
        model = start_info.get("model", "unknown")
        if "gpt" in model or "o1" in model or "o3" in model:
            provider = "openai"
        elif "claude" in model:
            provider = "anthropic"
        elif "mistral" in model or "mixtral" in model:
            provider = "mistral"

        span_data = {
            "span_type": "llm",
            "provider": provider,
            "name": model,
            "model": model,
            "messages_in": start_info["messages_in"],
            "response": response_text,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "latency_ms": latency_ms,
            "trace_id": self._trace_id,
            "project_id": get_project_id(),
            "step_index": next_step_index(),
        }
        send_span(span_data)

    def on_llm_error(self, error: Union[Exception, KeyboardInterrupt], *, run_id: Any, **kwargs):
        start_info = self._llm_starts.pop(str(run_id), None)
        if not start_info:
            return

        latency_ms = int((time.time() - start_info["t_start"]) * 1000)
        span_data = {
            "span_type": "llm",
            "provider": "langchain",
            "name": start_info.get("model", "unknown"),
            "model": start_info.get("model"),
            "messages_in": start_info["messages_in"],
            "response": None,
            "error": str(error),
            "latency_ms": latency_ms,
            "trace_id": self._trace_id,
            "project_id": get_project_id(),
            "step_index": next_step_index(),
        }
        send_span(span_data)

    # ── Tool events ──────────────────────────────────────────────────────

    def on_tool_start(self, serialized: dict, input_str: str, *, run_id: Any, **kwargs):
        self._ensure_trace()
        self._tool_starts[str(run_id)] = {
            "t_start": time.time(),
            "tool_name": serialized.get("name", "unknown_tool"),
            "tool_input": input_str,
        }

    def on_tool_end(self, output: str, *, run_id: Any, **kwargs):
        start_info = self._tool_starts.pop(str(run_id), None)
        if not start_info:
            return

        latency_ms = int((time.time() - start_info["t_start"]) * 1000)
        span_data = {
            "span_type": "tool",
            "name": start_info["tool_name"],
            "tool_name": start_info["tool_name"],
            "tool_input": {"input": start_info["tool_input"]},
            "tool_output": str(output),
            "latency_ms": latency_ms,
            "trace_id": self._trace_id,
            "project_id": get_project_id(),
            "step_index": next_step_index(),
        }
        send_span(span_data)

    def on_tool_error(self, error: Union[Exception, KeyboardInterrupt], *, run_id: Any, **kwargs):
        start_info = self._tool_starts.pop(str(run_id), None)
        if not start_info:
            return

        latency_ms = int((time.time() - start_info["t_start"]) * 1000)
        span_data = {
            "span_type": "tool",
            "name": start_info["tool_name"],
            "tool_name": start_info["tool_name"],
            "tool_input": {"input": start_info["tool_input"]},
            "tool_output": None,
            "error": str(error),
            "latency_ms": latency_ms,
            "trace_id": self._trace_id,
            "project_id": get_project_id(),
            "step_index": next_step_index(),
        }
        send_span(span_data)

    # ── Chain events (trace lifecycle) ───────────────────────────────────

    def on_chain_end(self, outputs: dict, *, run_id: Any, parent_run_id: Any = None, **kwargs):
        if parent_run_id is None and self._trace_id:
            end_trace(self._trace_id, status="completed")
            clear_trace_id()
            self._trace_id = None

    def on_chain_error(self, error: Union[Exception, KeyboardInterrupt], *, run_id: Any, parent_run_id: Any = None, **kwargs):
        if parent_run_id is None and self._trace_id:
            end_trace(self._trace_id, status="failed")
            clear_trace_id()
            self._trace_id = None
