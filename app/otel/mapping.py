"""
Map one OpenTelemetry span to AItrajectIQ span fields.

Conventions, tried in this order:
  1. Vercel AI SDK         `ai.*`                     (ai.generateText.doGenerate, ai.toolCall, ...)
  2. OpenInference         `openinference.span.kind`  (LlamaIndex, CrewAI, smolagents, OpenAI Agents SDK, DSPy, ...)
  3. OTel GenAI semconv    `gen_ai.*`                 (official instrumentations, OpenLLMetry, OpenLIT,
                                                        Semantic Kernel, Pydantic AI, Strands, AutoGen, ...)
  4. Traceloop workflows   `traceloop.span.kind`      (OpenLLMetry @workflow / @task / @agent / @tool)

Spans matching none of them (HTTP, database, framework internals) map to None.

Everything is normalized to the shape the evaluators read: `messages_in` is a list of
{"role": str, "content": str} and `response` is plain text, with tool calls rendered
as `[tool_call] name({...})`.
"""

import json
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional

from otel.decode import OtlpSpan

MAX_TEXT = 100_000
MAX_PREVIEW = 2_000


@dataclass
class MappedSpan:
    span_type: str              # llm | tool | agent | chain | retriever | embedding
    name: str
    convention: str
    provider: Optional[str] = None
    model: Optional[str] = None
    messages_in: Optional[list] = None
    response: Optional[str] = None
    tokens_in: Optional[int] = None         # cache included
    tokens_out: Optional[int] = None
    cache_read_tokens: Optional[int] = None
    cache_write_tokens: Optional[int] = None
    temperature: Optional[float] = None
    max_tokens: Optional[int] = None
    tool_name: Optional[str] = None
    tool_input: Optional[dict] = None
    tool_output: Optional[str] = None
    tool_error: Optional[str] = None
    agent_name: Optional[str] = None
    error: Optional[str] = None
    status: str = "ok"
    cost_usd: Optional[float] = None       # cost reported by the instrumentation, if any
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Generic helpers
# ---------------------------------------------------------------------------

def _json(value: Any) -> Any:
    """Parse JSON-looking strings; return anything else unchanged."""
    if isinstance(value, str):
        stripped = value.strip()
        if stripped[:1] in ("{", "["):
            try:
                return json.loads(stripped)
            except ValueError:
                return value
    return value


def _text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    try:
        return json.dumps(value, ensure_ascii=False, default=str)
    except (TypeError, ValueError):
        return str(value)


def _clip(text: Optional[str], limit: int = MAX_TEXT) -> Optional[str]:
    if text is None:
        return None
    return text if len(text) <= limit else text[:limit] + " [truncated]"


def _first(attrs: dict, *keys: str) -> Any:
    for key in keys:
        value = attrs.get(key)
        if value is not None and value != "":
            return value
    return None


def _int_or_none(value: Any) -> Optional[int]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _num(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _lower(value: Any) -> Optional[str]:
    return str(value).strip().lower() if value not in (None, "") else None


def _as_dict(value: Any) -> Optional[dict]:
    value = _json(value)
    if value is None or value == "":
        return None
    return value if isinstance(value, dict) else {"input": value}


def _parsed_if(value: Any, predicate) -> Any:
    """For a JSON string, the parsed value if it satisfies `predicate`, else the string itself.

    Model output is often JSON (`{"action": ...}`) and must stay verbatim; only JSON that
    clearly encodes content parts or messages is unpacked.
    """
    if isinstance(value, str):
        parsed = _json(value)
        if parsed is not value and predicate(parsed):
            return parsed
    return value


def _is_part_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(p, dict) and any(k in p for k in ("type", "text", "toolUse", "toolResult"))
        for p in value
    )


def _is_message(value: Any) -> bool:
    return isinstance(value, dict) and "role" in value and any(
        k in value for k in ("content", "parts", "tool_calls")
    )


def _is_message_list(value: Any) -> bool:
    return isinstance(value, list) and bool(value) and all(
        isinstance(m, dict) and any(k in m for k in ("role", "parts", "message")) for m in value
    )


def _is_prompt_object(value: Any) -> bool:
    return isinstance(value, dict) and ("messages" in value or ("prompt" in value and "role" not in value))


def _is_response_object(value: Any) -> bool:
    return isinstance(value, dict) and (
        isinstance(value.get("choices"), list) or isinstance(value.get("candidates"), list)
    )


def _indexed(attrs: dict, prefix: str) -> list:
    """Group flattened `prefix.N.rest` attributes into [{rest: value}, ...] ordered by N."""
    groups: dict = {}
    start = len(prefix) + 1
    for key, value in attrs.items():
        if not key.startswith(prefix + "."):
            continue
        index, _, rest = key[start:].partition(".")
        if index.isdigit():
            groups.setdefault(int(index), {})[rest] = value
    return [groups[i] for i in sorted(groups)]


# ---------------------------------------------------------------------------
# Message / content normalization
# ---------------------------------------------------------------------------

def _tool_call_text(name: Any, args: Any) -> str:
    args = _json(args)
    args_text = "" if args in (None, "") else (args if isinstance(args, str) else _text(args))
    return f"[tool_call] {name or 'tool'}({args_text})"


def _tool_call_from_obj(call: Any) -> str:
    if not isinstance(call, dict):
        return _text(call)
    function = call.get("function") if isinstance(call.get("function"), dict) else {}
    name = function.get("name") or call.get("name") or call.get("toolName")
    args = function.get("arguments") if function else None
    if args is None:
        args = call.get("arguments", call.get("args", call.get("input")))
    return _tool_call_text(name, args)


def _part_text(part: Any) -> str:
    if isinstance(part, str):
        return part
    if not isinstance(part, dict):
        return _text(part)
    ptype = str(part.get("type") or "").lower().replace("-", "_")
    if ptype in ("reasoning", "thinking", "redacted_thinking"):
        return ""
    if ptype in ("tool_call", "tool_use", "function_call"):
        function = part.get("function") if isinstance(part.get("function"), dict) else {}
        return _tool_call_text(
            part.get("name") or part.get("toolName") or function.get("name"),
            part.get("arguments", part.get("args", part.get("input", function.get("arguments")))),
        )
    if ptype in ("tool_call_response", "tool_result", "function_call_output"):
        result = part.get("response", part.get("result", part.get("output", part.get("content"))))
        return _content_text(result)
    for key in ("text", "content"):
        if isinstance(part.get(key), str):
            return part[key]
    if part.get("value") is not None:     # e.g. AI SDK v5 tool output {"type": "json", "value": ...}
        return part["value"] if isinstance(part["value"], str) else _text(part["value"])
    if "toolUse" in part and isinstance(part["toolUse"], dict):      # Bedrock / Strands blocks
        return _tool_call_text(part["toolUse"].get("name"), part["toolUse"].get("input"))
    if "toolResult" in part and isinstance(part["toolResult"], dict):
        return _content_text(part["toolResult"].get("content"))
    if "content" in part:
        return _content_text(part["content"])
    if ptype:
        return f"[{ptype}]"
    return ""


def _content_text(content: Any) -> str:
    """Flatten message content (string, part list, block list, or their JSON) to text."""
    content = _parsed_if(content, _is_part_list)
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return _part_text(content)
    if isinstance(content, list):
        return "\n".join(t for t in (_part_text(p) for p in content) if t)
    return str(content)


def _message(item: dict, default_role: str) -> dict:
    # gen_ai.choice / OpenAI choice objects wrap the message: {"index", "message": {...}}
    if isinstance(item.get("message"), dict) and "role" not in item:
        item = item["message"]
    role = str(item.get("role") or default_role)
    text = _content_text(item["parts"] if "parts" in item else item.get("content"))
    calls = _json(item.get("tool_calls") or item.get("toolCalls"))
    if isinstance(calls, list) and calls:
        text = "\n".join(t for t in [text, *(_tool_call_from_obj(c) for c in calls)] if t)
    return {"role": role, "content": text}


def normalize_messages(raw: Any, default_role: str = "user") -> list:
    """Normalize any message representation to [{"role", "content": str}]."""
    raw = _parsed_if(raw, lambda v: _is_message_list(v) or _is_message(v) or _is_prompt_object(v))
    if raw is None or raw == "" or raw == []:
        return []
    if isinstance(raw, str):
        return [{"role": default_role, "content": raw}]
    if isinstance(raw, dict):
        if "messages" in raw or ("prompt" in raw and "role" not in raw):
            # {"system", "messages"} (Vercel ai.prompt, OpenAI request bodies) or {"system", "prompt"}
            msgs = (normalize_messages(raw["messages"], default_role) if "messages" in raw
                    else [{"role": "user", "content": _content_text(raw.get("prompt"))}])
            if raw.get("system"):
                msgs.insert(0, {"role": "system", "content": _content_text(raw["system"])})
            return msgs
        raw = [raw]
    if not isinstance(raw, list):
        return [{"role": default_role, "content": _text(raw)}]
    out = []
    for item in raw:
        if isinstance(item, str):
            out.append({"role": default_role, "content": item})
        elif isinstance(item, dict):
            out.append(_message(item, default_role))
    return out


def messages_text(messages: Iterable[dict]) -> str:
    return "\n".join(m["content"] for m in messages if m.get("content"))


def response_text(value: Any) -> str:
    """Best-effort plain-text response from strings, messages, choices or raw API responses."""
    value = _parsed_if(value, lambda v: _is_response_object(v) or _is_message(v) or _is_message_list(v))
    if value is None:
        return ""
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        if isinstance(value.get("choices"), list):        # OpenAI-style response
            return response_text([c.get("message") or c.get("text") for c in value["choices"] if isinstance(c, dict)])
        if isinstance(value.get("candidates"), list):     # Gemini response
            return response_text([c.get("content") for c in value["candidates"] if isinstance(c, dict)])
        if any(k in value for k in ("message", "role", "content", "parts")):
            return messages_text(normalize_messages([value], "assistant"))
        if "output" in value:
            return response_text(value["output"])
        if "text" in value:
            return _text(value["text"])
        return _text(value)
    if isinstance(value, list):
        if all(isinstance(v, str) for v in value):
            return "\n".join(value)
        return messages_text(normalize_messages(value, "assistant"))
    return _text(value)


def _indexed_messages(attrs: dict, prefix: str) -> list:
    """OpenLLMetry `gen_ai.prompt.N.role/content` (+ `tool_calls.M.name/arguments`)."""
    msgs = []
    for item in _indexed(attrs, prefix):
        text = _content_text(item.get("content"))
        calls = [_tool_call_text(c.get("name"), c.get("arguments")) for c in _indexed(item, "tool_calls")]
        msgs.append({
            "role": str(item.get("role") or "user"),
            "content": "\n".join(t for t in [text, *calls] if t),
        })
    return msgs


def _openinference_messages(attrs: dict, prefix: str) -> list:
    """OpenInference `llm.input_messages.N.message.*`."""
    msgs = []
    for item in _indexed(attrs, prefix):
        text = _content_text(item.get("message.content"))
        if not text:
            text = "\n".join(
                str(p["message_content.text"])
                for p in _indexed(item, "message.contents")
                if p.get("message_content.text")
            )
        calls = [
            _tool_call_text(c.get("tool_call.function.name"), c.get("tool_call.function.arguments"))
            for c in _indexed(item, "message.tool_calls")
        ]
        msgs.append({
            "role": str(item.get("message.role") or "user"),
            "content": "\n".join(t for t in [text, *calls] if t),
        })
    return msgs


# ---------------------------------------------------------------------------
# GenAI content events (span events, or OTLP log records linked to the span)
# ---------------------------------------------------------------------------

_MESSAGE_EVENTS = {
    "gen_ai.system.message": "system",
    "gen_ai.user.message": "user",
    "gen_ai.assistant.message": "assistant",
    "gen_ai.tool.message": "tool",
}


def is_content_event(name: Optional[str]) -> bool:
    return bool(name) and (
        name in _MESSAGE_EVENTS
        or name in ("gen_ai.choice", "gen_ai.content.prompt", "gen_ai.content.completion",
                    "gen_ai.client.inference.operation.details")
    )


def event_payload(event: dict) -> dict:
    """Content of a span event (attributes) or log record (body, then attributes)."""
    payload = dict(event.get("attributes") or {})
    body = _json(event.get("body"))
    if isinstance(body, dict):
        payload.update(body)
    elif body is not None:
        payload.setdefault("content", body)
    return payload


def messages_from_events(events: Iterable[dict]) -> tuple:
    """Return (messages_in, response) from GenAI content events, in time order."""
    messages: list = []
    choices: list = []
    system = None
    for event in sorted(events, key=lambda e: e.get("time_ns") or 0):
        name = event.get("name") or ""
        payload = event_payload(event)
        if name in _MESSAGE_EVENTS:
            messages.append(_message(
                {"role": payload.get("role") or _MESSAGE_EVENTS[name],
                 "content": payload.get("content"),
                 "tool_calls": payload.get("tool_calls")},
                _MESSAGE_EVENTS[name],
            ))
        elif name == "gen_ai.choice":
            message = _parsed_if(payload.get("message"), lambda v: _is_part_list(v) or (
                isinstance(v, dict) and any(k in v for k in ("content", "parts", "tool_calls"))
            ))
            if isinstance(message, dict):
                choices.append(messages_text([_message({"role": "assistant", **message}, "assistant")]))
            elif message is not None:
                choices.append(_content_text(message))
            elif payload.get("content") is not None:
                choices.append(_content_text(payload.get("content")))
        elif name == "gen_ai.content.prompt":
            messages.extend(normalize_messages(payload.get("gen_ai.prompt")))
        elif name == "gen_ai.content.completion":
            choices.append(response_text(payload.get("gen_ai.completion")))
        elif name == "gen_ai.client.inference.operation.details":
            messages.extend(normalize_messages(payload.get("gen_ai.input.messages")))
            choices.append(response_text(payload.get("gen_ai.output.messages")))
            system = system or payload.get("gen_ai.system_instructions")
    if system and not any(m["role"] == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": _content_text(system)})
    messages = [{**m, "content": _clip(m["content"])} for m in messages]
    return messages, _clip("\n".join(c for c in choices if c))


# ---------------------------------------------------------------------------
# Convention mappers
# ---------------------------------------------------------------------------

def _io_metadata(mapped: MappedSpan, inp: Any, out: Any) -> None:
    if inp not in (None, ""):
        mapped.metadata["input"] = _clip(_text(_json(inp)), MAX_PREVIEW)
    if out not in (None, ""):
        mapped.metadata["output"] = _clip(_text(_json(out)), MAX_PREVIEW)


def _vercel_operation(span: OtlpSpan) -> Optional[str]:
    op = span.attributes.get("ai.operationId")
    if isinstance(op, str) and op.startswith("ai."):
        return op
    if span.name.startswith("ai.") and any(k.startswith("ai.") for k in span.attributes):
        return span.name.split(" ")[0]
    return None


def _map_vercel(span: OtlpSpan, op: str) -> MappedSpan:
    attrs = span.attributes
    provider = _first(attrs, "ai.model.provider", "gen_ai.system")
    common = dict(
        name=span.name,
        convention="vercel_ai",
        provider=_lower(str(provider).split(".")[0]) if provider else None,
        model=_first(attrs, "ai.response.model", "ai.model.id", "gen_ai.response.model", "gen_ai.request.model"),
    )

    if op.endswith((".doGenerate", ".doStream")):
        response = _first(attrs, "ai.response.text")
        if response is None and attrs.get("ai.response.object") is not None:
            response = _text(_json(attrs["ai.response.object"]))
        calls = _json(attrs.get("ai.response.toolCalls"))
        parts = [response or ""]
        if isinstance(calls, list):
            parts.extend(_tool_call_from_obj(c) for c in calls)
        return MappedSpan(
            span_type="llm",
            messages_in=normalize_messages(attrs.get("ai.prompt.messages")),
            response="\n".join(p for p in parts if p),
            tokens_in=_int_or_none(_first(attrs, "ai.usage.inputTokens", "ai.usage.promptTokens", "gen_ai.usage.input_tokens")),
            tokens_out=_int_or_none(_first(attrs, "ai.usage.outputTokens", "ai.usage.completionTokens", "gen_ai.usage.output_tokens")),
            temperature=_num(_first(attrs, "ai.settings.temperature", "gen_ai.request.temperature")),
            max_tokens=_int_or_none(_first(attrs, "ai.settings.maxOutputTokens", "ai.settings.maxTokens", "gen_ai.request.max_tokens")),
            **common,
        )
    if op.endswith(".doEmbed"):
        return MappedSpan(
            span_type="embedding",
            tokens_in=_int_or_none(_first(attrs, "ai.usage.tokens", "gen_ai.usage.input_tokens")),
            **common,
        )
    if op == "ai.toolCall":
        result = _first(attrs, "ai.toolCall.result", "ai.toolCall.output")
        return MappedSpan(
            span_type="tool",
            tool_name=attrs.get("ai.toolCall.name") or span.name,
            tool_input=_as_dict(_first(attrs, "ai.toolCall.args", "ai.toolCall.input")),
            tool_output=_clip(_text(_json(result))) if result is not None else None,
            **common,
        )
    # Outer ai.generateText / ai.streamText / ai.generateObject / ai.embed: their usage
    # duplicates the inner doGenerate/doStream spans, so they only carry structure.
    mapped = MappedSpan(span_type="chain", **common)
    _io_metadata(mapped, attrs.get("ai.prompt"), _first(attrs, "ai.response.text", "ai.response.object"))
    return mapped


_OPENINFERENCE_KINDS = {
    "LLM": "llm",
    "TOOL": "tool",
    "AGENT": "agent",
    "RETRIEVER": "retriever",
    "RERANKER": "retriever",
    "EMBEDDING": "embedding",
}


def _map_openinference(span: OtlpSpan) -> MappedSpan:
    attrs = span.attributes
    kind = str(attrs.get("openinference.span.kind") or "").upper()
    mapped = MappedSpan(
        span_type=_OPENINFERENCE_KINDS.get(kind, "chain"),
        name=span.name,
        convention="openinference",
    )
    input_value = attrs.get("input.value")
    output_value = attrs.get("output.value")

    if mapped.span_type == "llm":
        mapped.model = _first(attrs, "llm.model_name", "gen_ai.response.model", "gen_ai.request.model")
        mapped.provider = _lower(_first(attrs, "llm.provider", "llm.system", "gen_ai.system"))
        mapped.messages_in = (
            _openinference_messages(attrs, "llm.input_messages") or normalize_messages(input_value)
        )
        outputs = _openinference_messages(attrs, "llm.output_messages")
        mapped.response = messages_text(outputs) if outputs else response_text(output_value)
        mapped.tokens_in = _int_or_none(attrs.get("llm.token_count.prompt"))
        mapped.tokens_out = _int_or_none(attrs.get("llm.token_count.completion"))
        mapped.cost_usd = _num(attrs.get("llm.cost.total"))
        params = _json(attrs.get("llm.invocation_parameters"))
        if isinstance(params, dict):
            mapped.temperature = _num(params.get("temperature"))
            mapped.max_tokens = _int_or_none(_first(
                params, "max_tokens", "max_completion_tokens", "max_output_tokens"
            ))
    elif mapped.span_type == "embedding":
        mapped.model = _first(attrs, "embedding.model_name", "llm.model_name")
        mapped.provider = _lower(_first(attrs, "llm.provider", "llm.system"))
        mapped.tokens_in = _int_or_none(_first(attrs, "llm.token_count.prompt", "llm.token_count.total"))
    elif mapped.span_type == "tool":
        mapped.tool_name = attrs.get("tool.name") or span.name
        mapped.tool_input = _as_dict(input_value)
        mapped.tool_output = _clip(_text(_json(output_value))) if output_value is not None else None
    elif mapped.span_type == "retriever":
        mapped.tool_name = span.name
        mapped.tool_input = _as_dict(input_value)
        docs = [
            str(d["document.content"])
            for prefix in ("retrieval.documents", "reranker.output_documents")
            for d in _indexed(attrs, prefix)
            if d.get("document.content")
        ]
        if docs:
            mapped.tool_output = _clip("\n---\n".join(docs))
        elif output_value is not None:
            mapped.tool_output = _clip(_text(_json(output_value)))
    else:
        if mapped.span_type == "agent":
            mapped.agent_name = _first(attrs, "agent.name", "graph.node.name") or span.name
        _io_metadata(mapped, input_value, output_value)
    return mapped


_GENAI_OPERATIONS = {
    "chat": "llm",
    "chat.completions": "llm",      # Semantic Kernel (pre-1.28 semconv names)
    "text.completions": "llm",
    "text_completion": "llm",
    "completion": "llm",
    "generate_content": "llm",
    "embeddings": "embedding",
    "embedding": "embedding",
    "execute_tool": "tool",
    "invoke_agent": "agent",
    "create_agent": "agent",
}
_LLM_REQUEST_TYPES = {"chat": "llm", "completion": "llm", "embedding": "embedding", "rerank": "retriever"}


def _has_prefix(attrs: dict, prefixes: tuple) -> bool:
    return any(key.startswith(prefixes) for key in attrs)


def _genai_span_type(attrs: dict) -> Optional[str]:
    span_type = _GENAI_OPERATIONS.get(str(attrs.get("gen_ai.operation.name") or "").lower())
    if span_type:
        return span_type
    span_type = _LLM_REQUEST_TYPES.get(str(attrs.get("llm.request.type") or "").lower())
    if span_type:
        return span_type
    if attrs.get("gen_ai.tool.name"):
        return "tool"
    if _first(attrs, "gen_ai.request.model", "gen_ai.response.model") or _has_prefix(
        attrs, ("gen_ai.prompt", "gen_ai.completion", "gen_ai.input.messages", "gen_ai.usage.")
    ):
        return "llm"
    if attrs.get("gen_ai.agent.name"):
        return "agent"
    return None


def _genai_content(span: OtlpSpan) -> tuple:
    attrs = span.attributes
    messages = normalize_messages(attrs.get("gen_ai.input.messages"))
    response = response_text(attrs.get("gen_ai.output.messages"))
    if not messages:
        messages = _indexed_messages(attrs, "gen_ai.prompt") or _indexed_messages(attrs, "llm.prompts")
    if not response:
        response = messages_text(
            _indexed_messages(attrs, "gen_ai.completion") or _indexed_messages(attrs, "llm.completions")
        )
    if not messages and attrs.get("gen_ai.prompt") is not None:
        messages = normalize_messages(attrs["gen_ai.prompt"])
    if not response and attrs.get("gen_ai.completion") is not None:
        response = response_text(attrs["gen_ai.completion"])
    if not messages or not response:
        event_messages, event_response = messages_from_events(
            e for e in span.events if is_content_event(e.get("name"))
        )
        messages = messages or event_messages
        response = response or event_response
    system = attrs.get("gen_ai.system_instructions")
    if system and not any(m["role"] == "system" for m in messages):
        messages.insert(0, {"role": "system", "content": _content_text(system)})
    return messages, response


def _map_genai(span: OtlpSpan) -> Optional[MappedSpan]:
    attrs = span.attributes
    span_type = _genai_span_type(attrs)
    if span_type is None:
        return None
    mapped = MappedSpan(
        span_type=span_type,
        name=span.name,
        convention="genai",
        provider=_lower(_first(attrs, "gen_ai.provider.name", "gen_ai.system")),
        model=_first(attrs, "gen_ai.response.model", "gen_ai.request.model"),
    )
    if span_type in ("llm", "embedding"):
        mapped.tokens_in = _int_or_none(_first(
            attrs, "gen_ai.usage.input_tokens", "gen_ai.usage.prompt_tokens", "llm.usage.prompt_tokens"
        ))
        mapped.tokens_out = _int_or_none(_first(
            attrs, "gen_ai.usage.output_tokens", "gen_ai.usage.completion_tokens", "llm.usage.completion_tokens"
        ))
        mapped.temperature = _num(attrs.get("gen_ai.request.temperature"))
        mapped.max_tokens = _int_or_none(attrs.get("gen_ai.request.max_tokens"))
        mapped.cost_usd = _num(_first(attrs, "gen_ai.usage.cost", "llm.cost.total"))
        mapped.cache_read_tokens = _int_or_none(_first(
            attrs, "gen_ai.usage.cache_read.input_tokens", "gen_ai.usage.cache_read_input_tokens",
            "llm.token_count.prompt_details.cache_read",
        ))
        mapped.cache_write_tokens = _int_or_none(_first(
            attrs, "gen_ai.usage.cache_creation.input_tokens", "gen_ai.usage.cache_creation_input_tokens",
            "llm.token_count.prompt_details.cache_write",
        ))
        if span_type == "llm":
            mapped.messages_in, mapped.response = _genai_content(span)
    elif span_type == "tool":
        mapped.tool_name = attrs.get("gen_ai.tool.name") or span.name
        mapped.tool_input = _as_dict(_first(
            attrs, "gen_ai.tool.call.arguments", "traceloop.entity.input", "input.value"
        ))
        result = _first(attrs, "gen_ai.tool.call.result", "traceloop.entity.output", "output.value")
        mapped.tool_output = _clip(_text(_json(result))) if result is not None else None
        if attrs.get("gen_ai.tool.call.id"):
            mapped.metadata["tool_call_id"] = attrs["gen_ai.tool.call.id"]
    elif span_type == "agent":
        mapped.agent_name = attrs.get("gen_ai.agent.name") or span.name
        if attrs.get("gen_ai.agent.description"):
            mapped.metadata["description"] = _clip(str(attrs["gen_ai.agent.description"]), MAX_PREVIEW)
    return mapped


_TRACELOOP_KINDS = {"workflow": "chain", "task": "chain", "agent": "agent", "tool": "tool"}


def _map_traceloop(span: OtlpSpan) -> MappedSpan:
    attrs = span.attributes
    kind = str(attrs.get("traceloop.span.kind") or "").lower()
    name = attrs.get("traceloop.entity.name") or span.name
    mapped = MappedSpan(span_type=_TRACELOOP_KINDS.get(kind, "chain"), name=name, convention="traceloop")
    inp = attrs.get("traceloop.entity.input")
    out = attrs.get("traceloop.entity.output")
    if mapped.span_type == "tool":
        mapped.tool_name = name
        mapped.tool_input = _as_dict(inp)
        mapped.tool_output = _clip(_text(_json(out))) if out is not None else None
    else:
        if mapped.span_type == "agent":
            mapped.agent_name = name
        _io_metadata(mapped, inp, out)
    return mapped


# ---------------------------------------------------------------------------
# Common post-processing
# ---------------------------------------------------------------------------

_METADATA_KEYS = (
    ("session.id", "session_id"),
    ("gen_ai.conversation.id", "session_id"),
    ("user.id", "user_id"),
    ("ai.telemetry.functionId", "function_id"),
    ("traceloop.workflow.name", "workflow"),
)
_TAG_PREFIXES = ("ai.telemetry.metadata.", "traceloop.association.properties.", "aitrajectiq.metadata.")
_MAX_TAGS = 30


def span_error(span: OtlpSpan) -> Optional[str]:
    if span.status_code != 2:
        return None
    for event in span.events:
        if event.get("name") == "exception":
            attrs = event.get("attributes") or {}
            text = ": ".join(
                str(v) for v in (attrs.get("exception.type"), attrs.get("exception.message")) if v
            )
            if text:
                return _clip(text, MAX_PREVIEW)
    return span.status_message or "error"


def base_metadata(span: OtlpSpan) -> dict:
    attrs = span.attributes
    meta: dict = {}
    if span.resource.get("service.name"):
        meta["service"] = span.resource["service.name"]
    if span.scope_name:
        meta["instrumentation"] = span.scope_name
    for key, target in _METADATA_KEYS:
        if attrs.get(key) not in (None, "") and target not in meta:
            meta[target] = attrs[key]
    tags = {}
    for key, value in attrs.items():
        if key.startswith(_TAG_PREFIXES) and len(tags) < _MAX_TAGS:
            tags[key.split(".")[-1]] = _clip(_text(value), 200)
    openinference_meta = _json(attrs.get("metadata"))
    if isinstance(openinference_meta, dict):
        for key, value in list(openinference_meta.items())[: _MAX_TAGS - len(tags)]:
            # MongoDB field names can't contain dots or start with "$" on older servers.
            tags[str(key).replace(".", "_").lstrip("$")] = _clip(_text(value), 200)
    if tags:
        meta["tags"] = tags
    return meta


def map_span(span: OtlpSpan) -> Optional[MappedSpan]:
    """Map an OTel span to AItrajectIQ fields, or None if it carries no GenAI/agent signal."""
    attrs = span.attributes
    vercel_op = _vercel_operation(span)
    if vercel_op:
        mapped = _map_vercel(span, vercel_op)
    elif "openinference.span.kind" in attrs:
        mapped = _map_openinference(span)
    else:
        mapped = None
        if _has_prefix(attrs, ("gen_ai.",)) or "llm.request.type" in attrs:
            mapped = _map_genai(span)
        if mapped is None and "traceloop.span.kind" in attrs:
            mapped = _map_traceloop(span)
    if mapped is None:
        return None

    error = span_error(span)
    if error:
        mapped.status = "error"
        mapped.error = error
        if mapped.span_type == "tool" and not mapped.tool_error:
            mapped.tool_error = error
    mapped.metadata = {**base_metadata(span), **mapped.metadata}
    mapped.model = str(mapped.model) if mapped.model not in (None, "") else None
    mapped.response = _clip(mapped.response) if mapped.response else None
    if mapped.messages_in:
        mapped.messages_in = [{**m, "content": _clip(m.get("content") or "")} for m in mapped.messages_in]
    else:
        mapped.messages_in = None
    return mapped
