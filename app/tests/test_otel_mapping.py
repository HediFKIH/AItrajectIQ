import json

from otel.decode import OtlpSpan
from otel.mapping import map_span, messages_from_events, normalize_messages, response_text


def make_span(attributes=None, name="span", events=None, status_code=0, status_message=None):
    return OtlpSpan(
        trace_id="a" * 32,
        span_id="b" * 16,
        parent_span_id="c" * 16,
        name=name,
        kind=0,
        start_ns=1_700_000_000_000_000_000,
        end_ns=1_700_000_002_000_000_000,
        attributes=attributes or {},
        events=events or [],
        status_code=status_code,
        status_message=status_message,
        resource={"service.name": "svc"},
        scope_name="test-scope",
    )


# --- OpenTelemetry GenAI semantic conventions ------------------------------

def test_genai_semconv_chat_span():
    mapped = map_span(make_span({
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": "gpt-4o",
        "gen_ai.response.model": "gpt-4o-2024-08-06",
        "gen_ai.usage.input_tokens": 120,
        "gen_ai.usage.output_tokens": 30,
        "gen_ai.request.temperature": 0.3,
        "gen_ai.request.max_tokens": 256,
        "gen_ai.system_instructions": json.dumps([{"type": "text", "content": "You are a weather bot."}]),
        "gen_ai.input.messages": json.dumps([
            {"role": "user", "parts": [{"type": "text", "content": "Weather in Paris?"}]},
        ]),
        "gen_ai.output.messages": json.dumps([{
            "role": "assistant",
            "parts": [
                {"type": "text", "content": "Let me check."},
                {"type": "tool_call", "id": "call_1", "name": "get_weather", "arguments": {"city": "Paris"}},
            ],
            "finish_reason": "tool_call",
        }]),
    }, name="chat gpt-4o"))

    assert (mapped.span_type, mapped.convention) == ("llm", "genai")
    assert (mapped.provider, mapped.model) == ("openai", "gpt-4o-2024-08-06")
    assert (mapped.tokens_in, mapped.tokens_out, mapped.temperature, mapped.max_tokens) == (120, 30, 0.3, 256)
    assert mapped.messages_in == [
        {"role": "system", "content": "You are a weather bot."},
        {"role": "user", "content": "Weather in Paris?"},
    ]
    assert mapped.response == 'Let me check.\n[tool_call] get_weather({"city": "Paris"})'
    assert mapped.metadata["service"] == "svc"
    assert mapped.metadata["instrumentation"] == "test-scope"


def test_genai_span_events_fill_content():
    mapped = map_span(make_span(
        {"gen_ai.operation.name": "chat", "gen_ai.system": "aws.bedrock", "gen_ai.request.model": "claude-sonnet-4"},
        events=[
            {"name": "gen_ai.user.message", "time_ns": 1, "attributes": {"content": json.dumps([{"text": "Hello"}])}},
            {"name": "gen_ai.choice", "time_ns": 2,
             "attributes": {"message": json.dumps([{"text": "Hi there"}]), "finish_reason": "end_turn"}},
        ],
    ))

    assert mapped.messages_in == [{"role": "user", "content": "Hello"}]
    assert mapped.response == "Hi there"
    assert mapped.provider == "aws.bedrock"


def test_semantic_kernel_content_events():
    mapped = map_span(make_span(
        {"gen_ai.operation.name": "chat.completions", "gen_ai.system": "openai", "gen_ai.request.model": "gpt-4o"},
        events=[
            {"name": "gen_ai.content.prompt", "time_ns": 1,
             "attributes": {"gen_ai.prompt": json.dumps([{"role": "user", "content": "2+2?"}])}},
            {"name": "gen_ai.content.completion", "time_ns": 2,
             "attributes": {"gen_ai.completion": json.dumps([{"role": "assistant", "content": "4"}])}},
        ],
    ))

    assert mapped.span_type == "llm"
    assert mapped.messages_in == [{"role": "user", "content": "2+2?"}]
    assert mapped.response == "4"


def test_genai_tool_and_agent_spans():
    tool = map_span(make_span({
        "gen_ai.operation.name": "execute_tool",
        "gen_ai.tool.name": "lookup_order",
        "gen_ai.tool.call.id": "call_7",
        "gen_ai.tool.call.arguments": json.dumps({"order_id": "A-42"}),
        "gen_ai.tool.call.result": json.dumps({"status": "shipped"}),
    }, name="execute_tool lookup_order"))
    assert (tool.span_type, tool.tool_name) == ("tool", "lookup_order")
    assert tool.tool_input == {"order_id": "A-42"}
    assert tool.tool_output == '{"status": "shipped"}'
    assert tool.metadata["tool_call_id"] == "call_7"

    agent = map_span(make_span(
        {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "support-agent"},
        name="invoke_agent support-agent",
    ))
    assert (agent.span_type, agent.agent_name) == ("agent", "support-agent")


# --- OpenLLMetry / Traceloop ------------------------------------------------

def test_openllmetry_indexed_prompt_and_completion():
    mapped = map_span(make_span({
        "llm.request.type": "chat",
        "gen_ai.system": "Anthropic",
        "gen_ai.request.model": "claude-3-5-sonnet",
        "gen_ai.usage.prompt_tokens": 50,
        "gen_ai.usage.completion_tokens": 10,
        "gen_ai.prompt.0.role": "system",
        "gen_ai.prompt.0.content": "Be brief.",
        "gen_ai.prompt.1.role": "user",
        "gen_ai.prompt.1.content": "Capital of France?",
        "gen_ai.completion.0.role": "assistant",
        "gen_ai.completion.0.content": "Paris.",
        "gen_ai.completion.0.tool_calls.0.name": "lookup",
        "gen_ai.completion.0.tool_calls.0.arguments": '{"q": "France"}',
    }, name="anthropic.chat"))

    assert (mapped.span_type, mapped.provider) == ("llm", "anthropic")
    assert (mapped.tokens_in, mapped.tokens_out) == (50, 10)
    assert [m["role"] for m in mapped.messages_in] == ["system", "user"]
    assert mapped.response == 'Paris.\n[tool_call] lookup({"q": "France"})'


def test_traceloop_workflow_and_tool():
    workflow = map_span(make_span({
        "traceloop.span.kind": "workflow",
        "traceloop.entity.name": "support_flow",
        "traceloop.entity.input": json.dumps({"args": ["hi"]}),
    }, name="support_flow.workflow"))
    assert (workflow.span_type, workflow.name) == ("chain", "support_flow")
    assert workflow.metadata["input"] == '{"args": ["hi"]}'

    tool = map_span(make_span({
        "traceloop.span.kind": "tool",
        "traceloop.entity.name": "lookup_order",
        "traceloop.entity.input": json.dumps({"kwargs": {"id": 7}}),
        "traceloop.entity.output": "shipped",
    }, name="lookup_order.tool"))
    assert tool.span_type == "tool"
    assert tool.tool_input == {"kwargs": {"id": 7}}
    assert tool.tool_output == "shipped"


# --- OpenInference ------------------------------------------------------------

def test_openinference_llm_span():
    mapped = map_span(make_span({
        "openinference.span.kind": "LLM",
        "llm.model_name": "gpt-4o-mini",
        "llm.provider": "openai",
        "llm.input_messages.0.message.role": "system",
        "llm.input_messages.0.message.content": "You are helpful.",
        "llm.input_messages.1.message.role": "user",
        "llm.input_messages.1.message.contents.0.message_content.type": "text",
        "llm.input_messages.1.message.contents.0.message_content.text": "Summarize this.",
        "llm.output_messages.0.message.role": "assistant",
        "llm.output_messages.0.message.content": "Summary.",
        "llm.token_count.prompt": 40,
        "llm.token_count.completion": 5,
        "llm.invocation_parameters": json.dumps({"temperature": 0.1, "max_tokens": 100}),
        "llm.cost.total": 0.0002,
        "session.id": "sess-1",
        "metadata": json.dumps({"tenant": "acme"}),
    }, name="ChatOpenAI"))

    assert (mapped.span_type, mapped.convention) == ("llm", "openinference")
    assert mapped.messages_in == [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Summarize this."},
    ]
    assert mapped.response == "Summary."
    assert (mapped.tokens_in, mapped.tokens_out) == (40, 5)
    assert (mapped.temperature, mapped.max_tokens, mapped.cost_usd) == (0.1, 100, 0.0002)
    assert mapped.metadata["session_id"] == "sess-1"
    assert mapped.metadata["tags"] == {"tenant": "acme"}


def test_openinference_tool_error():
    mapped = map_span(make_span(
        {
            "openinference.span.kind": "TOOL",
            "tool.name": "search_docs",
            "input.value": json.dumps({"query": "refund policy"}),
            "output.value": "No results",
        },
        name="search_docs",
        status_code=2,
        events=[{"name": "exception", "time_ns": 5,
                 "attributes": {"exception.type": "TimeoutError", "exception.message": "upstream timed out"}}],
    ))

    assert (mapped.span_type, mapped.tool_name) == ("tool", "search_docs")
    assert mapped.tool_input == {"query": "refund policy"}
    assert mapped.tool_output == "No results"
    assert mapped.status == "error"
    assert mapped.tool_error == mapped.error == "TimeoutError: upstream timed out"


def test_openinference_agent_chain_and_retriever():
    agent = map_span(make_span(
        {"openinference.span.kind": "AGENT", "agent.name": "planner", "input.value": "plan a trip"}, name="Agent",
    ))
    assert (agent.span_type, agent.agent_name) == ("agent", "planner")
    assert agent.metadata["input"] == "plan a trip"

    assert map_span(make_span({"openinference.span.kind": "CHAIN"}, name="RunnableSequence")).span_type == "chain"

    retriever = map_span(make_span({
        "openinference.span.kind": "RETRIEVER",
        "input.value": "refunds",
        "retrieval.documents.0.document.content": "Refunds within 30 days.",
        "retrieval.documents.1.document.content": "Contact support.",
    }, name="VectorStoreRetriever"))
    assert retriever.span_type == "retriever"
    assert retriever.tool_input == {"input": "refunds"}
    assert retriever.tool_output == "Refunds within 30 days.\n---\nContact support."


# --- Vercel AI SDK ------------------------------------------------------------

def test_vercel_ai_sdk_spans():
    inner = map_span(make_span({
        "ai.operationId": "ai.generateText.doGenerate",
        "ai.model.provider": "openai.chat",
        "ai.model.id": "gpt-4o",
        "ai.prompt.messages": json.dumps([
            {"role": "system", "content": "You are a travel agent."},
            {"role": "user", "content": [{"type": "text", "text": "Book Rome"}]},
        ]),
        "ai.response.text": "",
        "ai.response.toolCalls": json.dumps([
            {"toolCallType": "function", "toolCallId": "1", "toolName": "book", "args": '{"city":"Rome"}'},
        ]),
        "ai.usage.promptTokens": 30,
        "ai.usage.completionTokens": 8,
        "gen_ai.request.temperature": 0,
    }, name="ai.generateText.doGenerate"))
    assert (inner.span_type, inner.convention) == ("llm", "vercel_ai")
    assert (inner.provider, inner.model) == ("openai", "gpt-4o")
    assert inner.messages_in[1] == {"role": "user", "content": "Book Rome"}
    assert inner.response == '[tool_call] book({"city": "Rome"})'
    assert (inner.tokens_in, inner.tokens_out, inner.temperature) == (30, 8, 0.0)

    outer = map_span(make_span({
        "ai.operationId": "ai.generateText",
        "ai.model.id": "gpt-4o",
        "ai.usage.promptTokens": 30,
        "ai.prompt": json.dumps({"prompt": "Book Rome"}),
    }, name="ai.generateText"))
    assert outer.span_type == "chain" and outer.tokens_in is None

    tool = map_span(make_span({
        "ai.operationId": "ai.toolCall",
        "ai.toolCall.name": "book",
        "ai.toolCall.args": '{"city":"Rome"}',
        "ai.toolCall.result": '{"ok":true}',
    }, name="ai.toolCall"))
    assert (tool.span_type, tool.tool_name) == ("tool", "book")
    assert tool.tool_input == {"city": "Rome"}
    assert tool.tool_output == '{"ok": true}'


# --- Everything else ----------------------------------------------------------

def test_plain_http_span_is_dropped():
    assert map_span(make_span({"http.request.method": "POST", "url.full": "https://api.example.com"}, name="POST")) is None


def test_response_text_handles_provider_payloads():
    assert response_text({"choices": [{"message": {"role": "assistant", "content": "A"}}]}) == "A"
    assert response_text({"candidates": [{"content": {"role": "model", "parts": [{"text": "B"}]}}]}) == "B"
    assert response_text({"role": "assistant", "content": [{"type": "text", "text": "C"}]}) == "C"
    assert normalize_messages({"system": "S", "messages": [{"role": "user", "content": "U"}]}) == [
        {"role": "system", "content": "S"},
        {"role": "user", "content": "U"},
    ]


def test_messages_from_log_events_are_time_ordered():
    events = [
        {"name": "gen_ai.choice", "time_ns": 3, "attributes": {"event.name": "gen_ai.choice"},
         "body": {"index": 0, "finish_reason": "stop", "message": {"role": "assistant", "content": "Done"}}},
        {"name": "gen_ai.user.message", "time_ns": 1, "attributes": {}, "body": {"content": "Do it"}},
        {"name": "gen_ai.assistant.message", "time_ns": 2, "attributes": {},
         "body": {"tool_calls": [{"id": "c1", "type": "function", "function": {"name": "run", "arguments": "{}"}}]}},
    ]

    messages, response = messages_from_events(events)

    assert messages == [
        {"role": "user", "content": "Do it"},
        {"role": "assistant", "content": "[tool_call] run({})"},
    ]
    assert response == "Done"
