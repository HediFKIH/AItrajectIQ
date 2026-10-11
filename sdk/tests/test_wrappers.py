"""Unit tests of the provider wrappers: no network, spans are captured in memory."""

import asyncio
import contextvars
import json
import time
from types import SimpleNamespace

import pytest

from aitrajectiq import wrappers


@pytest.fixture
def sent(monkeypatch):
    spans = []
    monkeypatch.setattr(wrappers, "send_span", spans.append)
    monkeypatch.setattr(wrappers, "get_or_create_auto_trace", lambda: "trace-1")
    monkeypatch.setattr(wrappers, "get_project_id", lambda: "proj-1")
    return spans


# ── usage helpers ──

def test_openai_usage_keeps_prompt_tokens_and_reads_cached():
    usage = SimpleNamespace(prompt_tokens=1000, completion_tokens=50,
                            prompt_tokens_details=SimpleNamespace(cached_tokens=800))
    assert wrappers._openai_usage(usage) == {"tokens_in": 1000, "tokens_out": 50, "cache_read_tokens": 800}


def test_openai_usage_accepts_dicts_and_missing_usage():
    assert wrappers._openai_usage({"prompt_tokens": 10, "completion_tokens": 2}) == {
        "tokens_in": 10, "tokens_out": 2, "cache_read_tokens": None}
    assert wrappers._openai_usage(None) == {"tokens_in": 0, "tokens_out": 0}


def test_anthropic_usage_adds_cache_tokens_to_input():
    usage = SimpleNamespace(input_tokens=20, output_tokens=5,
                            cache_read_input_tokens=900, cache_creation_input_tokens=100)
    assert wrappers._anthropic_usage(usage) == {
        "tokens_in": 1020, "tokens_out": 5, "cache_read_tokens": 900, "cache_write_tokens": 100}


def test_bedrock_usage_nova_adds_cache_reads():
    usage = wrappers._bedrock_usage({"usage": {"inputTokens": 30, "outputTokens": 7, "cacheReadInputTokenCount": 70}})
    assert usage == {"tokens_in": 100, "tokens_out": 7, "cache_read_tokens": 70, "cache_write_tokens": None}


def test_bedrock_text_across_model_families():
    assert wrappers._bedrock_text({"content": [{"type": "text", "text": "a"}, {"type": "text", "text": "b"}]}) == "ab"
    assert wrappers._bedrock_text({"output": {"message": {"content": [{"text": "nova"}]}}}) == "nova"
    assert wrappers._bedrock_text({"generation": "llama"}) == "llama"
    assert wrappers._bedrock_text({"results": [{"outputText": "titan"}]}) == "titan"


# ── Bedrock ──

class OneShotBody:
    """Like botocore's StreamingBody: the bytes can be read only once."""

    def __init__(self, data: bytes):
        self._data = data

    def read(self, *args):
        data, self._data = self._data, b""
        return data


ANSWER = {
    "content": [{"type": "text", "text": "Bonjour !"}],
    "usage": {"input_tokens": 12, "output_tokens": 3},
}
REQUEST = {
    "anthropic_version": "bedrock-2023-05-31", "max_tokens": 64, "temperature": 0.2,
    "system": "Be brief.", "messages": [{"role": "user", "content": "Salut"}],
}


def test_bedrock_body_stays_readable_by_the_application(sent):
    raw = json.dumps(ANSWER).encode("utf-8")
    resp = wrappers._traced_invoke_model(lambda **kw: {"body": OneShotBody(raw)},
                                         {"modelId": "anthropic.claude", "body": json.dumps(REQUEST)})

    assert json.loads(resp["body"].read()) == ANSWER
    assert len(sent) == 1
    span = sent[0]
    assert span["response"] == "Bonjour !"
    assert (span["tokens_in"], span["tokens_out"]) == (12, 3)
    assert span["temperature"] == 0.2 and span["max_tokens"] == 64
    assert span["messages_in"][0] == {"role": "system", "content": "Be brief."}


def test_bedrock_failure_sends_error_span_and_reraises(sent):
    def failing(**kw):
        raise RuntimeError("throttled")

    with pytest.raises(RuntimeError, match="throttled"):
        wrappers._traced_invoke_model(failing, {"modelId": "anthropic.claude", "body": json.dumps(REQUEST)})

    assert len(sent) == 1
    assert sent[0]["status"] == "failed"
    assert sent[0]["error"] == "RuntimeError: throttled"


def test_wrap_bedrock_client_instruments_once(sent):
    client = SimpleNamespace(invoke_model=lambda **kw: {"body": OneShotBody(json.dumps(ANSWER).encode())})
    wrappers.wrap_bedrock_client(client)
    wrappers.wrap_bedrock_client(client)

    client.invoke_model(modelId="m", body=json.dumps(REQUEST))
    assert len(sent) == 1


# ── OpenAI streaming ──

def chunk(text=None, usage=None):
    choices = [SimpleNamespace(delta=SimpleNamespace(content=text))] if text is not None else []
    return SimpleNamespace(model="gpt-test", usage=usage, choices=choices)


STREAM = [chunk("Hel"), chunk("lo"), chunk(usage=SimpleNamespace(prompt_tokens=9, completion_tokens=2))]
STREAM_KWARGS = {"model": "gpt-test", "messages": [{"role": "user", "content": "hi"}], "stream": True}


def test_stream_yields_every_chunk_and_sends_one_span(sent):
    finish = wrappers._openai_stream_finisher(SimpleNamespace(), STREAM_KWARGS, time.time())
    stream = wrappers._TracedStream(iter(STREAM), finish)

    assert list(stream) == STREAM
    stream.close()
    assert len(sent) == 1
    span = sent[0]
    assert span["response"] == "Hello"
    assert (span["tokens_in"], span["tokens_out"]) == (9, 2)
    assert span["status"] is None


def test_stream_failure_is_recorded(sent):
    def broken():
        yield chunk("partial")
        raise ConnectionError("reset")

    finish = wrappers._openai_stream_finisher(SimpleNamespace(), STREAM_KWARGS, time.time())
    stream = wrappers._TracedStream(broken(), finish)
    with pytest.raises(ConnectionError):
        list(stream)

    assert len(sent) == 1
    assert sent[0]["response"] == "partial"
    assert sent[0]["status"] == "failed"
    assert sent[0]["error"] == "ConnectionError: reset"


def test_stream_closed_early_still_sends_span(sent):
    finish = wrappers._openai_stream_finisher(SimpleNamespace(), STREAM_KWARGS, time.time())
    with wrappers._TracedStream(iter(STREAM), finish) as stream:
        next(stream)

    assert len(sent) == 1
    assert sent[0]["response"] == "Hel"


def test_stream_span_is_sent_in_the_context_of_the_call():
    request = contextvars.ContextVar("request", default=None)
    seen = []
    token = request.set("call-context")
    stream = wrappers._TracedStream(iter(STREAM), lambda *args: seen.append(request.get()))
    request.reset(token)

    list(stream)
    assert seen == ["call-context"]


def test_async_stream_sends_one_span(sent):
    async def source():
        for c in STREAM:
            yield c

    async def consume():
        finish = wrappers._openai_stream_finisher(SimpleNamespace(), STREAM_KWARGS, time.time())
        stream = wrappers._TracedAsyncStream(source(), finish)
        return [c async for c in stream]

    assert asyncio.run(consume()) == STREAM
    assert len(sent) == 1
    assert sent[0]["response"] == "Hello"
