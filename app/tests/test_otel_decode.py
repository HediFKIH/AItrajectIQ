import base64
import gzip
import json

import pytest

from otel.decode import (
    JSON,
    PROTOBUF,
    OtlpDecodeError,
    UnsupportedMediaType,
    decode_logs,
    decode_traces,
    encode_export_response,
    normalize_id,
)

TRACE_ID = "5b8efff798038103d269b633813fc60c"
SPAN_ID = "eee19b7ec3c1b174"
PARENT_ID = "eee19b7ec3c1b173"


def _trace_payload():
    return {
        "resourceSpans": [{
            "resource": {"attributes": [{"key": "service.name", "value": {"stringValue": "support-bot"}}]},
            "scopeSpans": [{
                "scope": {"name": "openinference.instrumentation.openai"},
                "spans": [{
                    "traceId": TRACE_ID,
                    "spanId": SPAN_ID,
                    "parentSpanId": PARENT_ID,
                    "name": "chat gpt-4o-mini",
                    "kind": 3,
                    "startTimeUnixNano": "1700000000000000000",
                    "endTimeUnixNano": "1700000001500000000",
                    "attributes": [
                        {"key": "gen_ai.operation.name", "value": {"stringValue": "chat"}},
                        {"key": "gen_ai.usage.input_tokens", "value": {"intValue": "12"}},
                        {"key": "gen_ai.request.temperature", "value": {"doubleValue": 0.2}},
                        {"key": "flag", "value": {"boolValue": True}},
                        {"key": "list", "value": {"arrayValue": {"values": [{"stringValue": "a"}, {"intValue": 2}]}}},
                        {"key": "map", "value": {"kvlistValue": {"values": [{"key": "k", "value": {"stringValue": "v"}}]}}},
                    ],
                    "events": [{
                        "name": "exception",
                        "timeUnixNano": "1700000001000000000",
                        "attributes": [{"key": "exception.message", "value": {"stringValue": "boom"}}],
                    }],
                    "status": {"code": 2, "message": "failed"},
                }],
            }],
        }]
    }


def test_decode_json_spans():
    spans, rejected, media_type = decode_traces(json.dumps(_trace_payload()).encode(), "application/json", None)

    assert media_type == JSON and rejected == 0 and len(spans) == 1
    span = spans[0]
    assert (span.trace_id, span.span_id, span.parent_span_id) == (TRACE_ID, SPAN_ID, PARENT_ID)
    assert span.start_ns == 1_700_000_000_000_000_000
    assert span.end_ns - span.start_ns == 1_500_000_000
    assert span.attributes["gen_ai.usage.input_tokens"] == 12
    assert span.attributes["gen_ai.request.temperature"] == 0.2
    assert span.attributes["flag"] is True
    assert span.attributes["list"] == ["a", 2]
    assert span.attributes["map"] == {"k": "v"}
    assert (span.status_code, span.status_message) == (2, "failed")
    assert span.resource["service.name"] == "support-bot"
    assert span.scope_name == "openinference.instrumentation.openai"
    assert span.events[0]["attributes"]["exception.message"] == "boom"
    assert not span.is_root


def test_decode_gzip_with_sniffed_content_type():
    body = gzip.compress(json.dumps(_trace_payload()).encode())

    spans, _, media_type = decode_traces(body, None, "gzip")

    assert media_type == JSON and len(spans) == 1


def test_spans_without_ids_are_rejected():
    payload = _trace_payload()
    payload["resourceSpans"][0]["scopeSpans"][0]["spans"][0]["traceId"] = ""

    spans, rejected, _ = decode_traces(json.dumps(payload).encode(), "application/json", None)

    assert spans == [] and rejected == 1


def test_normalize_id_accepts_hex_base64_and_bytes():
    raw = bytes.fromhex(TRACE_ID)
    assert normalize_id(TRACE_ID.upper()) == TRACE_ID
    assert normalize_id(base64.b64encode(raw).decode()) == TRACE_ID
    assert normalize_id(raw) == TRACE_ID
    assert normalize_id("0" * 16) is None
    assert normalize_id("") is None


def test_invalid_bodies():
    with pytest.raises(OtlpDecodeError):
        decode_traces(b"{not json", "application/json", None)
    with pytest.raises(UnsupportedMediaType):
        decode_traces(b"{}", "text/plain", None)
    with pytest.raises(UnsupportedMediaType):
        decode_traces(b"{}", "application/json", "br")


def test_decode_logs_reads_event_name_and_body():
    payload = {"resourceLogs": [{"scopeLogs": [{"logRecords": [{
        "timeUnixNano": "1700000000500000000",
        "traceId": TRACE_ID,
        "spanId": SPAN_ID,
        "body": {"kvlistValue": {"values": [{"key": "content", "value": {"stringValue": "Hi"}}]}},
        "attributes": [{"key": "event.name", "value": {"stringValue": "gen_ai.user.message"}}],
    }]}]}]}

    records, media_type = decode_logs(json.dumps(payload).encode(), "application/json", None)

    assert media_type == JSON
    record = records[0]
    assert record.event_name == "gen_ai.user.message"
    assert record.body == {"content": "Hi"}
    assert (record.trace_id, record.span_id) == (TRACE_ID, SPAN_ID)


def test_export_responses():
    assert encode_export_response("traces", JSON) == b"{}"
    body = json.loads(encode_export_response("traces", JSON, rejected=2, error_message="bad"))
    assert body == {"partialSuccess": {"rejectedSpans": 2, "errorMessage": "bad"}}
    assert encode_export_response("metrics", PROTOBUF) == b""


def test_decode_protobuf_like_json():
    service = pytest.importorskip("opentelemetry.proto.collector.trace.v1.trace_service_pb2")
    common = pytest.importorskip("opentelemetry.proto.common.v1.common_pb2")
    resource = pytest.importorskip("opentelemetry.proto.resource.v1.resource_pb2")
    trace = pytest.importorskip("opentelemetry.proto.trace.v1.trace_pb2")

    span = trace.Span(
        trace_id=bytes.fromhex(TRACE_ID),
        span_id=bytes.fromhex(SPAN_ID),
        name="chat gpt-4o-mini",
        start_time_unix_nano=1_700_000_000_000_000_000,
        end_time_unix_nano=1_700_000_001_500_000_000,
        attributes=[common.KeyValue(key="gen_ai.usage.input_tokens", value=common.AnyValue(int_value=12))],
        status=trace.Status(code=trace.Status.STATUS_CODE_ERROR, message="failed"),
    )
    request = service.ExportTraceServiceRequest(resource_spans=[trace.ResourceSpans(
        resource=resource.Resource(attributes=[
            common.KeyValue(key="service.name", value=common.AnyValue(string_value="support-bot")),
        ]),
        scope_spans=[trace.ScopeSpans(scope=common.InstrumentationScope(name="test"), spans=[span])],
    )])

    spans, rejected, media_type = decode_traces(request.SerializeToString(), "application/x-protobuf", None)

    assert media_type == PROTOBUF and rejected == 0
    decoded = spans[0]
    assert (decoded.trace_id, decoded.span_id, decoded.parent_span_id) == (TRACE_ID, SPAN_ID, None)
    assert decoded.is_root
    assert decoded.attributes["gen_ai.usage.input_tokens"] == 12
    assert decoded.status_code == 2
    assert decoded.end_ns - decoded.start_ns == 1_500_000_000
    assert decoded.resource["service.name"] == "support-bot"
    assert encode_export_response("traces", PROTOBUF, rejected=1, error_message="x") != b""
