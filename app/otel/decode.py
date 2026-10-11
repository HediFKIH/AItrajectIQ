"""
OTLP/HTTP decoding.

Turns an export request body (protobuf or JSON, optionally gzip/deflate-compressed)
into flat `OtlpSpan` / `OtlpLogRecord` objects: attribute lists become plain dicts,
trace and span ids become lowercase hex. Both encodings go through the same parser:
protobuf messages are first converted with `MessageToDict`, which produces the OTLP/JSON
field names, except that it encodes ids in base64 where OTLP/JSON uses hex.

Protobuf needs the `opentelemetry-proto` package. JSON works without it.
"""

import base64
import gzip
import importlib
import json
import re
import zlib
from dataclasses import dataclass
from typing import Any, Optional

PROTOBUF = "application/x-protobuf"
JSON = "application/json"

_HEX_ID = re.compile(r"^(?:[0-9a-fA-F]{16}|[0-9a-fA-F]{32})$")

# signal -> (protobuf module, request class, response class, proto field, JSON field)
_SIGNALS = {
    "traces": (
        "opentelemetry.proto.collector.trace.v1.trace_service_pb2",
        "ExportTraceServiceRequest", "ExportTraceServiceResponse",
        "rejected_spans", "rejectedSpans",
    ),
    "logs": (
        "opentelemetry.proto.collector.logs.v1.logs_service_pb2",
        "ExportLogsServiceRequest", "ExportLogsServiceResponse",
        "rejected_log_records", "rejectedLogRecords",
    ),
    "metrics": (
        "opentelemetry.proto.collector.metrics.v1.metrics_service_pb2",
        "ExportMetricsServiceRequest", "ExportMetricsServiceResponse",
        "rejected_data_points", "rejectedDataPoints",
    ),
}


class OtlpDecodeError(ValueError):
    """The body could not be decoded (HTTP 400)."""


class UnsupportedMediaType(OtlpDecodeError):
    """Content type or encoding the receiver can't handle (HTTP 415)."""


@dataclass
class OtlpSpan:
    trace_id: str
    span_id: str
    parent_span_id: Optional[str]
    name: str
    kind: int
    start_ns: int
    end_ns: int
    attributes: dict
    events: list            # [{"name": str, "time_ns": int, "attributes": dict}]
    status_code: int        # 0 unset, 1 ok, 2 error
    status_message: Optional[str]
    resource: dict
    scope_name: Optional[str]

    @property
    def is_root(self) -> bool:
        return not self.parent_span_id


@dataclass
class OtlpLogRecord:
    trace_id: Optional[str]
    span_id: Optional[str]
    time_ns: int
    event_name: Optional[str]
    body: Any
    attributes: dict
    resource: dict


# ---------------------------------------------------------------------------
# Value helpers
# ---------------------------------------------------------------------------

def _int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def any_value(value: Any) -> Any:
    """Decode an OTLP AnyValue (`{"stringValue": ...}`, `{"arrayValue": ...}`, ...)."""
    if not isinstance(value, dict):
        return value
    if "stringValue" in value:
        return value["stringValue"]
    if "boolValue" in value:
        return bool(value["boolValue"])
    if "intValue" in value:
        return _int(value["intValue"], value["intValue"])
    if "doubleValue" in value:
        try:
            return float(value["doubleValue"])
        except (TypeError, ValueError):
            return value["doubleValue"]
    if "arrayValue" in value:
        return [any_value(v) for v in (value["arrayValue"] or {}).get("values") or []]
    if "kvlistValue" in value:
        return key_values((value["kvlistValue"] or {}).get("values"))
    if "bytesValue" in value:
        return value["bytesValue"]
    return None


def key_values(items: Any) -> dict:
    """Decode a list of OTLP KeyValue into a dict."""
    out = {}
    for kv in items or []:
        if isinstance(kv, dict) and "key" in kv:
            out[kv["key"]] = any_value(kv.get("value") or {})
    return out


def normalize_id(value: Any) -> Optional[str]:
    """Trace/span id as lowercase hex, from hex (OTLP/JSON), base64 (MessageToDict) or bytes.

    All-zero ids mean "no id". Hex and base64 can't be confused: 8/16-byte ids are
    16/32 characters in hex but 12/24 in base64.
    """
    if not value:
        return None
    if isinstance(value, (bytes, bytearray)):
        raw = bytes(value)
    else:
        text = str(value).strip()
        if _HEX_ID.match(text):
            text = text.lower()
            return None if set(text) == {"0"} else text
        try:
            raw = base64.b64decode(text, validate=True)
        except ValueError:
            return text.lower() or None
    if not raw or not any(raw):
        return None
    return raw.hex()


def _status_code(status: Any) -> int:
    code = (status or {}).get("code", 0) if isinstance(status, dict) else 0
    if isinstance(code, str):
        upper = code.strip().upper()
        if upper.isdigit():
            return int(upper)
        if "ERROR" in upper:
            return 2
        return 1 if "OK" in upper else 0
    return _int(code)


# ---------------------------------------------------------------------------
# Body handling
# ---------------------------------------------------------------------------

def _decompress(body: bytes, content_encoding: Optional[str]) -> bytes:
    encoding = (content_encoding or "").strip().lower()
    try:
        if encoding == "gzip" or (not encoding and body[:2] == b"\x1f\x8b"):
            return gzip.decompress(body)
        if encoding == "deflate":
            try:
                return zlib.decompress(body)
            except zlib.error:
                return zlib.decompress(body, -zlib.MAX_WBITS)
    except (OSError, EOFError, zlib.error) as e:
        raise OtlpDecodeError(f"Could not decompress the {encoding or 'gzip'} body: {e}")
    if encoding in ("", "identity"):
        return body
    raise UnsupportedMediaType(f"Unsupported Content-Encoding {encoding!r} (use gzip or none)")


def _media_type(content_type: Optional[str], body: bytes) -> str:
    ct = (content_type or "").split(";")[0].strip().lower()
    if ct in ("application/x-protobuf", "application/protobuf", "application/octet-stream"):
        return PROTOBUF
    if ct == "application/json" or ct.endswith("+json"):
        return JSON
    if not ct:
        return JSON if body.lstrip()[:1] == b"{" else PROTOBUF
    raise UnsupportedMediaType(
        f"Unsupported Content-Type {ct!r}: send application/x-protobuf or application/json"
    )


def _proto_class(signal: str, attr_index: int):
    module_name = _SIGNALS[signal][0]
    try:
        module = importlib.import_module(module_name)
    except ImportError:
        raise UnsupportedMediaType(
            "OTLP protobuf needs the opentelemetry-proto package on the AItrajectIQ backend "
            "(pip install opentelemetry-proto). Alternatively export with "
            "OTEL_EXPORTER_OTLP_PROTOCOL=http/json."
        )
    return getattr(module, _SIGNALS[signal][attr_index])


def _load(signal: str, body: bytes, media_type: str) -> dict:
    if media_type == JSON:
        try:
            payload = json.loads(body.decode("utf-8") or "{}")
        except (UnicodeDecodeError, json.JSONDecodeError) as e:
            raise OtlpDecodeError(f"Invalid OTLP/JSON body: {e}")
        if not isinstance(payload, dict):
            raise OtlpDecodeError("Invalid OTLP/JSON body: expected an object")
        return payload

    request_cls = _proto_class(signal, 1)
    from google.protobuf.json_format import MessageToDict
    from google.protobuf.message import DecodeError

    message = request_cls()
    try:
        message.ParseFromString(body)
    except DecodeError as e:
        raise OtlpDecodeError(f"Invalid OTLP protobuf body: {e}")
    return MessageToDict(message, use_integers_for_enums=True)


# ---------------------------------------------------------------------------
# Payload parsing (OTLP/JSON shape)
# ---------------------------------------------------------------------------

def parse_traces(payload: dict) -> tuple:
    """Return (spans, rejected_count) from an ExportTraceServiceRequest dict."""
    spans: list = []
    rejected = 0
    for rs in payload.get("resourceSpans") or []:
        if not isinstance(rs, dict):
            continue
        resource = key_values((rs.get("resource") or {}).get("attributes"))
        for ss in rs.get("scopeSpans") or rs.get("instrumentationLibrarySpans") or []:
            scope = ss.get("scope") or ss.get("instrumentationLibrary") or {}
            for sp in ss.get("spans") or []:
                trace_id = normalize_id(sp.get("traceId"))
                span_id = normalize_id(sp.get("spanId"))
                if not trace_id or not span_id:
                    rejected += 1
                    continue
                status = sp.get("status") or {}
                spans.append(OtlpSpan(
                    trace_id=trace_id,
                    span_id=span_id,
                    parent_span_id=normalize_id(sp.get("parentSpanId")),
                    name=sp.get("name") or "",
                    kind=_int(sp.get("kind")),
                    start_ns=_int(sp.get("startTimeUnixNano")),
                    end_ns=_int(sp.get("endTimeUnixNano")),
                    attributes=key_values(sp.get("attributes")),
                    events=[
                        {
                            "name": ev.get("name") or "",
                            "time_ns": _int(ev.get("timeUnixNano")),
                            "attributes": key_values(ev.get("attributes")),
                        }
                        for ev in sp.get("events") or []
                        if isinstance(ev, dict)
                    ],
                    status_code=_status_code(status),
                    status_message=status.get("message") or None,
                    resource=resource,
                    scope_name=scope.get("name"),
                ))
    return spans, rejected


def parse_logs(payload: dict) -> list:
    """Return the log records of an ExportLogsServiceRequest dict."""
    records: list = []
    for rl in payload.get("resourceLogs") or []:
        if not isinstance(rl, dict):
            continue
        resource = key_values((rl.get("resource") or {}).get("attributes"))
        for sl in rl.get("scopeLogs") or rl.get("instrumentationLibraryLogs") or []:
            for lr in sl.get("logRecords") or []:
                if not isinstance(lr, dict):
                    continue
                attributes = key_values(lr.get("attributes"))
                records.append(OtlpLogRecord(
                    trace_id=normalize_id(lr.get("traceId")),
                    span_id=normalize_id(lr.get("spanId")),
                    time_ns=_int(lr.get("timeUnixNano")) or _int(lr.get("observedTimeUnixNano")),
                    event_name=lr.get("eventName") or attributes.get("event.name"),
                    body=any_value(lr.get("body") or {}),
                    attributes=attributes,
                    resource=resource,
                ))
    return records


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def decode_traces(body: bytes, content_type: Optional[str], content_encoding: Optional[str]) -> tuple:
    """Return (spans, rejected_count, media_type). Raises OtlpDecodeError."""
    data = _decompress(body, content_encoding)
    media_type = _media_type(content_type, data)
    spans, rejected = parse_traces(_load("traces", data, media_type))
    return spans, rejected, media_type


def decode_logs(body: bytes, content_type: Optional[str], content_encoding: Optional[str]) -> tuple:
    """Return (log_records, media_type). Raises OtlpDecodeError."""
    data = _decompress(body, content_encoding)
    media_type = _media_type(content_type, data)
    return parse_logs(_load("logs", data, media_type)), media_type


def response_media_type(content_type: Optional[str]) -> str:
    return PROTOBUF if "protobuf" in (content_type or "").lower() else JSON


def encode_export_response(
    signal: str,
    media_type: str,
    rejected: int = 0,
    error_message: Optional[str] = None,
) -> bytes:
    """Serialize an Export*ServiceResponse in the request's encoding."""
    _, _, _, proto_field, json_field = _SIGNALS[signal]
    if media_type == JSON:
        if not rejected and not error_message:
            return b"{}"
        partial = {json_field: rejected}
        if error_message:
            partial["errorMessage"] = error_message
        return json.dumps({"partialSuccess": partial}).encode("utf-8")

    if not rejected and not error_message:
        return b""   # an empty protobuf message serializes to zero bytes
    try:
        response = _proto_class(signal, 2)()
    except UnsupportedMediaType:
        return b""
    setattr(response.partial_success, proto_field, rejected)
    if error_message:
        response.partial_success.error_message = error_message
    return response.SerializeToString()
