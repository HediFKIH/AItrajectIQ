"""
End-to-end check of the OpenTelemetry receiver.

Sends a support-agent trace (invoke_agent -> chat -> execute_tool -> chat) with
OpenTelemetry GenAI semantic-convention attributes to the AItrajectIQ backend over
OTLP/HTTP, exactly like an instrumented framework would, then reads it back through
the REST API and checks spans, tokens and cost.

    pip install "opentelemetry-sdk==1.27.0" "opentelemetry-exporter-otlp-proto-http==1.27.0"
    $env:AITRAJECTIQ_API_KEY = "atq_..."          (bash: export AITRAJECTIQ_API_KEY=atq_...)
    python test_app/otel_smoke.py [--base-url http://localhost:8000] [--wait 60]

Evaluation metrics only appear if the backend has an eval LLM configured and the
project enables metrics; the script reports them when they arrive within --wait seconds.
"""

import argparse
import json
import os
import sys
import time
import urllib.request

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor

SYSTEM = "You are a support agent. Use the lookup_order tool to answer questions about orders."
QUESTION = "Where is my order A-42?"
ANSWER = "Your order A-42 has shipped and should arrive tomorrow."

FIRST_CALL = {
    "gen_ai.operation.name": "chat",
    "gen_ai.provider.name": "openai",
    "gen_ai.request.model": "gpt-4o-mini",
    "gen_ai.request.temperature": 0.2,
    "gen_ai.usage.input_tokens": 182,
    "gen_ai.usage.output_tokens": 24,
    "gen_ai.system_instructions": json.dumps([{"type": "text", "content": SYSTEM}]),
    "gen_ai.input.messages": json.dumps([
        {"role": "user", "parts": [{"type": "text", "content": QUESTION}]},
    ]),
    "gen_ai.output.messages": json.dumps([{
        "role": "assistant",
        "parts": [{"type": "tool_call", "id": "call_1", "name": "lookup_order", "arguments": {"order_id": "A-42"}}],
        "finish_reason": "tool_call",
    }]),
}

TOOL_CALL = {
    "gen_ai.operation.name": "execute_tool",
    "gen_ai.tool.name": "lookup_order",
    "gen_ai.tool.call.id": "call_1",
    "gen_ai.tool.call.arguments": json.dumps({"order_id": "A-42"}),
    "gen_ai.tool.call.result": json.dumps({"status": "shipped", "eta": "tomorrow"}),
}

SECOND_CALL = {
    "gen_ai.operation.name": "chat",
    "gen_ai.provider.name": "openai",
    "gen_ai.request.model": "gpt-4o-mini",
    "gen_ai.request.temperature": 0.2,
    "gen_ai.usage.input_tokens": 240,
    "gen_ai.usage.output_tokens": 18,
    "gen_ai.system_instructions": json.dumps([{"type": "text", "content": SYSTEM}]),
    "gen_ai.input.messages": json.dumps([
        {"role": "user", "parts": [{"type": "text", "content": QUESTION}]},
        {"role": "assistant", "parts": [
            {"type": "tool_call", "id": "call_1", "name": "lookup_order", "arguments": {"order_id": "A-42"}},
        ]},
        {"role": "tool", "parts": [
            {"type": "tool_call_response", "id": "call_1", "response": {"status": "shipped", "eta": "tomorrow"}},
        ]},
    ]),
    "gen_ai.output.messages": json.dumps([
        {"role": "assistant", "parts": [{"type": "text", "content": ANSWER}], "finish_reason": "stop"},
    ]),
}


def get_json(url: str, api_key: str) -> dict:
    request = urllib.request.Request(url, headers={"X-API-Key": api_key})
    with urllib.request.urlopen(request, timeout=10) as response:
        return json.loads(response.read().decode("utf-8"))


def emit_trace(base_url: str, api_key: str) -> str:
    provider = TracerProvider(resource=Resource.create({"service.name": "aitrajectiq-otel-smoke"}))
    exporter = OTLPSpanExporter(endpoint=f"{base_url}/otel/v1/traces", headers={"x-api-key": api_key})
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    tracer = provider.get_tracer("aitrajectiq.otel_smoke")

    agent_attributes = {"gen_ai.operation.name": "invoke_agent", "gen_ai.agent.name": "support-agent"}
    with tracer.start_as_current_span("invoke_agent support-agent", attributes=agent_attributes) as root:
        trace_id = format(root.get_span_context().trace_id, "032x")
        with tracer.start_as_current_span("chat gpt-4o-mini", attributes=FIRST_CALL):
            time.sleep(0.05)
        with tracer.start_as_current_span("execute_tool lookup_order", attributes=TOOL_CALL):
            time.sleep(0.02)
        with tracer.start_as_current_span("chat gpt-4o-mini", attributes=SECOND_CALL):
            time.sleep(0.05)
    provider.shutdown()
    return trace_id


def find_trace(base_url: str, api_key: str, project_id: str, otel_trace_id: str, wait_s: float):
    deadline = time.time() + wait_s
    while time.time() < deadline:
        listing = get_json(f"{base_url}/traces/?project_id={project_id}&limit=50", api_key)
        for summary in listing.get("traces", []):
            if summary.get("otel_trace_id") == otel_trace_id and summary.get("status") != "running":
                return summary
        time.sleep(1)
    return None


def trace_checks(summary: dict, detail: dict) -> dict:
    """What the backend must have stored for the trace sent by emit_trace."""
    span_types = sorted(s["span_type"] for s in detail["spans"])
    expected_tokens_in = FIRST_CALL["gen_ai.usage.input_tokens"] + SECOND_CALL["gen_ai.usage.input_tokens"]
    return {
        "status is completed": summary["status"] == "completed",
        "1 agent, 2 llm and 1 tool span": span_types == ["agent", "llm", "llm", "tool"],
        "input tokens summed": summary["total_tokens_in"] == expected_tokens_in,
        "cost computed": summary["total_cost_usd"] > 0,
        "final answer captured": any(ANSWER in (s.get("response") or "") for s in detail["spans"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default=os.environ.get("AITRAJECTIQ_BASE_URL", "http://localhost:8000"))
    parser.add_argument("--api-key", default=os.environ.get("AITRAJECTIQ_API_KEY"))
    parser.add_argument("--wait", type=float, default=60.0, help="seconds to wait for ingestion and metrics")
    args = parser.parse_args()
    if not args.api_key:
        print("Set AITRAJECTIQ_API_KEY or pass --api-key", file=sys.stderr)
        return 2
    base_url = args.base_url.rstrip("/")

    project = get_json(f"{base_url}/auth/whoami", args.api_key)
    print(f"Project: {project['project_name']} ({project['project_id']})")

    otel_trace_id = emit_trace(base_url, args.api_key)
    print(f"Sent OpenTelemetry trace {otel_trace_id}")

    summary = find_trace(base_url, args.api_key, project["project_id"], otel_trace_id, args.wait)
    if summary is None:
        print(f"FAIL: trace {otel_trace_id} was not finalized within {args.wait:.0f}s", file=sys.stderr)
        return 1

    detail = get_json(f"{base_url}/traces/{summary['id']}", args.api_key)
    checks = trace_checks(summary, detail)
    for label, ok in checks.items():
        print(f"  [{'ok' if ok else 'FAIL'}] {label}")
    print(f"Trace {summary['id']}: {summary['total_spans']} spans, "
          f"{summary['total_tokens_in']}+{summary['total_tokens_out']} tokens, ${summary['total_cost_usd']:.6f}")
    if not all(checks.values()):
        return 1

    deadline = time.time() + args.wait
    while time.time() < deadline:
        detail = get_json(f"{base_url}/traces/{summary['id']}", args.api_key)
        span_metrics = [m for s in detail["spans"] for m in s.get("metrics", [])]
        if detail["metrics"] or span_metrics:
            print(f"Metrics: {len(span_metrics)} span-level, {len(detail['metrics'])} trace-level")
            break
        time.sleep(2)
    else:
        print("No metrics yet (needs an eval LLM configured on the backend and metrics enabled on the project).")
    print("OK")
    return 0


if __name__ == "__main__":
    sys.exit(main())
