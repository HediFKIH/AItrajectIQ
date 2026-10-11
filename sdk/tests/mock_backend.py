"""
Mini serveur backend de test pour le SDK AItrajectIQ.
Reçoit les spans et traces, les affiche dans le terminal.

Lancer avec : python tests/mock_backend.py
"""

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from datetime import datetime
import uvicorn
import json

app = FastAPI(title="AItrajectIQ Mock Backend")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

traces = {}
spans = []
span_counter = 0


def _print_box(title, content):
    width = max(len(title) + 4, max((len(l) for l in content.split("\n")), default=0) + 4)
    print(f"\n{'='*width}")
    print(f"  {title}")
    print(f"{'='*width}")
    print(content)
    print(f"{'='*width}\n")


@app.post("/ingest/trace/start")
async def trace_start(request: Request):
    body = await request.json()
    trace_id = f"trace-{len(traces)+1:04d}"
    traces[trace_id] = {
        "id": trace_id,
        "name": body.get("name", "unnamed"),
        "project_id": body.get("project_id"),
        "metadata": body.get("metadata", {}),
        "status": "running",
        "start_time": datetime.now().isoformat(),
        "spans": [],
        "total_cost_usd": 0.0,
    }
    _print_box(
        f"TRACE STARTED: {trace_id}",
        f"  Name:     {body.get('name')}\n"
        f"  Project:  {body.get('project_id')}\n"
        f"  Metadata: {json.dumps(body.get('metadata', {}))}"
    )
    return {"trace_id": trace_id}


@app.post("/ingest/trace/end")
async def trace_end(request: Request):
    body = await request.json()
    trace_id = body.get("trace_id")
    status = body.get("status", "completed")

    trace = traces.get(trace_id, {})
    trace["status"] = status
    trace["end_time"] = datetime.now().isoformat()

    trace_spans = [s for s in spans if s.get("trace_id") == trace_id]
    total_cost = sum(s.get("cost_usd", 0) or 0 for s in trace_spans)
    total_tokens = sum((s.get("tokens_in", 0) or 0) + (s.get("tokens_out", 0) or 0) for s in trace_spans)

    _print_box(
        f"TRACE ENDED: {trace_id} [{status.upper()}]",
        f"  Name:         {trace.get('name')}\n"
        f"  Spans:        {len(trace_spans)}\n"
        f"  Total cost:   ${total_cost:.4f}\n"
        f"  Total tokens: {total_tokens}"
    )

    return {
        "trace_id": trace_id,
        "total_cost_usd": total_cost,
        "span_count": len(trace_spans),
        "duration_ms": 0,
    }


@app.post("/ingest/span")
async def ingest_span(request: Request):
    global span_counter
    body = await request.json()
    span_counter += 1
    span_id = f"span-{span_counter:04d}"
    body["span_id"] = span_id

    PRICES = {
        "gpt-4o": {"in": 0.0025, "out": 0.010},
        "gpt-4o-mini": {"in": 0.00015, "out": 0.0006},
        "gpt-4-turbo": {"in": 0.010, "out": 0.030},
        "claude-3-5-sonnet-20241022": {"in": 0.003, "out": 0.015},
        "claude-3-haiku-20240307": {"in": 0.00025, "out": 0.00125},
        "mistral-large-latest": {"in": 0.002, "out": 0.006},
    }

    cost = 0.0
    if body.get("span_type") == "llm":
        model = body.get("model", "")
        prices = PRICES.get(model, {"in": 0, "out": 0})
        tokens_in = body.get("tokens_in", 0) or 0
        tokens_out = body.get("tokens_out", 0) or 0
        cost = tokens_in * prices["in"] / 1000 + tokens_out * prices["out"] / 1000
        body["cost_usd"] = cost

    spans.append(body)

    if body.get("span_type") == "llm":
        response_preview = (body.get("response") or "")[:100]
        _print_box(
            f"SPAN [{span_id}] LLM — {body.get('provider')} / {body.get('model')}",
            f"  Trace:    {body.get('trace_id')}\n"
            f"  Step:     {body.get('step_index')}\n"
            f"  Latency:  {body.get('latency_ms')}ms\n"
            f"  Tokens:   {body.get('tokens_in')} in / {body.get('tokens_out')} out\n"
            f"  Cost:     ${cost:.4f}\n"
            f"  Response: {response_preview}..."
        )
    else:
        _print_box(
            f"SPAN [{span_id}] TOOL — {body.get('tool_name')}",
            f"  Trace:    {body.get('trace_id')}\n"
            f"  Step:     {body.get('step_index')}\n"
            f"  Latency:  {body.get('latency_ms')}ms\n"
            f"  Input:    {json.dumps(body.get('tool_input', {}))[:100]}\n"
            f"  Output:   {str(body.get('tool_output', ''))[:100]}\n"
            f"  Error:    {body.get('error')}"
        )

    return {"span_id": span_id}


@app.get("/traces")
async def list_traces():
    result = []
    for tid, t in traces.items():
        trace_spans = [s for s in spans if s.get("trace_id") == tid]
        result.append({
            **t,
            "span_count": len(trace_spans),
            "total_cost_usd": sum(s.get("cost_usd", 0) or 0 for s in trace_spans),
        })
    return result


@app.get("/traces/{trace_id}")
async def get_trace(trace_id: str):
    trace = traces.get(trace_id)
    if not trace:
        return {"error": "not found"}
    trace_spans = sorted(
        [s for s in spans if s.get("trace_id") == trace_id],
        key=lambda s: s.get("step_index", 0)
    )
    return {**trace, "spans": trace_spans}


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8090)
    args = parser.parse_args()
    print("\n" + "="*60)
    print(f"  AItrajectIQ Mock Backend")
    print(f"  http://localhost:{args.port}")
    print("  Prêt à recevoir les spans du SDK")
    print("="*60 + "\n")
    uvicorn.run(app, host="0.0.0.0", port=args.port)
