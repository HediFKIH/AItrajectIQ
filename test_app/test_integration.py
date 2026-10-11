"""
End-to-end tests of the OpenTelemetry receiver and the MCP server, built on the smoke
scripts (otel_smoke.py, mcp_smoke.py) so both run the same checks.
"""

import asyncio
import os

import pytest

pytestmark = pytest.mark.integration

WAIT_S = float(os.environ.get("AITRAJECTIQ_TEST_WAIT", "60"))


def test_otel_trace_is_ingested_and_finalized(base_url, api_key, project):
    pytest.importorskip("opentelemetry.exporter.otlp.proto.http.trace_exporter")
    import otel_smoke

    otel_trace_id = otel_smoke.emit_trace(base_url, api_key)
    summary = otel_smoke.find_trace(base_url, api_key, project["project_id"], otel_trace_id, WAIT_S)
    assert summary is not None, f"trace {otel_trace_id} was not finalized within {WAIT_S:.0f}s"

    detail = otel_smoke.get_json(f"{base_url}/traces/{summary['id']}", api_key)
    failed = [label for label, ok in otel_smoke.trace_checks(summary, detail).items() if not ok]
    assert not failed, f"trace {summary['id']}: {failed}"


def test_mcp_tools_answer(base_url, api_key, project):
    pytest.importorskip("mcp")
    import mcp_smoke

    url = os.environ.get("AITRAJECTIQ_MCP_URL", f"{base_url}/mcp")
    _, tool_names, results = asyncio.run(mcp_smoke.call_tools(url, api_key))

    missing = {name for name, _ in mcp_smoke.CALLS} - set(tool_names)
    assert not missing, f"tools not exposed: {sorted(missing)}"
    errors = {name: text[:300] for name, _, is_error, text in results if is_error}
    assert not errors, f"tool calls failed: {errors}"
    assert all(text for _, _, _, text in results), "a tool returned no text"
