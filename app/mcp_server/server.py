"""
MCP server: lets Cursor, Claude Code, Claude Desktop and any MCP client query
AItrajectIQ traces, costs and evaluation metrics.

Served by the backend over Streamable HTTP at /mcp (stateless, JSON responses).
Every request must carry a project API key with the `read` scope, as
`Authorization: Bearer atq_...` or `X-API-Key: atq_...`. Tools take no project
argument: they only ever see the project of that key.

Needs the `mcp` package (1.8 or later). Without it the endpoint is disabled and the
rest of the backend runs normally.
"""

import contextvars
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from typing import Any, Optional

from starlette.datastructures import Headers
from starlette.responses import JSONResponse

from models.project import Project
from models.trace import TraceStatus
from security.api_keys import READ, ApiKeyInfo, extract_api_key, has_scope, resolve_api_key
from services import insights, recommendations

log = logging.getLogger("aitrajectiq.mcp")

MCP_PATH = "/mcp"

INSTRUCTIONS = (
    "AItrajectIQ observability data for one project (the project of your API key): "
    "traces of agent runs, their LLM / tool / agent spans, costs, and evaluation metrics "
    "such as faithfulness, hallucination, answer relevance, task completion and loop detection. "
    "Start with get_project_overview, use list_traces, get_costs, get_metrics_summary and "
    "find_problem_spans to locate issues, then get_trace to inspect one run step by step. "
    "get_recommendations suggests cheaper models, prompt caching, and prompt or agent fixes; "
    "get_trace_recommendations does the same step by step for one trace."
)

_current_key: contextvars.ContextVar = contextvars.ContextVar("aitrajectiq_mcp_key", default=None)


def _value(item: Any) -> Any:
    return getattr(item, "value", item)


def _clamp(value: int, low: int, high: int) -> int:
    return max(low, min(high, int(value)))


async def _key_from_context(ctx: Any) -> Optional[ApiKeyInfo]:
    info = _current_key.get()
    if info is not None:
        return info
    try:
        request = ctx.request_context.request
    except Exception:
        request = None
    raw_key = extract_api_key(request.headers) if request is not None else None
    return await resolve_api_key(raw_key)


async def _project(ctx: Any) -> Project:
    info = await _key_from_context(ctx)
    if info is None or not has_scope(info, READ):
        raise PermissionError("A valid AItrajectIQ API key with the 'read' scope is required")
    project = await Project.get(info.project_id)
    if project is None:
        raise PermissionError("The project of this API key no longer exists")
    return project


def _since(hours: float) -> datetime:
    return datetime.utcnow() - timedelta(hours=hours)


def _build_server():
    from mcp.server.fastmcp import Context, FastMCP

    options: dict = dict(
        name="AItrajectIQ",
        instructions=INSTRUCTIONS,
        stateless_http=True,
        json_response=True,
    )
    try:
        from mcp.server.transport_security import TransportSecuritySettings
    except ImportError:
        pass
    else:
        # Every request must present an API key header, which a DNS-rebinding page
        # cannot supply; host checks would only break access through other hostnames.
        options["transport_security"] = TransportSecuritySettings(enable_dns_rebinding_protection=False)

    server = FastMCP(**options)

    @server.tool()
    async def get_project_overview(ctx: Context, days: int = 7) -> dict[str, Any]:
        """Summary of the project over the last `days` days: number of traces (failed, still running),
        LLM and tool calls, tokens, cost in USD, average trace duration and spans in error,
        plus the evaluation metrics enabled on the project."""
        project = await _project(ctx)
        days = _clamp(days, 1, 365)
        return {
            "project": {
                "id": str(project.id),
                "name": project.name,
                "type": _value(project.type),
                "task": _value(project.task),
                "enabled_metrics": [_value(m) for m in project.enabled_metrics],
            },
            "period_days": days,
            "totals": await insights.period_totals(project.id, _since(days * 24)),
        }

    @server.tool()
    async def list_traces(
        ctx: Context,
        status: Optional[str] = None,
        since_hours: int = 24,
        limit: int = 20,
        origin: Optional[str] = None,
    ) -> dict[str, Any]:
        """Most recent traces (agent runs), newest first, with status, origin, span counts, tokens,
        cost, duration and trace-level metrics. `status` filters on running, completed, failed or
        timeout; `origin` on cursor, claude-code, sdk or otel."""
        project = await _project(ctx)
        statuses = {s.value for s in TraceStatus}
        if status is not None and status not in statuses:
            raise ValueError(f"status must be one of {sorted(statuses)}")
        if origin is not None and origin not in insights.ORIGINS:
            raise ValueError(f"origin must be one of {list(insights.ORIGINS)}")
        return await insights.recent_traces(
            project.id, _since(_clamp(since_hours, 1, 24 * 365)), status, _clamp(limit, 1, 100), origin
        )

    @server.tool()
    async def get_trace(ctx: Context, trace_id: str, include_content: bool = False) -> dict[str, Any]:
        """One trace step by step: summary, trace metrics and every span (agent, LLM, tool,
        retriever...) with its depth in the call tree, latency, model, tokens, cost, errors and
        span metrics. `trace_id` is the AItrajectIQ trace id or the 32-character OpenTelemetry
        trace id. Set `include_content` to get longer prompts, responses and tool outputs."""
        project = await _project(ctx)
        trace = await insights.find_project_trace(project.id, trace_id)
        if trace is None:
            raise ValueError(f"Trace {trace_id!r} not found in project {project.name!r}")
        return await insights.trace_detail(trace, include_content)

    @server.tool()
    async def get_trace_recommendations(ctx: Context, trace_id: str) -> dict[str, Any]:
        """What could have been done better in one trace, step by step: LLM calls that a cheaper
        model could have handled (with the saving in USD), context resent without prompt caching,
        a step concentrating the cost, context growing over an agent run, tool calls repeated
        with the same arguments, tool and model errors, and the trace's poor metric verdicts with
        the judge's reason. Also gives the run's complexity (simple / moyenne / complexe), its
        actual and optimized cost, and notes per span id."""
        project = await _project(ctx)
        trace = await insights.find_project_trace(project.id, trace_id)
        if trace is None:
            raise ValueError(f"Trace {trace_id!r} not found in project {project.name!r}")
        return await recommendations.trace_recommendations(trace)

    @server.tool()
    async def get_costs(ctx: Context, days: int = 7, group_by: str = "model") -> dict[str, Any]:
        """LLM cost in USD, number of calls and tokens over the last `days` days,
        grouped by `model`, `provider` or `day`."""
        project = await _project(ctx)
        return await insights.cost_breakdown(project.id, _since(_clamp(days, 1, 365) * 24), group_by)

    @server.tool()
    async def get_metrics_summary(ctx: Context, days: int = 7) -> dict[str, Any]:
        """Average of every evaluation metric over the last `days` days: span level (e.g.
        general_faithfulness, hallucination, answer_relevance, moderation) and trace level
        (e.g. task_completion, claims_grounded, tool_error_rate), with cost totals."""
        project = await _project(ctx)
        return await insights.metrics_summary(project.id, _clamp(days, 1, 365))

    @server.tool()
    async def find_problem_spans(
        ctx: Context,
        kind: str = "errors",
        metric: Optional[str] = None,
        threshold: Optional[float] = None,
        since_hours: int = 168,
        limit: int = 20,
    ) -> dict[str, Any]:
        """Spans that need attention, newest (or slowest) first.
        kind="errors": failed LLM, tool or agent spans.
        kind="slow": spans slower than `threshold` milliseconds (default 10000).
        kind="low_score": spans with a poor evaluation `metric` (default general_faithfulness):
        below `threshold` (default 0.5), above it for moderation, hallucination with findings,
        or false for boolean metrics such as is_json."""
        project = await _project(ctx)
        return await insights.problem_spans(
            project.id,
            kind,
            _since(_clamp(since_hours, 1, 24 * 365)),
            metric=metric,
            threshold=threshold,
            limit=_clamp(limit, 1, 100),
        )

    @server.tool()
    async def list_logs(ctx: Context, limit: int = 20, blocked_only: bool = False) -> dict[str, Any]:
        """Most recent conversational logs: prompt, output, model, latency, guardrail flags
        (prompt-injection block, PII) and metrics. `blocked_only` keeps only blocked prompts."""
        project = await _project(ctx)
        return await insights.recent_logs(project.id, _clamp(limit, 1, 100), blocked_only)

    @server.tool()
    async def get_recommendations(ctx: Context, days: int = 7) -> dict[str, Any]:
        """Recommendations from the last `days` days of traffic, most important first: simple
        calls that a cheaper model of the same provider could handle (with the saving in USD),
        complex calls failing on a small model, a long system prompt resent without prompt
        caching, prompt fixes for poor span metrics (hallucination, relevance, JSON format...),
        agent fixes for poor trace metrics (loops, tool errors, unfinished tasks), and models
        whose calls fail often. Also returns calls and cost per task complexity
        (simple / moyenne / complexe) and per model."""
        project = await _project(ctx)
        return await recommendations.recommendations(project.id, _since(_clamp(days, 1, 365) * 24))

    return server


class _McpAsgiApp:
    """Checks the API key, then hands the request to FastMCP's Streamable HTTP session manager.

    Calling the session manager directly (instead of FastMCP's Starlette sub-app) keeps
    /mcp and /mcp/ working without redirects across `mcp` versions.
    """

    def __init__(self, session_manager):
        self.session_manager = session_manager

    async def __call__(self, scope, receive, send):
        info = await resolve_api_key(extract_api_key(Headers(scope=scope)))
        if info is None:
            response = JSONResponse(
                {"error": "unauthorized", "detail": "Missing or invalid AItrajectIQ API key"},
                status_code=401,
                headers={"WWW-Authenticate": 'Bearer realm="aitrajectiq"'},
            )
            await response(scope, receive, send)
            return
        if not has_scope(info, READ):
            response = JSONResponse(
                {"error": "forbidden", "detail": "This API key does not have the 'read' scope"},
                status_code=403,
            )
            await response(scope, receive, send)
            return
        token = _current_key.set(info)
        try:
            await self.session_manager.handle_request(scope, receive, send)
        finally:
            _current_key.reset(token)


def mount_mcp(app) -> Optional[Any]:
    """Add the /mcp endpoint to a FastAPI app. Returns the FastMCP server, or None when
    the `mcp` package is missing or too old."""
    try:
        server = _build_server()
        server.streamable_http_app()   # creates the session manager on first call
        asgi = _McpAsgiApp(server.session_manager)
    except (ImportError, AttributeError, RuntimeError) as exc:
        log.warning("MCP endpoint disabled (%s). Install it with: pip install 'mcp>=1.8'", exc)
        return None
    app.add_route(MCP_PATH, asgi, include_in_schema=False)
    app.add_route(MCP_PATH + "/", asgi, include_in_schema=False)
    log.info("MCP endpoint enabled at %s", MCP_PATH)
    return server


@asynccontextmanager
async def mcp_lifespan(server: Optional[Any]):
    """Run the FastMCP session manager for the lifetime of the FastAPI app."""
    if server is None:
        yield
        return
    async with server.session_manager.run():
        yield
