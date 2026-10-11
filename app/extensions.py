"""
API keys, the dashboard API, the OpenTelemetry receiver and the MCP endpoint, for any
FastAPI entry point.

    app = FastAPI(lifespan=lifespan)
    mcp_server = install(app)

    @asynccontextmanager
    async def lifespan(app):
        await init_db()
        async with background_services(mcp_server):
            yield
"""

import asyncio
import contextlib
from typing import Any, Optional

from fastapi import FastAPI

from mcp_server.server import mcp_lifespan, mount_mcp
from otel.ingest import run_sweeper
from otel.routes import router as otel_router
from routes.api_key_routes import router as api_key_router
from routes.dashboard_routes import router as dashboard_router
from routes.integration_routes import router as integration_router
from services import pricing
from services.sdk_traces import run_sweeper as run_sdk_sweeper


def install(app: FastAPI) -> Optional[Any]:
    """Add the API key routes, the dashboard API (/api/v1/*), the setup files
    (/integrations/*), /otel/v1/* and /mcp. Returns the MCP server, or None."""
    app.include_router(api_key_router)
    app.include_router(dashboard_router)
    app.include_router(integration_router)
    app.include_router(otel_router)
    return mount_mcp(app)


@contextlib.asynccontextmanager
async def background_services(mcp_server: Optional[Any]):
    """Run the idle trace sweepers (OpenTelemetry and SDK), the model price sync and the MCP
    session manager. Enter after init_db(): they use the database."""
    pricing.load_cache()
    sweepers = [
        asyncio.create_task(run_sweeper()),
        asyncio.create_task(run_sdk_sweeper()),
        asyncio.create_task(pricing.run_price_sync()),
    ]
    try:
        async with mcp_lifespan(mcp_server):
            yield
    finally:
        for sweeper in sweepers:
            sweeper.cancel()
        for sweeper in sweepers:
            with contextlib.suppress(asyncio.CancelledError):
                await sweeper
