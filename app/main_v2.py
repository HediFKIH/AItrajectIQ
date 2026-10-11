"""
AItrajectIQ Backend v2 — lightweight entry point.
Only loads: DB + ingest/traces/metrics routes + eval LLM + project/user CRUD,
plus project API keys, the OpenTelemetry receiver (/otel/v1/*) and the MCP endpoint (/mcp).
No heavy deps (no comet, no faiss, no sentence-transformers, no llm_guard).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
import logging
from config.db import init_db
from evaluation.runner import configure_eval_llm_from_env
from evaluation.web_search import configure_search, SearchConfig

# CRUD routes
from routes.user_routes import router as user_router
from routes.project_routes import router as project_router

# New observability routes
from routes.ingest import router as ingest_router
from routes.traces import router as traces_router
from routes.metrics import router as metrics_router
from routes.log_routes import router as log_router
from routes.record_routes import router as record_routes
from routes.ingest_logs import router as ingest_logs_router

# API keys, OpenTelemetry receiver, MCP endpoint
from extensions import background_services, install

logging.basicConfig(level=logging.INFO)
load_dotenv(dotenv_path="../.env")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    
    configure_eval_llm_from_env()

    # Web search fact-check
    fact_provider = os.getenv("FACT_CHECK_PROVIDER", "none")
    fact_api_key = os.getenv("FACT_CHECK_API_KEY", "")
    configure_search(SearchConfig(provider=fact_provider, api_key=fact_api_key))

    async with background_services(mcp_server):
        yield


app = FastAPI(title="AItrajectIQ", version="2.0", lifespan=lifespan)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",  
    "https://localhost:5173", 
    "http://localhost:8081",
    "http://127.0.0.1:8081",
]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(user_router)
app.include_router(project_router)
app.include_router(ingest_router)
app.include_router(traces_router)
app.include_router(metrics_router)
app.include_router(log_router)
app.include_router(record_routes)
app.include_router(ingest_logs_router)
mcp_server = install(app)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
