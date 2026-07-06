"""
LLMInsight Backend v2 — lightweight entry point.
Only loads: DB + ingest/traces/metrics routes + eval LLM + project/user CRUD.
No heavy deps (no comet, no faiss, no sentence-transformers, no llm_guard).
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager
from dotenv import load_dotenv
import os
import logging

from config.db import init_db
from evaluation.runner import configure_eval_llm, EvalLLMConfig
from evaluation.web_search import configure_search, SearchConfig

logging.basicConfig(level=logging.INFO)
load_dotenv(dotenv_path="../.env")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    # Eval LLM
    #eval_provider = os.getenv("EVAL_LLM_PROVIDER", "openai")
    #eval_api_key = os.getenv("EVAL_LLM_API_KEY", "")
    #eval_model = os.getenv("EVAL_LLM_MODEL", "gpt-4o")
    eval_provider = os.getenv("EVAL_LLM_PROVIDER", "azure")
    eval_api_key = os.getenv("LLMINSIGHT_API_KEY", "")    
    eval_model = os.getenv("DEPLOYMENT_NAME", "gpt-4o-mini")
    azure_endpoint = os.getenv("AZURE_ENDPOINT", "")
    deployment_name = os.getenv("DEPLOYMENT_NAME", "")
    api_version = os.getenv("EVAL_LLM_AZURE_API_VERSION", "2025-01-01-preview")

    if eval_api_key:
        configure_eval_llm(EvalLLMConfig(
            provider=eval_provider,
            model=eval_model,
            api_key=eval_api_key,
            base_url=os.getenv("EVAL_LLM_BASE_URL"),
            #azure_endpoint=os.getenv("EVAL_LLM_AZURE_ENDPOINT"),
            #azure_api_version=os.getenv("EVAL_LLM_AZURE_API_VERSION", "2024-12-01-preview"),
            azure_endpoint=azure_endpoint,
            azure_api_version=api_version,
        ))
        logging.info("Eval LLM: %s / %s", eval_provider, eval_model)

    # Web search fact-check
    fact_provider = os.getenv("FACT_CHECK_PROVIDER", "none")
    fact_api_key = os.getenv("FACT_CHECK_API_KEY", "")
    configure_search(SearchConfig(provider=fact_provider, api_key=fact_api_key))

    yield


app = FastAPI(title="LLMInsight", version="2.0", lifespan=lifespan)
origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",  
    "https://localhost:5173", 
    "http://localhost:8081",
    "http://127.0.0.1:8081",
]
app.add_middleware(
    CORSMiddleware,
    #allow_origins=["*"],
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# CRUD routes
from routes.user_routes import router as user_router
from routes.project_routes import router as project_router

# New observability routes
from routes.ingest import router as ingest_router
from routes.traces import router as traces_router
from routes.metrics import router as metrics_router
from routes.log_routes import router as log_router
from routes.record_routes import router as record_routes
from routes.help_route import router as help_router

app.include_router(user_router)
app.include_router(project_router)
app.include_router(ingest_router)
app.include_router(traces_router)
app.include_router(metrics_router)
app.include_router(log_router)
app.include_router(record_routes)
app.include_router(help_router)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
