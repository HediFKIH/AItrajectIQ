from typing import Any, Generator
from fastapi import FastAPI
from config.db import init_db
from contextlib import asynccontextmanager
from routes import router
from proxy.router import router as proxy_router
from resources.routes import router as resources_router
from FactChecker.test_router import router as retriever_router
import faiss
from sentence_transformers import SentenceTransformer
from openai import OpenAI
from llm_guard.input_scanners.prompt_injection import PromptInjection, MatchType
from fastapi.middleware.cors import CORSMiddleware


from dotenv import load_dotenv
import os 

from comet import download_model, load_from_checkpoint
from evaluation.runner import configure_eval_llm, EvalLLMConfig
from evaluation.web_search import configure_search, SearchConfig
import llminsight

load_dotenv(dotenv_path="../.env")

openai_api_key = os.getenv("OPENAI_API_KEY")

model_path = download_model("Unbabel/wmt22-comet-da")

@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()

    # Eval LLM — configurable via .env
    # EVAL_LLM_PROVIDER: openai | azure | anthropic
    # EVAL_LLM_MODEL:    gpt-4o, claude-3-5-sonnet, etc.
    # EVAL_LLM_API_KEY:  API key for the eval provider
    # EVAL_LLM_BASE_URL: (optional) for OpenAI-compatible APIs (Ollama, Groq, DeepSeek…)
    eval_provider = os.getenv("EVAL_LLM_PROVIDER", "openai")
    eval_api_key = os.getenv("EVAL_LLM_API_KEY", openai_api_key or "")
    eval_model = os.getenv("EVAL_LLM_MODEL", "gpt-4o")

    if eval_api_key:
        configure_eval_llm(EvalLLMConfig(
            provider=eval_provider,
            model=eval_model,
            api_key=eval_api_key,
            base_url=os.getenv("EVAL_LLM_BASE_URL"),
            azure_endpoint=os.getenv("EVAL_LLM_AZURE_ENDPOINT"),
            azure_api_version=os.getenv("EVAL_LLM_AZURE_API_VERSION", "2024-02-15-preview"),
        ))

        

    # Fact-check via web search (for generative LLM projects)
    # FACT_CHECK_PROVIDER: none | tavily | serper | brave
    # FACT_CHECK_API_KEY:  API key for the search provider
    fact_provider = os.getenv("FACT_CHECK_PROVIDER", "none")
    fact_api_key = os.getenv("FACT_CHECK_API_KEY", "")
    configure_search(SearchConfig(provider=fact_provider, api_key=fact_api_key))

    # Load heavy resources ONCE
    # app.state.faiss_index = faiss.read_index("resources/recent_wiki.index")
    # app.state.category_index= faiss.read_index("resources/recent_category.index")
    app.state.prompt_injection_scanner = PromptInjection(threshold=0.5, match_type=MatchType.FULL)


    app.state.embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    app.state.openai_client = OpenAI(api_key=openai_api_key)
    app.state.translation_eval_model = load_from_checkpoint(model_path)
    yield


origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",  
    "https://localhost:5173", 
    "http://localhost:8081",
    "http://127.0.0.1:8081",
]



app=FastAPI(lifespan=lifespan)


app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],

)

app.include_router(router)
app.include_router(proxy_router)
app.include_router(resources_router)
app.include_router(retriever_router)



