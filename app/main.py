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

load_dotenv(dotenv_path="../.env")

openai_api_key = os.getenv("OPENAI_API_KEY")



@asynccontextmanager
async def lifespan(app: FastAPI):
    await init_db()
    # Load heavy resources ONCE
    # app.state.faiss_index = faiss.read_index("resources/recent_wiki.index")
    # app.state.category_index= faiss.read_index("resources/recent_category.index")
    app.state.prompt_injection_scanner = PromptInjection(threshold=0.5, match_type=MatchType.FULL)


    app.state.embedding_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
    app.state.openai_client = OpenAI(api_key=openai_api_key)
    yield


origins = [
    "http://localhost:5173",
    "http://127.0.0.1:5173",  
    "https://localhost:5173", 
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



