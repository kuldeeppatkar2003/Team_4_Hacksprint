from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
import json
import asyncio
import logging
from src.storage.database import db
from src.storage.chroma_store import vector_store
from src.processing.llm_processor import LLMProcessor
from src.utils.manager import manager

from contextlib import asynccontextmanager
from src.ingestion.scraper import poll_feeds

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Start the scraper in the background
    scraper_task = asyncio.create_task(poll_feeds())
    logger.info("Background scraper started.")
    yield
    # Clean up
    scraper_task.cancel()
    try:
        await scraper_task
    except asyncio.CancelledError:
        logger.info("Background scraper stopped.")

app = FastAPI(title="RSS Intelligence API", lifespan=lifespan)

# CORS setup
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

logger = logging.getLogger(__name__)
llm = LLMProcessor()

# WebSocket endpoints use the shared manager

@app.get("/api/articles")
async def get_articles(limit: int = 20):
    return await db.get_latest_articles(limit)

@app.get("/api/stats")
async def get_stats():
    # Simple count for now
    count = await db.articles.count_documents({})
    return {"total_articles": count}

@app.get("/api/insights")
async def get_insights():
    return await db.get_insights()

@app.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            await websocket.receive_text() # Keep alive
    except WebSocketDisconnect:
        manager.disconnect(websocket)

@app.post("/api/chat")
async def chat(query: dict):
    user_query = query.get("text")
    # 1. RAG: Search ChromaDB
    results = vector_store.query(user_query)
    
    # 2. Build Context
    context = ""
    if results and results['documents']:
        context = "\n".join(results['documents'][0])
        
    # 3. Ask LLM (Simple implementation for now)
    prompt = f"Context: {context}\n\nQuestion: {user_query}\nAnswer based on context:"
    response = llm.model.generate_content(prompt)
    
    return {"answer": response.text, "sources": results['metadatas'][0] if results else []}

# Mount static files
app.mount("/", StaticFiles(directory="src/static", html=True), name="static")
