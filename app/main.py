"""
app/main.py

FastAPI entry point. Loads the CLIP model and FAISS index once at
startup (lifespan), not per-request. Serves images statically under
/images so frontend clients (e.g. the Flutter app) can fetch them by URL.

Run with:
    uvicorn app.main:app --reload
"""

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.logging_config import setup_logging
from app.services.searcher import ImageSearcher
from app.routers.search import router as search_router

setup_logging()
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading ImageSearcher (FAISS index + CLIP model)...")
    app.state.searcher = ImageSearcher()
    logger.info("ImageSearcher ready.")
    yield
    logger.info("Shutting down.")


app = FastAPI(title="VisionSeek API", version="1.0.0", lifespan=lifespan)

# Dev-only: allows the Flutter web app (running on a different port) to
# call this API from the browser. Tighten allow_origins to a specific
# list before any real deployment.
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(search_router)
app.mount("/images", StaticFiles(directory="datasets/Images"), name="images")


@app.get("/health")
def health(request: Request) -> dict:
    ready = getattr(request.app.state, "searcher", None) is not None
    return {"status": "ok" if ready else "loading", "searcher_ready": ready}