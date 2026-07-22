import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request

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
app.include_router(search_router)


@app.get("/health")
def health(request: Request) -> dict:
    ready = getattr(request.app.state, "searcher", None) is not None
    return {"status": "ok" if ready else "loading", "searcher_ready": ready}

