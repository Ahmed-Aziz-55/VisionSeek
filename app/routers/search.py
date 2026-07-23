"""
app/routers/search.py

Search endpoint. The ImageSearcher instance is created once at app
startup (see app/main.py's lifespan) and reused across requests.
image_url is derived from image_path so the Flutter frontend (or any
HTTP client) gets a fetchable URL, not a server-side filesystem path.
"""

import logging
from pathlib import Path

from fastapi import APIRouter, HTTPException, Request

from app.schemas.search import SearchRequest, SearchResponse, SearchResult

logger = logging.getLogger(__name__)
router = APIRouter()


@router.post("/search", response_model=SearchResponse)
def search(payload: SearchRequest, request: Request) -> SearchResponse:
    searcher = getattr(request.app.state, "searcher", None)
    if searcher is None:
        raise HTTPException(status_code=503, detail="Searcher not initialized yet")

    try:
        raw_results = searcher.search(payload.query, top_k=payload.top_k)
    except Exception as e:
        logger.error(f"Search failed for query '{payload.query}': {e}")
        raise HTTPException(status_code=500, detail="Search failed") from e

    results = []
    for r in raw_results:
        filename = Path(r["image_path"]).name
        results.append(SearchResult(
            image_path=r["image_path"],
            image_url=f"/images/{filename}",
            caption=r["caption"],
            score=r["score"],
        ))

    return SearchResponse(
        query=payload.query,
        count=len(results),
        results=results,
    )