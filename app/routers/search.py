import logging

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

    results = [SearchResult(**r) for r in raw_results]

    return SearchResponse(
        query=payload.query,
        count=len(results),
        results=results,
    )