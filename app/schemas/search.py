from pydantic import BaseModel, Field


class SearchRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Text query to search images for")
    top_k: int = Field(default=5, ge=1, le=50, description="Number of results to return")


class SearchResult(BaseModel):
    image_path: str
    image_url: str
    caption: str
    score: float


class SearchResponse(BaseModel):
    query: str
    count: int
    results: list[SearchResult]