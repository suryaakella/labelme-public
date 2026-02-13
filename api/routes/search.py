import logging
from typing import List, Optional

from fastapi import APIRouter, Query
from pydantic import BaseModel
from sqlalchemy import text

from config.database import get_session
from pipeline.embeddings import CLIPEmbedder

logger = logging.getLogger(__name__)
router = APIRouter(tags=["search"])


class SearchResult(BaseModel):
    video_id: int
    title: Optional[str]
    url: str
    platform: str
    similarity: float
    duration_sec: Optional[float]
    thumbnail_url: Optional[str]
    status: str
    creator_username: Optional[str] = None
    engagement: Optional[dict] = None
    analytics_summary: Optional[dict] = None


class SearchResponse(BaseModel):
    query: str
    results: List[SearchResult]
    total: int


@router.get("/search", response_model=SearchResponse)
async def search_videos(
    q: str = Query(..., min_length=1, description="Search query"),
    limit: int = Query(default=20, ge=1, le=100),
    platform: Optional[str] = Query(default=None),
    min_similarity: float = Query(default=0.0, ge=0.0, le=1.0),
):
    embedder = CLIPEmbedder()
    query_embedding = embedder.embed_text(q)
    embedding_str = "[" + ",".join(str(x) for x in query_embedding.tolist()) + "]"

    platform_filter = ""
    params = {"embedding": embedding_str, "limit": limit, "min_sim": min_similarity}

    if platform:
        platform_filter = "AND v.platform = :platform"
        params["platform"] = platform

    # Use CAST() instead of ::vector to avoid asyncpg named-param conflict
    sql = f"""
        SELECT v.id, v.title, v.url, v.platform, v.duration_sec,
               v.thumbnail_key, v.status,
               1 - (ve.embedding <=> CAST(:embedding AS vector)) AS similarity,
               v.creator_username, v.engagement, v.analytics
        FROM video_embeddings ve
        JOIN videos v ON v.id = ve.video_id
        WHERE v.status = 'completed'
          AND 1 - (ve.embedding <=> CAST(:embedding AS vector)) > :min_sim
          {platform_filter}
        ORDER BY ve.embedding <=> CAST(:embedding AS vector)
        LIMIT :limit
    """

    async with get_session() as session:
        result = await session.execute(text(sql), params)
        rows = result.fetchall()

    results = []
    for row in rows:
        vid_id = row[0]
        thumb_url = f"/api/videos/{vid_id}/thumbnail" if row[5] else None
        analytics_data = row[10] or {}
        analytics_summary = None
        if analytics_data:
            analytics_summary = {
                "performance_tier": (analytics_data.get("engagement") or {}).get("performance_tier"),
                "brand_safety_tier": (analytics_data.get("brand_safety") or {}).get("tier"),
                "sentiment_avg": (analytics_data.get("comment_sentiment") or {}).get("avg_compound"),
            }
        results.append(SearchResult(
            video_id=vid_id,
            title=row[1],
            url=row[2],
            platform=row[3],
            duration_sec=row[4],
            thumbnail_url=thumb_url,
            status=row[6],
            similarity=round(row[7], 4),
            creator_username=row[8],
            engagement=row[9],
            analytics_summary=analytics_summary,
        ))

    return SearchResponse(query=q, results=results, total=len(results))
