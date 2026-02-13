import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel
from sqlalchemy import select, desc

from config.database import get_session, IngestionTask, Video, VideoTag

logger = logging.getLogger(__name__)
router = APIRouter(tags=["tasks"])


class TaskResponse(BaseModel):
    id: int
    query: str
    platform: str
    max_results: int
    status: str
    progress: float
    total_videos: Optional[int]
    processed_videos: Optional[int]
    error_message: Optional[str]
    current_step: Optional[str]
    created_at: str
    updated_at: str


@router.get("/tasks", response_model=List[TaskResponse])
async def list_tasks(
    status: Optional[str] = Query(default=None),
    limit: int = Query(default=50, ge=1, le=200),
):
    async with get_session() as session:
        query = select(IngestionTask)
        if status:
            query = query.where(IngestionTask.status == status)
        query = query.order_by(desc(IngestionTask.created_at)).limit(limit)

        result = await session.execute(query)
        tasks = result.scalars().all()

    return [
        TaskResponse(
            id=t.id,
            query=t.query,
            platform=t.platform,
            max_results=t.max_results,
            status=t.status,
            progress=t.progress,
            total_videos=t.total_videos,
            processed_videos=t.processed_videos,
            error_message=t.error_message,
            current_step=t.current_step,
            created_at=t.created_at.isoformat() if t.created_at else "",
            updated_at=t.updated_at.isoformat() if t.updated_at else "",
        )
        for t in tasks
    ]


@router.get("/tasks/{task_id}", response_model=TaskResponse)
async def get_task(task_id: int):
    async with get_session() as session:
        task = await session.get(IngestionTask, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

    return TaskResponse(
        id=task.id,
        query=task.query,
        platform=task.platform,
        max_results=task.max_results,
        status=task.status,
        progress=task.progress,
        total_videos=task.total_videos,
        processed_videos=task.processed_videos,
        error_message=task.error_message,
        current_step=task.current_step,
        created_at=task.created_at.isoformat() if task.created_at else "",
        updated_at=task.updated_at.isoformat() if task.updated_at else "",
    )


class TaskVideoSummary(BaseModel):
    id: int
    url: str
    platform: str
    title: Optional[str]
    duration_sec: Optional[float]
    status: str
    thumbnail_url: Optional[str]
    created_at: str
    tags: List[str] = []
    creator_username: Optional[str] = None
    engagement: Optional[dict] = None
    error_message: Optional[str] = None
    current_step: Optional[str] = None
    performance_tier: Optional[str] = None
    brand_safety_tier: Optional[str] = None
    sentiment_avg: Optional[float] = None
    sentiment_label: Optional[str] = None
    content_categories: List[str] = []
    is_ai_generated: Optional[bool] = None
    gdpr_status: Optional[str] = None


class TaskVideosResponse(BaseModel):
    task: TaskResponse
    videos: List[TaskVideoSummary]


@router.get("/tasks/{task_id}/videos", response_model=TaskVideosResponse)
async def get_task_videos(task_id: int):
    async with get_session() as session:
        task = await session.get(IngestionTask, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        result = await session.execute(
            select(Video).where(Video.task_id == task_id).order_by(desc(Video.created_at))
        )
        videos = result.scalars().all()

        video_ids = [v.id for v in videos]
        tags_map = {}
        if video_ids:
            tag_result = await session.execute(
                select(VideoTag).where(VideoTag.video_id.in_(video_ids))
            )
            for vt in tag_result.scalars().all():
                tags_map.setdefault(vt.video_id, []).append(vt.tag)

    task_resp = TaskResponse(
        id=task.id,
        query=task.query,
        platform=task.platform,
        max_results=task.max_results,
        status=task.status,
        progress=task.progress,
        total_videos=task.total_videos,
        processed_videos=task.processed_videos,
        error_message=task.error_message,
        current_step=task.current_step,
        created_at=task.created_at.isoformat() if task.created_at else "",
        updated_at=task.updated_at.isoformat() if task.updated_at else "",
    )

    video_summaries = []
    for v in videos:
        analytics = v.analytics or {}
        gdpr = v.gdpr_flags or {}

        # Extract analytics summary fields
        engagement_data = analytics.get("engagement", {})
        brand_safety = analytics.get("brand_safety", {})
        sentiment = analytics.get("comment_sentiment", {})
        ai_gen = analytics.get("ai_generated", {})
        categories = analytics.get("content_categories", [])

        # Sentiment label from compound score
        sent_avg = sentiment.get("avg_compound")
        sent_label = None
        if sent_avg is not None:
            if sent_avg >= 0.05:
                sent_label = "positive"
            elif sent_avg <= -0.05:
                sent_label = "negative"
            else:
                sent_label = "neutral"

        # Top 3 category names
        top_categories = []
        if isinstance(categories, list):
            for cat in categories[:3]:
                if isinstance(cat, dict):
                    top_categories.append(cat.get("name", str(cat)))
                else:
                    top_categories.append(str(cat))

        video_summaries.append(TaskVideoSummary(
            id=v.id,
            url=v.url,
            platform=v.platform,
            title=v.title,
            duration_sec=v.duration_sec,
            status=v.status,
            thumbnail_url=f"/api/videos/{v.id}/thumbnail" if v.thumbnail_key else None,
            created_at=v.created_at.isoformat() if v.created_at else "",
            tags=tags_map.get(v.id, []),
            creator_username=v.creator_username,
            engagement=v.engagement,
            error_message=v.error_message,
            current_step=v.current_step,
            performance_tier=engagement_data.get("performance_tier"),
            brand_safety_tier=brand_safety.get("tier"),
            sentiment_avg=sent_avg,
            sentiment_label=sent_label,
            content_categories=top_categories,
            is_ai_generated=ai_gen.get("is_ai_generated"),
            gdpr_status=gdpr.get("status"),
        ))

    return TaskVideosResponse(task=task_resp, videos=video_summaries)
