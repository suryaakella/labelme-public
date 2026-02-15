import io
import json
import logging
import re
import zipfile
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select, desc

from config.database import get_session, IngestionTask, Video, VideoTag, Annotation
from config.storage import storage

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


def _safe_filename(name: str) -> str:
    """Sanitize a string for use in a filename."""
    if not name:
        return "untitled"
    return re.sub(r'[^\w\-.]', '_', name)[:80]


@router.get("/tasks/{task_id}/export")
async def export_task(task_id: int):
    async with get_session() as session:
        task = await session.get(IngestionTask, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

        result = await session.execute(
            select(Video).where(Video.task_id == task_id).order_by(Video.id)
        )
        videos = result.scalars().all()

        if not videos:
            raise HTTPException(status_code=404, detail="No videos found for this task")

        # Fetch annotations for all videos
        video_ids = [v.id for v in videos]
        annotations = {}
        if video_ids:
            ann_result = await session.execute(
                select(Annotation).where(Annotation.video_id.in_(video_ids))
            )
            for ann in ann_result.scalars().all():
                annotations[ann.video_id] = ann.data

    # Build ZIP
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        manifest = {
            "task": {
                "id": task.id,
                "query": task.query,
                "platform": task.platform,
                "status": task.status,
                "max_results": task.max_results,
                "total_videos": task.total_videos,
                "processed_videos": task.processed_videos,
                "created_at": task.created_at.isoformat() if task.created_at else None,
            },
            "videos": [],
        }

        for v in videos:
            entry = {
                "id": v.id,
                "url": v.url,
                "title": v.title,
                "platform": v.platform,
                "duration_sec": v.duration_sec,
                "creator_username": v.creator_username,
                "status": v.status,
                "has_annotation": v.id in annotations,
                "has_video_file": bool(v.storage_key),
            }
            manifest["videos"].append(entry)

            # Write annotation JSON
            if v.id in annotations:
                zf.writestr(
                    f"annotations/{v.id}.json",
                    json.dumps(annotations[v.id], indent=2, default=str),
                )

            # Download and include video file from MinIO
            if v.storage_key:
                try:
                    response = storage.client.get_object(storage.bucket, v.storage_key)
                    video_data = response.read()
                    response.close()
                    response.release_conn()
                    safe_title = _safe_filename(v.title)
                    zf.writestr(f"videos/{v.id}_{safe_title}.mp4", video_data)
                except Exception as e:
                    logger.warning(f"Failed to download video {v.id} from MinIO: {e}")
                    entry["has_video_file"] = False
                    entry["download_error"] = str(e)

        zf.writestr("manifest.json", json.dumps(manifest, indent=2, default=str))

    buf.seek(0)
    safe_query = _safe_filename(task.query)
    filename = f"task_{task.id}_{safe_query}_export.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
