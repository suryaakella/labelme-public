import asyncio
import json
import logging
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select

from config import settings
from config.database import (
    get_session, Video, VideoTag, Detection, ActivitySegment,
    Transcript, TranscriptSegment, Annotation, Comment, IngestionTask,
)
from config.task_queue import task_queue
from ingestion.orchestrator import IngestionOrchestrator
from ingestion.query_parser import parse_ingest_query
from api.routes.videos import (
    VideoDetail, DetectionItem, ActivityItem,
    TranscriptSegmentItem, CommentItem, _gdpr_status_message,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["run"])


# ── Request / Response models ───────────────────────────────────

class RunRequest(BaseModel):
    query: str = Field(..., min_length=1)


class RunResponse(BaseModel):
    task_id: int


class RunResultResponse(BaseModel):
    task_id: int
    status: str
    progress: float = 0.0
    current_step: Optional[str] = None
    total_videos: int = 0
    processed_videos: int = 0
    error_message: Optional[str] = None
    videos: List[VideoDetail] = []


# ── Batch helper ────────────────────────────────────────────────

async def _build_task_video_details(task_id: int) -> List[VideoDetail]:
    """Load all video details for a task using batched IN queries."""
    async with get_session() as session:
        # 1. All videos for this task
        result = await session.execute(
            select(Video).where(Video.task_id == task_id)
        )
        videos = result.scalars().all()
        if not videos:
            return []

        video_ids = [v.id for v in videos]

        # 2. Tags
        tag_result = await session.execute(
            select(VideoTag).where(VideoTag.video_id.in_(video_ids))
        )
        tags_map: dict[int, list[str]] = {}
        for vt in tag_result.scalars().all():
            tags_map.setdefault(vt.video_id, []).append(vt.tag)

        # 3. Detections
        det_result = await session.execute(
            select(Detection).where(Detection.video_id.in_(video_ids))
        )
        det_map: dict[int, list[DetectionItem]] = {}
        for d in det_result.scalars().all():
            det_map.setdefault(d.video_id, []).append(DetectionItem(
                label=d.label,
                confidence=d.confidence,
                bbox={"x": d.bbox_x, "y": d.bbox_y, "w": d.bbox_w, "h": d.bbox_h},
            ))

        # 4. Activities
        act_result = await session.execute(
            select(ActivitySegment).where(ActivitySegment.video_id.in_(video_ids))
        )
        act_map: dict[int, list[ActivityItem]] = {}
        for a in act_result.scalars().all():
            act_map.setdefault(a.video_id, []).append(ActivityItem(
                activity_type=a.activity_type,
                start_time=a.start_time,
                end_time=a.end_time,
                confidence=a.confidence,
            ))

        # 5. Transcripts
        trans_result = await session.execute(
            select(Transcript).where(Transcript.video_id.in_(video_ids))
        )
        trans_map: dict[int, str] = {}
        for t in trans_result.scalars().all():
            trans_map[t.video_id] = t.full_text

        # 6. Transcript segments
        seg_result = await session.execute(
            select(TranscriptSegment).where(TranscriptSegment.video_id.in_(video_ids))
        )
        seg_map: dict[int, list[TranscriptSegmentItem]] = {}
        for s in seg_result.scalars().all():
            seg_map.setdefault(s.video_id, []).append(TranscriptSegmentItem(
                start_time=s.start_time, end_time=s.end_time, text=s.text,
            ))

        # 7. Annotations
        ann_result = await session.execute(
            select(Annotation).where(Annotation.video_id.in_(video_ids))
        )
        ann_map: dict[int, dict] = {}
        for a in ann_result.scalars().all():
            ann_map[a.video_id] = a.data

        # 8. Comments
        comment_result = await session.execute(
            select(Comment).where(Comment.video_id.in_(video_ids))
        )
        comment_map: dict[int, list[CommentItem]] = {}
        for c in comment_result.scalars().all():
            comment_map.setdefault(c.video_id, []).append(CommentItem(
                username=c.username,
                text=c.text,
                like_count=c.like_count or 0,
                reply_count=c.reply_count or 0,
                sentiment=c.sentiment,
                posted_at=c.posted_at.isoformat() if c.posted_at else None,
            ))

    # Build VideoDetail list
    details: List[VideoDetail] = []
    for v in videos:
        storage_url = f"/api/videos/{v.id}/file" if v.storage_key else None
        thumb_url = f"/api/videos/{v.id}/thumbnail" if v.thumbnail_key else None
        details.append(VideoDetail(
            id=v.id,
            url=v.url,
            platform=v.platform,
            title=v.title,
            description=v.description,
            duration_sec=v.duration_sec,
            width=v.width,
            height=v.height,
            fps=v.fps,
            file_size_bytes=v.file_size_bytes,
            status=v.status,
            current_step=v.current_step,
            storage_url=storage_url,
            thumbnail_url=thumb_url,
            tags=tags_map.get(v.id, []),
            detections=det_map.get(v.id, []),
            activities=act_map.get(v.id, []),
            transcript_text=trans_map.get(v.id),
            transcript_segments=seg_map.get(v.id, []),
            annotation=ann_map.get(v.id),
            created_at=v.created_at.isoformat() if v.created_at else "",
            creator_username=v.creator_username,
            creator_id=v.creator_id,
            creator_followers=v.creator_followers,
            creator_avatar_url=v.creator_avatar_url,
            hashtags=v.hashtags or [],
            mentions=v.mentions or [],
            music_info=v.music_info,
            language=v.language,
            sticker_texts=v.sticker_texts or [],
            is_ad=v.is_ad or False,
            is_ai_generated=v.is_ai_generated or False,
            posted_at=v.posted_at.isoformat() if v.posted_at else None,
            engagement=v.engagement,
            analytics=v.analytics,
            gdpr_flags=v.gdpr_flags,
            gdpr_status=_gdpr_status_message(v.gdpr_flags),
            comments=comment_map.get(v.id, []),
        ))
    return details


# ── Endpoints ───────────────────────────────────────────────────

@router.post("/run", response_model=RunResponse)
async def create_run(req: RunRequest):
    """Accept a query, create an ingestion task, return task_id immediately."""
    parsed = await parse_ingest_query(req.query)

    # GDPR check
    if settings.gdpr_query_check_enabled and settings.google_api_key:
        try:
            from labeling.gdpr import screen_query
            result = await screen_query(parsed.search_topic)
            if result.is_personal_data_query:
                raise HTTPException(status_code=422, detail={
                    "error": "gdpr_query_blocked",
                    "message": (
                        "This query is not GDPR compliant. Queries targeting specific "
                        "individuals' personal data cannot be processed. Please use "
                        "topic-based, brand, or category searches instead."
                    ),
                    "risk_level": result.risk_level,
                    "explanation": result.explanation,
                })
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"GDPR query check failed ({e}), allowing ingestion to proceed")

    async with get_session() as session:
        task = IngestionTask(
            query=req.query,
            search_topic=parsed.search_topic,
            platform=parsed.platform,
            max_results=parsed.max_results,
            status="pending",
        )
        session.add(task)
        await session.flush()
        task_id = task.id

    orchestrator = IngestionOrchestrator()
    task_queue.submit_async(orchestrator.run, task_id, task_id=str(task_id))

    return RunResponse(task_id=task_id)


@router.get("/run/{task_id}/stream")
async def stream_run(task_id: int):
    """SSE stream that emits progress events until the task completes or fails."""

    # Verify the task exists before opening the stream
    async with get_session() as session:
        task = await session.get(IngestionTask, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")

    async def event_generator():
        last_progress = None
        last_step = None

        while True:
            async with get_session() as session:
                task = await session.get(IngestionTask, task_id)

            if not task:
                payload = json.dumps({"task_id": task_id, "status": "not_found", "error_message": "Task not found"})
                yield f"event: error\ndata: {payload}\n\n"
                return

            current = (task.progress, task.current_step, task.status)

            if task.status == "completed":
                # Emit final progress
                payload = json.dumps({
                    "task_id": task_id,
                    "status": task.status,
                    "progress": task.progress,
                    "current_step": task.current_step,
                    "total_videos": task.total_videos or 0,
                    "processed_videos": task.processed_videos or 0,
                })
                yield f"event: progress\ndata: {payload}\n\n"

                # Build full result
                videos = await _build_task_video_details(task_id)
                complete_payload = json.dumps({
                    "task_id": task_id,
                    "status": "completed",
                    "videos": [v.model_dump() for v in videos],
                }, default=str)
                yield f"event: complete\ndata: {complete_payload}\n\n"
                return

            if task.status == "failed":
                payload = json.dumps({
                    "task_id": task_id,
                    "status": "failed",
                    "error_message": task.error_message,
                })
                yield f"event: error\ndata: {payload}\n\n"
                return

            # Emit progress only when something changed
            progress_key = (task.progress, task.current_step)
            if progress_key != (last_progress, last_step):
                last_progress = task.progress
                last_step = task.current_step
                payload = json.dumps({
                    "task_id": task_id,
                    "status": task.status,
                    "progress": task.progress,
                    "current_step": task.current_step,
                    "total_videos": task.total_videos or 0,
                    "processed_videos": task.processed_videos or 0,
                })
                yield f"event: progress\ndata: {payload}\n\n"

            await asyncio.sleep(2)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/run/{task_id}/result", response_model=RunResultResponse)
async def get_run_result(task_id: int):
    """Fetch the current state of a run task (works for any status)."""
    async with get_session() as session:
        task = await session.get(IngestionTask, task_id)
        if not task:
            raise HTTPException(status_code=404, detail="Task not found")
        task_status = task.status
        task_progress = task.progress
        task_current_step = task.current_step
        task_total_videos = task.total_videos or 0
        task_processed_videos = task.processed_videos or 0
        task_error_message = task.error_message

    videos = await _build_task_video_details(task_id) if task_status == "completed" else []

    return RunResultResponse(
        task_id=task_id,
        status=task_status,
        progress=task_progress,
        current_step=task_current_step,
        total_videos=task_total_videos,
        processed_videos=task_processed_videos,
        error_message=task_error_message,
        videos=videos,
    )
