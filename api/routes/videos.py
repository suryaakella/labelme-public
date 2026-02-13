import io
import logging
from typing import List, Optional

from fastapi import APIRouter, Query, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from sqlalchemy import select, func, desc

from config.database import (
    get_session, Video, VideoTag, Detection, ActivitySegment,
    Transcript, TranscriptSegment, Annotation, Keyframe, Comment,
)
from config.storage import storage

logger = logging.getLogger(__name__)
router = APIRouter(tags=["videos"])


def _gdpr_status_message(gdpr_flags: Optional[dict]) -> Optional[str]:
    """Derive a human-readable GDPR status from gdpr_flags JSONB."""
    if not gdpr_flags:
        return None
    status = gdpr_flags.get("status", "")
    if status == "blocked":
        pii_types = gdpr_flags.get("pii_types", [])
        count = gdpr_flags.get("pii_count", 0)
        return (
            f"Not GDPR compliant — {count} personal data item(s) detected "
            f"({', '.join(pii_types)}). This video has been blocked and will "
            f"not be processed further."
        )
    if status == "unverified":
        reason = gdpr_flags.get("reason", "")
        return f"GDPR compliance could not be verified. {reason}"
    if status == "error":
        return "GDPR scan encountered an error. This video has been quarantined until it can be verified."
    if status == "clean":
        return "GDPR compliant — no personal data detected."
    return None


class VideoSummary(BaseModel):
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
    analytics_summary: Optional[dict] = None
    gdpr_flags: Optional[dict] = None
    gdpr_status: Optional[str] = None


class VideoListResponse(BaseModel):
    videos: List[VideoSummary]
    total: int
    page: int
    page_size: int


class DetectionItem(BaseModel):
    label: str
    confidence: float
    bbox: dict


class ActivityItem(BaseModel):
    activity_type: str
    start_time: float
    end_time: float
    confidence: float


class TranscriptSegmentItem(BaseModel):
    start_time: float
    end_time: float
    text: str


class CommentItem(BaseModel):
    username: Optional[str] = None
    text: str
    like_count: int = 0
    reply_count: int = 0
    sentiment: Optional[dict] = None
    posted_at: Optional[str] = None


class VideoDetail(BaseModel):
    id: int
    url: str
    platform: str
    title: Optional[str]
    description: Optional[str]
    duration_sec: Optional[float]
    width: Optional[int]
    height: Optional[int]
    fps: Optional[float]
    file_size_bytes: Optional[int]
    status: str
    current_step: Optional[str] = None
    storage_url: Optional[str]
    thumbnail_url: Optional[str]
    tags: List[str]
    detections: List[DetectionItem]
    activities: List[ActivityItem]
    transcript_text: Optional[str]
    transcript_segments: List[TranscriptSegmentItem]
    annotation: Optional[dict]
    created_at: str
    # Rich metadata
    creator_username: Optional[str] = None
    creator_id: Optional[str] = None
    creator_followers: Optional[int] = None
    creator_avatar_url: Optional[str] = None
    hashtags: List[str] = []
    mentions: List[str] = []
    music_info: Optional[dict] = None
    language: Optional[dict] = None
    sticker_texts: List[str] = []
    is_ad: bool = False
    is_ai_generated: bool = False
    posted_at: Optional[str] = None
    engagement: Optional[dict] = None
    analytics: Optional[dict] = None
    gdpr_flags: Optional[dict] = None
    gdpr_status: Optional[str] = None
    comments: List[CommentItem] = []


@router.get("/videos", response_model=VideoListResponse)
async def list_videos(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    platform: Optional[str] = Query(default=None),
    status: Optional[str] = Query(default=None),
    tag: Optional[str] = Query(default=None),
):
    offset = (page - 1) * page_size

    async with get_session() as session:
        query = select(Video)

        if platform:
            query = query.where(Video.platform == platform)
        if status:
            query = query.where(Video.status == status)
        if tag:
            query = query.join(VideoTag, VideoTag.video_id == Video.id).where(VideoTag.tag == tag)

        # Count
        count_q = select(func.count()).select_from(query.subquery())
        total = (await session.execute(count_q)).scalar() or 0

        # Paginate
        query = query.order_by(desc(Video.created_at)).offset(offset).limit(page_size)
        result = await session.execute(query)
        videos = result.scalars().all()

        # Fetch tags for each video
        video_ids = [v.id for v in videos]
        tag_result = await session.execute(
            select(VideoTag).where(VideoTag.video_id.in_(video_ids))
        ) if video_ids else None
        tags_map = {}
        if tag_result:
            for vt in tag_result.scalars().all():
                tags_map.setdefault(vt.video_id, []).append(vt.tag)

    summaries = []
    for v in videos:
        thumb_url = f"/api/videos/{v.id}/thumbnail" if v.thumbnail_key else None
        analytics_data = v.analytics or {}
        analytics_summary = None
        if analytics_data:
            analytics_summary = {
                "performance_tier": (analytics_data.get("engagement") or {}).get("performance_tier"),
                "brand_safety_tier": (analytics_data.get("brand_safety") or {}).get("tier"),
                "sentiment_avg": (analytics_data.get("comment_sentiment") or {}).get("avg_compound"),
            }
        summaries.append(VideoSummary(
            id=v.id,
            url=v.url,
            platform=v.platform,
            title=v.title,
            duration_sec=v.duration_sec,
            status=v.status,
            thumbnail_url=thumb_url,
            created_at=v.created_at.isoformat() if v.created_at else "",
            tags=tags_map.get(v.id, []),
            creator_username=v.creator_username,
            engagement=v.engagement,
            analytics_summary=analytics_summary,
            gdpr_flags=v.gdpr_flags,
            gdpr_status=_gdpr_status_message(v.gdpr_flags),
        ))

    return VideoListResponse(videos=summaries, total=total, page=page, page_size=page_size)


@router.get("/videos/{video_id}", response_model=VideoDetail)
async def get_video(video_id: int):
    async with get_session() as session:
        video = await session.get(Video, video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        # Tags
        tag_result = await session.execute(
            select(VideoTag).where(VideoTag.video_id == video_id)
        )
        tags = [vt.tag for vt in tag_result.scalars().all()]

        # Detections
        det_result = await session.execute(
            select(Detection).where(Detection.video_id == video_id)
        )
        detections = [
            DetectionItem(
                label=d.label,
                confidence=d.confidence,
                bbox={"x": d.bbox_x, "y": d.bbox_y, "w": d.bbox_w, "h": d.bbox_h},
            )
            for d in det_result.scalars().all()
        ]

        # Activities
        act_result = await session.execute(
            select(ActivitySegment).where(ActivitySegment.video_id == video_id)
        )
        activities = [
            ActivityItem(
                activity_type=a.activity_type,
                start_time=a.start_time,
                end_time=a.end_time,
                confidence=a.confidence,
            )
            for a in act_result.scalars().all()
        ]

        # Transcript
        trans_result = await session.execute(
            select(Transcript).where(Transcript.video_id == video_id)
        )
        transcript = trans_result.scalars().first()
        transcript_text = transcript.full_text if transcript else None

        seg_result = await session.execute(
            select(TranscriptSegment).where(TranscriptSegment.video_id == video_id)
        )
        segments = [
            TranscriptSegmentItem(start_time=s.start_time, end_time=s.end_time, text=s.text)
            for s in seg_result.scalars().all()
        ]

        # Annotation
        ann_result = await session.execute(
            select(Annotation).where(Annotation.video_id == video_id)
        )
        annotation_obj = ann_result.scalars().first()
        annotation = annotation_obj.data if annotation_obj else None

        # Comments
        comment_result = await session.execute(
            select(Comment).where(Comment.video_id == video_id)
        )
        comments = [
            CommentItem(
                username=c.username,
                text=c.text,
                like_count=c.like_count or 0,
                reply_count=c.reply_count or 0,
                sentiment=c.sentiment,
                posted_at=c.posted_at.isoformat() if c.posted_at else None,
            )
            for c in comment_result.scalars().all()
        ]

        storage_url = f"/api/videos/{video_id}/file" if video.storage_key else None
        thumb_url = f"/api/videos/{video_id}/thumbnail" if video.thumbnail_key else None

    return VideoDetail(
        id=video.id,
        url=video.url,
        platform=video.platform,
        title=video.title,
        description=video.description,
        duration_sec=video.duration_sec,
        width=video.width,
        height=video.height,
        fps=video.fps,
        file_size_bytes=video.file_size_bytes,
        status=video.status,
        current_step=video.current_step,
        storage_url=storage_url,
        thumbnail_url=thumb_url,
        tags=tags,
        detections=detections,
        activities=activities,
        transcript_text=transcript_text,
        transcript_segments=segments,
        annotation=annotation,
        created_at=video.created_at.isoformat() if video.created_at else "",
        # Rich metadata
        creator_username=video.creator_username,
        creator_id=video.creator_id,
        creator_followers=video.creator_followers,
        creator_avatar_url=video.creator_avatar_url,
        hashtags=video.hashtags or [],
        mentions=video.mentions or [],
        music_info=video.music_info,
        language=video.language,
        sticker_texts=video.sticker_texts or [],
        is_ad=video.is_ad or False,
        is_ai_generated=video.is_ai_generated or False,
        posted_at=video.posted_at.isoformat() if video.posted_at else None,
        engagement=video.engagement,
        analytics=video.analytics,
        gdpr_flags=video.gdpr_flags,
        gdpr_status=_gdpr_status_message(video.gdpr_flags),
        comments=comments,
    )


@router.get("/videos/{video_id}/file")
async def get_video_file(video_id: int):
    """Proxy the original video file from MinIO."""
    async with get_session() as session:
        video = await session.get(Video, video_id)
        if not video or not video.storage_key:
            raise HTTPException(status_code=404, detail="Video file not found")
        storage_key = video.storage_key
        title = video.title or f"video_{video_id}"

    try:
        response = storage.client.get_object(storage.bucket, storage_key)
        filename = f"{title}.mp4".replace("/", "_").replace('"', "")
        return StreamingResponse(
            response,
            media_type="video/mp4",
            headers={
                "Content-Disposition": f'inline; filename="{filename}"',
                "Accept-Ranges": "bytes",
            },
        )
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/videos/{video_id}/thumbnail")
async def get_video_thumbnail(video_id: int):
    """Proxy the thumbnail image from MinIO."""
    async with get_session() as session:
        video = await session.get(Video, video_id)
        if not video or not video.thumbnail_key:
            raise HTTPException(status_code=404, detail="Thumbnail not found")
        thumbnail_key = video.thumbnail_key

    ext = thumbnail_key.rsplit(".", 1)[-1] if "." in thumbnail_key else "jpg"
    content_type = {"jpg": "image/jpeg", "jpeg": "image/jpeg", "png": "image/png", "webp": "image/webp"}.get(ext, "image/jpeg")

    try:
        response = storage.client.get_object(storage.bucket, thumbnail_key)
        return StreamingResponse(response, media_type=content_type)
    except Exception as e:
        raise HTTPException(status_code=404, detail=str(e))


@router.get("/videos/{video_id}/annotation/download")
async def download_annotation(video_id: int):
    """Download the annotation JSON for a video."""
    import json
    async with get_session() as session:
        video = await session.get(Video, video_id)
        if not video:
            raise HTTPException(status_code=404, detail="Video not found")

        ann_result = await session.execute(
            select(Annotation).where(Annotation.video_id == video_id)
        )
        annotation = ann_result.scalars().first()
        if not annotation:
            raise HTTPException(status_code=404, detail="No annotation for this video")

        data = json.dumps(annotation.data, indent=2)
        filename = f"annotation_video_{video_id}.json"

    return StreamingResponse(
        io.BytesIO(data.encode()),
        media_type="application/json",
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
