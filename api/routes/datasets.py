import io
import logging
import zipfile
from typing import List, Optional

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from sqlalchemy import select, func

from config.database import (
    get_session, Dataset, DatasetVideo, Video, VideoTag, Annotation,
)

logger = logging.getLogger(__name__)
router = APIRouter(tags=["datasets"])


class DatasetCreateRequest(BaseModel):
    name: str = Field(..., min_length=1)
    description: str = ""
    filters: dict = Field(default_factory=dict)  # e.g. {"platform": "youtube", "tags": ["cnc"], "status": "completed"}


class DatasetResponse(BaseModel):
    id: int
    name: str
    description: Optional[str]
    filters: dict
    video_count: int
    created_at: str


@router.post("/datasets", response_model=DatasetResponse)
async def create_dataset(req: DatasetCreateRequest):
    async with get_session() as session:
        dataset = Dataset(
            name=req.name,
            description=req.description,
            filters=req.filters,
        )
        session.add(dataset)
        await session.flush()

        # Apply filters to select videos
        query = select(Video.id)
        if req.filters.get("platform"):
            query = query.where(Video.platform == req.filters["platform"])
        if req.filters.get("status"):
            query = query.where(Video.status == req.filters["status"])
        if req.filters.get("tags"):
            query = query.join(VideoTag, VideoTag.video_id == Video.id).where(
                VideoTag.tag.in_(req.filters["tags"])
            )

        result = await session.execute(query)
        video_ids = [row[0] for row in result.fetchall()]

        for vid in video_ids:
            dv = DatasetVideo(dataset_id=dataset.id, video_id=vid)
            session.add(dv)

        return DatasetResponse(
            id=dataset.id,
            name=dataset.name,
            description=dataset.description,
            filters=dataset.filters,
            video_count=len(video_ids),
            created_at=dataset.created_at.isoformat() if dataset.created_at else "",
        )


@router.get("/datasets", response_model=List[DatasetResponse])
async def list_datasets():
    async with get_session() as session:
        result = await session.execute(select(Dataset))
        datasets = result.scalars().all()

        responses = []
        for ds in datasets:
            count_result = await session.execute(
                select(func.count(DatasetVideo.id)).where(DatasetVideo.dataset_id == ds.id)
            )
            count = count_result.scalar() or 0
            responses.append(DatasetResponse(
                id=ds.id,
                name=ds.name,
                description=ds.description,
                filters=ds.filters,
                video_count=count,
                created_at=ds.created_at.isoformat() if ds.created_at else "",
            ))

    return responses


@router.get("/datasets/{dataset_id}/export")
async def export_dataset(dataset_id: int):
    async with get_session() as session:
        dataset = await session.get(Dataset, dataset_id)
        if not dataset:
            raise HTTPException(status_code=404, detail="Dataset not found")

        # Get all videos in this dataset
        result = await session.execute(
            select(Video)
            .join(DatasetVideo, DatasetVideo.video_id == Video.id)
            .where(DatasetVideo.dataset_id == dataset_id)
        )
        videos = result.scalars().all()

        # Get annotations
        annotations = {}
        for v in videos:
            ann_result = await session.execute(
                select(Annotation).where(Annotation.video_id == v.id)
            )
            ann = ann_result.scalars().first()
            if ann:
                annotations[v.id] = ann.data

    # Build ZIP
    import json
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        manifest = {
            "dataset": dataset.name,
            "description": dataset.description,
            "filters": dataset.filters,
            "videos": [],
        }

        for v in videos:
            entry = {
                "id": v.id,
                "url": v.url,
                "title": v.title,
                "platform": v.platform,
                "duration_sec": v.duration_sec,
                "annotation": annotations.get(v.id),
            }
            manifest["videos"].append(entry)

            # Write individual annotation file
            if v.id in annotations:
                zf.writestr(
                    f"annotations/{v.id}.json",
                    json.dumps(annotations[v.id], indent=2),
                )

        zf.writestr("manifest.json", json.dumps(manifest, indent=2))

    buf.seek(0)
    filename = f"{dataset.name.replace(' ', '_')}_export.zip"
    return StreamingResponse(
        buf,
        media_type="application/zip",
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )
