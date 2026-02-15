import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from sqlalchemy import text, func, select

from config import settings
from config.database import get_session, Video, IngestionTask
from config.storage import storage
from config.task_queue import task_queue
from sqlalchemy import update

from api.routes.ingest import router as ingest_router
from api.routes.search import router as search_router
from api.routes.videos import router as videos_router
from api.routes.datasets import router as datasets_router
from api.routes.tasks import router as tasks_router
from api.routes.run import router as run_router

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Startup
    logger.info("Starting ForgeIndex API")
    try:
        storage.ensure_bucket()
    except Exception as e:
        logger.warning(f"Could not ensure MinIO bucket: {e}")

    # Clean up stale tasks/videos from previous crashes
    try:
        async with get_session() as session:
            await session.execute(
                update(IngestionTask)
                .where(IngestionTask.status == "running")
                .values(status="failed", error_message="Server restarted", current_step=None)
            )
            await session.execute(
                update(Video)
                .where(Video.status.in_(["processing", "labeling", "downloading"]))
                .values(status="failed", error_message="Server restarted during processing", current_step=None)
            )
        logger.info("Cleaned up stale tasks/videos from previous run")
    except Exception as e:
        logger.warning(f"Could not clean stale tasks: {e}")

    yield
    # Shutdown
    task_queue.shutdown(wait=False)
    logger.info("ForgeIndex API stopped")


app = FastAPI(
    title="ForgeIndex",
    description="Video dataset pipeline for manufacturing intelligence",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(ingest_router, prefix="/api")
app.include_router(search_router, prefix="/api")
app.include_router(videos_router, prefix="/api")
app.include_router(datasets_router, prefix="/api")
app.include_router(tasks_router, prefix="/api")
app.include_router(run_router, prefix="/api")


@app.get("/health")
async def health():
    return {"status": "ok"}


@app.get("/stats")
async def stats():
    async with get_session() as session:
        # Only count fully processed videos
        completed = await session.execute(
            select(func.count(Video.id)).where(Video.status == "completed")
        )
        completed_count = completed.scalar() or 0

        processing = await session.execute(
            select(func.count(Video.id)).where(Video.status.in_(["processing", "labeling", "downloading"]))
        )
        processing_count = processing.scalar() or 0

        failed = await session.execute(
            select(func.count(Video.id)).where(Video.status == "failed")
        )
        failed_count = failed.scalar() or 0

        gdpr_blocked = await session.execute(
            select(func.count(Video.id)).where(Video.status == "gdpr_blocked")
        )
        gdpr_blocked_count = gdpr_blocked.scalar() or 0

        tasks_result = await session.execute(
            select(func.count(IngestionTask.id)).where(IngestionTask.status == "running")
        )
        active_tasks = tasks_result.scalar() or 0

    return {
        "total_videos": completed_count,
        "completed_videos": completed_count,
        "processing_videos": processing_count,
        "failed_videos": failed_count,
        "gdpr_blocked_videos": gdpr_blocked_count,
        "active_tasks": active_tasks,
    }
