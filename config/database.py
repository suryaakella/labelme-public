import threading
from contextlib import asynccontextmanager
from datetime import datetime

from sqlalchemy import (
    Column, Integer, String, Text, Float, BigInteger, Boolean,
    ForeignKey, UniqueConstraint, CheckConstraint, Index,
    func,
)
from sqlalchemy.dialects.postgresql import JSONB, TIMESTAMP
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import DeclarativeBase, sessionmaker, relationship

from pgvector.sqlalchemy import Vector

from config import settings


class Base(DeclarativeBase):
    pass


# ── ORM Models ──────────────────────────────────────────────────

class Video(Base):
    __tablename__ = "videos"

    id = Column(Integer, primary_key=True)
    url = Column(Text, nullable=False, unique=True)
    platform = Column(String(32), nullable=False)
    title = Column(Text)
    description = Column(Text)
    duration_sec = Column(Float)
    width = Column(Integer)
    height = Column(Integer)
    fps = Column(Float)
    file_size_bytes = Column(BigInteger)
    storage_key = Column(Text)
    thumbnail_key = Column(Text)
    status = Column(String(32), nullable=False, default="pending")
    error_message = Column(Text)
    metadata_ = Column("metadata", JSONB, default={})
    # Rich Instagram metadata
    creator_username = Column(String(256))
    creator_id = Column(String(256))
    creator_followers = Column(Integer)
    creator_avatar_url = Column(Text)
    hashtags = Column(JSONB, default=[])
    mentions = Column(JSONB, default=[])
    music_info = Column(JSONB)
    language = Column(JSONB)
    sticker_texts = Column(JSONB, default=[])
    is_ad = Column(Boolean, default=False)
    is_ai_generated = Column(Boolean, default=False)
    posted_at = Column(TIMESTAMP(timezone=True))
    engagement = Column(JSONB, default={})
    analytics = Column(JSONB, default={})
    gdpr_flags = Column(JSONB)
    current_step = Column(Text)
    task_id = Column(Integer, ForeignKey("ingestion_tasks.id", ondelete="SET NULL"))
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    task = relationship("IngestionTask", back_populates="videos")
    embeddings = relationship("VideoEmbedding", back_populates="video", cascade="all, delete-orphan")
    keyframes = relationship("Keyframe", back_populates="video", cascade="all, delete-orphan")
    detections = relationship("Detection", back_populates="video", cascade="all, delete-orphan")
    activity_segments = relationship("ActivitySegment", back_populates="video", cascade="all, delete-orphan")
    transcript = relationship("Transcript", back_populates="video", uselist=False, cascade="all, delete-orphan")
    annotations = relationship("Annotation", back_populates="video", uselist=False, cascade="all, delete-orphan")
    tags = relationship("VideoTag", back_populates="video", cascade="all, delete-orphan")
    comments = relationship("Comment", back_populates="video", cascade="all, delete-orphan")


class VideoEmbedding(Base):
    __tablename__ = "video_embeddings"

    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    embedding = Column(Vector(512), nullable=False)
    model_name = Column(String(64), nullable=False, default="ViT-B-32")
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    video = relationship("Video", back_populates="embeddings")

    __table_args__ = (UniqueConstraint("video_id", "model_name"),)


class Keyframe(Base):
    __tablename__ = "keyframes"

    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    frame_num = Column(Integer, nullable=False)
    timestamp = Column(Float, nullable=False)
    storage_key = Column(Text, nullable=False)
    embedding = Column(Vector(512))
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    video = relationship("Video", back_populates="keyframes")
    detections = relationship("Detection", back_populates="keyframe", cascade="all, delete-orphan")


class Detection(Base):
    __tablename__ = "detections"

    id = Column(Integer, primary_key=True)
    keyframe_id = Column(Integer, ForeignKey("keyframes.id", ondelete="CASCADE"), nullable=False)
    video_id = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    label = Column(String(128), nullable=False)
    confidence = Column(Float, nullable=False)
    bbox_x = Column(Float, nullable=False)
    bbox_y = Column(Float, nullable=False)
    bbox_w = Column(Float, nullable=False)
    bbox_h = Column(Float, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    keyframe = relationship("Keyframe", back_populates="detections")
    video = relationship("Video", back_populates="detections")


class ActivitySegment(Base):
    __tablename__ = "activity_segments"

    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    activity_type = Column(String(128), nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    confidence = Column(Float, nullable=False)
    metadata_ = Column("metadata", JSONB, default={})
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    video = relationship("Video", back_populates="activity_segments")


class Transcript(Base):
    __tablename__ = "transcripts"

    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, unique=True)
    full_text = Column(Text, nullable=False)
    language = Column(String(16))
    model_name = Column(String(64), nullable=False, default="base")
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    video = relationship("Video", back_populates="transcript")
    segments = relationship("TranscriptSegment", back_populates="transcript", cascade="all, delete-orphan")


class TranscriptSegment(Base):
    __tablename__ = "transcript_segments"

    id = Column(Integer, primary_key=True)
    transcript_id = Column(Integer, ForeignKey("transcripts.id", ondelete="CASCADE"), nullable=False)
    video_id = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    start_time = Column(Float, nullable=False)
    end_time = Column(Float, nullable=False)
    text = Column(Text, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    transcript = relationship("Transcript", back_populates="segments")


class Annotation(Base):
    __tablename__ = "annotations"

    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False, unique=True)
    data = Column(JSONB, nullable=False, default={})
    version = Column(Integer, nullable=False, default=1)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    video = relationship("Video", back_populates="annotations")


class VideoTag(Base):
    __tablename__ = "video_tags"

    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    tag = Column(String(128), nullable=False)
    source = Column(String(32), nullable=False, default="auto")
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    video = relationship("Video", back_populates="tags")

    __table_args__ = (UniqueConstraint("video_id", "tag"),)


class IngestionTask(Base):
    __tablename__ = "ingestion_tasks"

    id = Column(Integer, primary_key=True)
    query = Column(Text, nullable=False)
    platform = Column(String(32), nullable=False, default="youtube")
    max_results = Column(Integer, nullable=False, default=10)
    status = Column(String(32), nullable=False, default="pending")
    progress = Column(Float, nullable=False, default=0.0)
    total_videos = Column(Integer, default=0)
    processed_videos = Column(Integer, default=0)
    error_message = Column(Text)
    current_step = Column(Text)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    videos = relationship("Video", back_populates="task")


class Dataset(Base):
    __tablename__ = "datasets"

    id = Column(Integer, primary_key=True)
    name = Column(String(256), nullable=False)
    description = Column(Text)
    filters = Column(JSONB, default={})
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())
    updated_at = Column(TIMESTAMP(timezone=True), server_default=func.now(), onupdate=func.now())

    videos = relationship("DatasetVideo", back_populates="dataset", cascade="all, delete-orphan")


class DatasetVideo(Base):
    __tablename__ = "dataset_videos"

    id = Column(Integer, primary_key=True)
    dataset_id = Column(Integer, ForeignKey("datasets.id", ondelete="CASCADE"), nullable=False)
    video_id = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    added_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    dataset = relationship("Dataset", back_populates="videos")

    __table_args__ = (UniqueConstraint("dataset_id", "video_id"),)


class DuplicatePair(Base):
    __tablename__ = "duplicate_pairs"

    id = Column(Integer, primary_key=True)
    video_id_a = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    video_id_b = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    similarity = Column(Float, nullable=False)
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    __table_args__ = (
        UniqueConstraint("video_id_a", "video_id_b"),
        CheckConstraint("video_id_a < video_id_b"),
    )


class Comment(Base):
    __tablename__ = "comments"

    id = Column(Integer, primary_key=True)
    video_id = Column(Integer, ForeignKey("videos.id", ondelete="CASCADE"), nullable=False)
    comment_id = Column(String(256))
    username = Column(String(256))
    text = Column(Text, nullable=False)
    like_count = Column(Integer, default=0)
    reply_count = Column(Integer, default=0)
    user_region = Column(String(16))
    language = Column(String(16))
    sentiment = Column(JSONB)
    posted_at = Column(TIMESTAMP(timezone=True))
    created_at = Column(TIMESTAMP(timezone=True), server_default=func.now())

    video = relationship("Video", back_populates="comments")


# ── Engine & Session ────────────────────────────────────────────

engine = create_async_engine(settings.database_url, echo=False, pool_size=10, max_overflow=20)

AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

# Thread-local storage for per-thread engines (used by background task queue)
_thread_local = threading.local()


def _create_thread_engine():
    """Create a dedicated async engine for the current thread's event loop."""
    _thread_local.engine = create_async_engine(
        settings.database_url, echo=False, pool_size=5, max_overflow=10
    )
    _thread_local.session_factory = sessionmaker(
        _thread_local.engine, class_=AsyncSession, expire_on_commit=False
    )


async def _dispose_thread_engine():
    """Dispose the current thread's engine."""
    eng = getattr(_thread_local, "engine", None)
    if eng:
        await eng.dispose()


@asynccontextmanager
async def get_session():
    # Use thread-local engine if available (background tasks), otherwise main engine
    factory = getattr(_thread_local, "session_factory", None) or AsyncSessionLocal
    async with factory() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
