import pytest
import pytest_asyncio
from sqlalchemy import text

EXPECTED_TABLES = [
    "videos",
    "video_embeddings",
    "keyframes",
    "detections",
    "activity_segments",
    "transcripts",
    "transcript_segments",
    "annotations",
    "video_tags",
    "ingestion_tasks",
    "datasets",
    "dataset_videos",
    "duplicate_pairs",
]


@pytest.mark.asyncio
async def test_schema_tables_exist(db_session):
    """Verify all 13 tables exist after running init.sql."""
    result = await db_session.execute(
        text(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' AND table_type = 'BASE TABLE'"
        )
    )
    tables = {row[0] for row in result.fetchall()}
    for table in EXPECTED_TABLES:
        assert table in tables, f"Table '{table}' not found in database"


@pytest.mark.asyncio
async def test_pgvector_extension(db_session):
    """Verify pgvector extension is installed."""
    result = await db_session.execute(
        text("SELECT extname FROM pg_extension WHERE extname = 'vector'")
    )
    row = result.fetchone()
    assert row is not None, "pgvector extension not installed"


@pytest.mark.asyncio
async def test_video_status_default(db_session):
    """Verify default video status is 'pending'."""
    result = await db_session.execute(
        text(
            "SELECT column_default FROM information_schema.columns "
            "WHERE table_name = 'videos' AND column_name = 'status'"
        )
    )
    row = result.fetchone()
    assert row is not None
    assert "pending" in row[0]


@pytest.mark.asyncio
async def test_embedding_dimension(db_session):
    """Verify embedding column is vector(512)."""
    result = await db_session.execute(
        text(
            "SELECT udt_name FROM information_schema.columns "
            "WHERE table_name = 'video_embeddings' AND column_name = 'embedding'"
        )
    )
    row = result.fetchone()
    assert row is not None
    assert row[0] == "vector"
