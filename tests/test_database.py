import pytest
from sqlalchemy import inspect

from config.database import (
    Base, Video, VideoEmbedding, Keyframe, Detection, ActivitySegment,
    Transcript, TranscriptSegment, Annotation, VideoTag,
    IngestionTask, Dataset, DatasetVideo, DuplicatePair,
)


def test_all_models_registered():
    model_classes = [
        Video, VideoEmbedding, Keyframe, Detection, ActivitySegment,
        Transcript, TranscriptSegment, Annotation, VideoTag,
        IngestionTask, Dataset, DatasetVideo, DuplicatePair,
    ]
    assert len(model_classes) == 13
    table_names = {cls.__tablename__ for cls in model_classes}
    assert len(table_names) == 13


def test_video_model_columns():
    mapper = inspect(Video)
    col_names = {c.key for c in mapper.column_attrs}
    required = {"id", "url", "platform", "status", "storage_key", "created_at"}
    assert required.issubset(col_names)


def test_video_embedding_dimension():
    mapper = inspect(VideoEmbedding)
    col_names = {c.key for c in mapper.column_attrs}
    assert "embedding" in col_names


def test_video_relationships():
    mapper = inspect(Video)
    rel_names = {r.key for r in mapper.relationships}
    expected = {"embeddings", "keyframes", "detections", "activity_segments", "transcript", "annotations", "tags"}
    assert expected.issubset(rel_names)
