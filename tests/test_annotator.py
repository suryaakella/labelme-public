from unittest.mock import MagicMock, patch

import numpy as np

from labeling.llm_annotator import (
    LLMAnnotationService,
    LLMAnnotation,
    SceneDescription,
    TAG_CANDIDATES,
    CLIP_TAG_THRESHOLD,
    _select_representative_keyframes,
)


def test_tag_candidates_are_descriptive():
    """Each tag candidate should have a meaningful CLIP-friendly description."""
    assert len(TAG_CANDIDATES) >= 10
    for tag, desc in TAG_CANDIDATES.items():
        assert len(desc) > 10, f"Tag '{tag}' description too short: '{desc}'"


def test_clip_tag_threshold():
    assert 0.1 <= CLIP_TAG_THRESHOLD <= 0.5


def test_select_representative_keyframes_few():
    """When fewer frames than max, return all."""
    kfs = [MagicMock(timestamp=i) for i in range(3)]
    selected = _select_representative_keyframes(kfs, max_frames=6)
    assert len(selected) == 3


def test_select_representative_keyframes_many():
    """When more frames than max, pick first + evenly spaced + last."""
    kfs = [MagicMock(timestamp=float(i)) for i in range(20)]
    selected = _select_representative_keyframes(kfs, max_frames=6)
    assert len(selected) <= 6
    assert selected[0] is kfs[0]
    assert selected[-1] is kfs[-1]


def test_llm_annotation_model():
    """LLMAnnotation Pydantic model accepts valid data."""
    ann = LLMAnnotation(
        summary="A cooking video",
        scenes=[
            SceneDescription(
                timestamp_sec=0.0,
                description="Chef prepares ingredients",
                objects_in_context=["chef", "cutting board"],
                activities=["chopping vegetables"],
            )
        ],
        primary_activities=["cooking"],
        content_tags=["cooking", "food"],
        visual_style="Close-up shots",
        transcript_context="Narrator explains recipe",
    )
    assert ann.summary == "A cooking video"
    assert len(ann.scenes) == 1
    assert ann.scenes[0].activities == ["chopping vegetables"]


@patch("labeling.llm_annotator.CLIPEmbedder")
def test_compute_tags_from_llm_and_clip(mock_embedder_cls):
    """Tags combine LLM content_tags, LLM activities, and CLIP zero-shot."""
    mock_embedder = MagicMock()
    mock_embedder_cls.return_value = mock_embedder

    matching_vec = np.array([1.0] + [0.0] * 511, dtype=np.float32)
    non_matching_vec = np.array([0.0] * 512, dtype=np.float32)
    mock_embedder.embed_text.side_effect = lambda desc: (
        matching_vec if "cooking" in desc else non_matching_vec
    )

    svc = LLMAnnotationService()

    video = MagicMock()
    video.platform = "instagram"

    video_embedding = MagicMock()
    video_embedding.embedding = matching_vec.tolist()

    llm_result = LLMAnnotation(
        summary="test",
        primary_activities=["street vending", "cooking"],
        content_tags=["Thai Food", "street-food"],
    )

    tags = svc._compute_tags(video, video_embedding, llm_result)

    # LLM content tags (lowercased, spaces to dashes)
    assert "thai-food" in tags
    assert "street-food" in tags

    # LLM activities with prefix
    assert "activity:street-vending" in tags
    assert "activity:cooking" in tags

    # CLIP zero-shot
    assert "cooking" in tags

    # Platform tag
    assert "platform:instagram" in tags


@patch("labeling.llm_annotator.CLIPEmbedder")
def test_compute_tags_no_embedding(mock_embedder_cls):
    """When no video embedding exists, CLIP tags are skipped gracefully."""
    mock_embedder_cls.return_value = MagicMock()

    svc = LLMAnnotationService()
    video = MagicMock()
    video.platform = "youtube"

    llm_result = LLMAnnotation(
        summary="test",
        primary_activities=["gaming"],
        content_tags=["fps", "shooter"],
    )

    tags = svc._compute_tags(video, None, llm_result)
    assert "platform:youtube" in tags
    assert "activity:gaming" in tags
    assert "fps" in tags
