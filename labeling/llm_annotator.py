"""LLM-powered multimodal video annotation using Gemini.

Sends representative keyframe images + transcript to a vision-capable LLM
and gets back structured, research-grade annotations describing what's
actually happening in the video.

Falls back to the old AnnotationService if GOOGLE_API_KEY is not set.
"""

import asyncio
import base64
import logging
import tempfile
import os
from collections import defaultdict
from typing import List, Optional, Set

import numpy as np
from pydantic import BaseModel, Field
from sqlalchemy import select, func

from config import settings
from config.database import (
    get_session, Video, VideoEmbedding, Detection, Keyframe,
    Transcript, TranscriptSegment, Annotation, VideoTag,
)
from config.storage import storage
from pipeline.embeddings import CLIPEmbedder

logger = logging.getLogger(__name__)


# ── Pydantic models for structured LLM output ────────────────────

class SceneDescription(BaseModel):
    timestamp_sec: float = Field(description="Approximate timestamp in seconds")
    description: str = Field(description="What is happening in this frame/scene")
    objects_in_context: List[str] = Field(
        default_factory=list,
        description="Objects visible, described in context (e.g. 'chef holding wok', not just 'person')",
    )
    activities: List[str] = Field(
        default_factory=list,
        description="Activities occurring (e.g. 'stir-frying vegetables')",
    )


class LLMAnnotation(BaseModel):
    summary: str = Field(description="2-4 sentence description of the entire video")
    scenes: List[SceneDescription] = Field(
        default_factory=list,
        description="Per-keyframe scene descriptions",
    )
    primary_activities: List[str] = Field(
        default_factory=list,
        description="Main activities in the video (e.g. ['cooking', 'street vending'])",
    )
    content_tags: List[str] = Field(
        default_factory=list,
        description="Content tags for categorization (e.g. ['cooking', 'thai-food', 'street-food'])",
    )
    visual_style: str = Field(
        default="",
        description="Brief description of visual/production style",
    )
    transcript_context: str = Field(
        default="",
        description="How the audio/narration relates to the visual content",
    )


# ── CLIP tag candidates (retained for supplementary tagging) ─────

TAG_CANDIDATES: dict[str, str] = {
    "tutorial": "a tutorial or how-to instructional video",
    "review": "a product review or unboxing video",
    "cooking": "a cooking or recipe video with food preparation",
    "sports": "a sports game, workout, or athletic activity",
    "music": "a music performance, concert, or music video",
    "travel": "a travel vlog or scenic destination video",
    "tech": "a technology, programming, or gadget video",
    "nature": "nature, wildlife, or outdoor scenery",
    "vlog": "a personal vlog or day-in-the-life video",
    "comedy": "a comedy skit, prank, or funny video",
    "gaming": "a video game playthrough or gaming stream",
    "education": "an educational lecture or explainer video",
    "fashion": "a fashion, beauty, or style video",
    "news": "a news report or current events coverage",
    "diy": "a do-it-yourself craft or home improvement project",
    "fitness": "a fitness routine, yoga, or exercise video",
    "art": "an art creation, painting, or drawing process",
    "automotive": "a car, motorcycle, or vehicle-related video",
    "pets": "a video featuring pets or domestic animals",
    "dance": "a dance performance or choreography video",
}

CLIP_TAG_THRESHOLD = 0.25

# ── Lazy-loaded LLM chain (singleton) ────────────────────────────

_annotation_chain = None


def _get_annotation_chain():
    global _annotation_chain
    if _annotation_chain is None:
        from langchain_google_genai import ChatGoogleGenerativeAI

        llm = ChatGoogleGenerativeAI(
            model=settings.annotation_llm_model,
            google_api_key=settings.google_api_key,
            max_output_tokens=settings.annotation_max_output_tokens,
            temperature=0,
        )
        _annotation_chain = llm.with_structured_output(LLMAnnotation)
    return _annotation_chain


# ── Helpers ──────────────────────────────────────────────────────

def _select_representative_keyframes(keyframes: list, max_frames: int = 6) -> list:
    """Pick 4-6 representative frames: first, last, evenly spaced from middle."""
    if len(keyframes) <= max_frames:
        return keyframes

    selected = [keyframes[0]]
    middle_count = max_frames - 2
    step = (len(keyframes) - 2) / (middle_count + 1)
    for i in range(1, middle_count + 1):
        idx = int(round(i * step))
        idx = max(1, min(idx, len(keyframes) - 2))
        if keyframes[idx] not in selected:
            selected.append(keyframes[idx])
    selected.append(keyframes[-1])
    return selected[:max_frames]


def _download_keyframe_to_base64(storage_key: str) -> Optional[str]:
    """Download a keyframe image from MinIO and return as base64 string."""
    try:
        with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as tmp:
            tmp_path = tmp.name
        storage.download_file(storage_key, tmp_path)
        with open(tmp_path, "rb") as f:
            data = f.read()
        return base64.b64encode(data).decode("utf-8")
    except Exception as e:
        logger.warning(f"Failed to download keyframe {storage_key}: {e}")
        return None
    finally:
        try:
            os.unlink(tmp_path)
        except OSError:
            pass


def _build_multimodal_messages(
    images_b64: List[dict],
    transcript_text: Optional[str],
    video_title: Optional[str],
    platform: Optional[str],
) -> list:
    """Build LangChain HumanMessage with images + text for Gemini."""
    from langchain_core.messages import SystemMessage, HumanMessage

    system_prompt = (
        "You are a research-grade video annotation system. You will be shown "
        "representative keyframe images from a video, in chronological order, "
        "along with the transcript if available.\n\n"
        "Your job:\n"
        "1. Describe what you ACTUALLY SEE in each frame\n"
        "2. Infer activities from visual content, not from object labels\n"
        "3. Write a comprehensive summary of the entire video\n"
        "4. Tag the content with relevant categories\n"
        "5. Describe the visual/production style\n\n"
        "Be specific and grounded. If you see a person cooking, say 'cooking' — "
        "don't say 'person detected near kitchen object'. Describe scenes as a "
        "researcher would annotate them."
    )

    content_parts = []

    # Add each keyframe image
    for img_info in images_b64:
        content_parts.append({
            "type": "text",
            "text": f"[Keyframe at {img_info['timestamp']:.1f}s]",
        })
        content_parts.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/jpeg;base64,{img_info['data']}"},
        })

    # Add transcript context
    context_text = ""
    if video_title:
        context_text += f"Video title: {video_title}\n"
    if platform:
        context_text += f"Platform: {platform}\n"
    if transcript_text:
        # Truncate to ~2000 chars to stay within token limits
        truncated = transcript_text[:2000]
        if len(transcript_text) > 2000:
            truncated += "... [truncated]"
        context_text += f"\nTranscript:\n{truncated}\n"

    if context_text:
        content_parts.append({"type": "text", "text": context_text})

    content_parts.append({
        "type": "text",
        "text": "Analyze these keyframes and provide a structured annotation.",
    })

    return [
        SystemMessage(content=system_prompt),
        HumanMessage(content=content_parts),
    ]


# ── Main Service ─────────────────────────────────────────────────

class LLMAnnotationService:
    def __init__(self):
        self._embedder = CLIPEmbedder()
        self._tag_embeddings: dict[str, np.ndarray] | None = None

    def _get_tag_embeddings(self) -> dict[str, np.ndarray]:
        if self._tag_embeddings is None:
            self._tag_embeddings = {
                tag: self._embedder.embed_text(desc)
                for tag, desc in TAG_CANDIDATES.items()
            }
        return self._tag_embeddings

    async def annotate_video(self, video_id: int):
        # If no API key, fall back to old annotation service
        if not settings.google_api_key:
            logger.warning(f"No GOOGLE_API_KEY, falling back to rule-based annotation for video {video_id}")
            from labeling.annotator import AnnotationService
            fallback = AnnotationService()
            await fallback.annotate_video(video_id)
            return

        async with get_session() as session:
            video = await session.get(Video, video_id)
            if not video:
                return

            # Fetch keyframes
            kf_result = await session.execute(
                select(Keyframe)
                .where(Keyframe.video_id == video_id)
                .order_by(Keyframe.timestamp)
            )
            keyframes = kf_result.scalars().all()

            # Fetch transcript
            trans_result = await session.execute(
                select(Transcript).where(Transcript.video_id == video_id)
            )
            transcript = trans_result.scalars().first()

            seg_result = await session.execute(
                select(TranscriptSegment).where(TranscriptSegment.video_id == video_id)
            )
            segments = seg_result.scalars().all()

            # Fetch video embedding for CLIP tagging
            emb_result = await session.execute(
                select(VideoEmbedding).where(VideoEmbedding.video_id == video_id)
            )
            video_embedding = emb_result.scalars().first()

        # Select representative keyframes
        selected_kfs = _select_representative_keyframes(keyframes)

        # Download and encode keyframe images
        images_b64 = []
        for kf in selected_kfs:
            b64 = _download_keyframe_to_base64(kf.storage_key)
            if b64:
                images_b64.append({"data": b64, "timestamp": kf.timestamp})

        if not images_b64:
            logger.warning(f"No keyframe images available for video {video_id}, falling back")
            from labeling.annotator import AnnotationService
            fallback = AnnotationService()
            await fallback.annotate_video(video_id)
            return

        # Build transcript text
        transcript_text = transcript.full_text if transcript else None

        # Call LLM
        try:
            messages = _build_multimodal_messages(
                images_b64, transcript_text, video.title, video.platform,
            )
            chain = _get_annotation_chain()
            loop = asyncio.get_event_loop()
            llm_result: LLMAnnotation = await loop.run_in_executor(
                None, lambda: chain.invoke(messages)
            )
        except Exception as e:
            logger.error(f"LLM annotation failed for video {video_id}: {e}", exc_info=True)
            from labeling.annotator import AnnotationService
            fallback = AnnotationService()
            await fallback.annotate_video(video_id)
            return

        # Build transcript data for annotation JSONB
        transcript_data = None
        if transcript:
            transcript_data = {
                "full_text": transcript.full_text,
                "language": transcript.language,
                "segments": [
                    {"start": s.start_time, "end": s.end_time, "text": s.text}
                    for s in segments
                ],
            }

        # Build platform context
        platform_context = {}
        if video.creator_username:
            platform_context["creator"] = {
                "username": video.creator_username,
                "id": video.creator_id,
                "followers": video.creator_followers,
            }
        if video.engagement:
            platform_context["engagement"] = video.engagement
        if video.hashtags:
            platform_context["hashtags"] = video.hashtags
        if video.mentions:
            platform_context["mentions"] = video.mentions
        if video.music_info:
            platform_context["music_info"] = video.music_info
        if video.is_ad:
            platform_context["is_ad"] = True
        if video.is_ai_generated:
            platform_context["is_ai_generated"] = True
        if video.posted_at:
            platform_context["posted_at"] = video.posted_at.isoformat()

        # Build final annotation JSONB
        annotation_data = {
            "version": 2,
            "summary": llm_result.summary,
            "scenes": [s.model_dump() for s in llm_result.scenes],
            "primary_activities": llm_result.primary_activities,
            "content_tags": llm_result.content_tags,
            "visual_style": llm_result.visual_style,
            "transcript_context": llm_result.transcript_context,
            "video_metadata": {
                "video_id": video.id,
                "title": video.title,
                "platform": video.platform,
                "duration_sec": video.duration_sec,
            },
            "platform_context": platform_context if platform_context else None,
            "transcript": transcript_data,
        }

        # Compute tags: LLM content_tags + LLM activities + CLIP zero-shot + platform tag
        tags = self._compute_tags(
            video, video_embedding, llm_result,
        )

        # Store
        async with get_session() as session:
            annotation = Annotation(
                video_id=video_id,
                data=annotation_data,
                version=2,
            )
            session.add(annotation)

            for tag in tags:
                vt = VideoTag(video_id=video_id, tag=tag, source="auto")
                session.add(vt)

        logger.info(f"Video {video_id}: LLM annotation built (v2), {len(tags)} tags")

    def _compute_tags(
        self,
        video,
        video_embedding,
        llm_result: LLMAnnotation,
    ) -> Set[str]:
        tags = set()

        # LLM content tags
        for tag in llm_result.content_tags:
            tags.add(tag.lower().replace(" ", "-"))

        # LLM activities as activity: prefixed tags
        for activity in llm_result.primary_activities:
            tags.add(f"activity:{activity.lower().replace(' ', '-')}")

        # CLIP zero-shot tagging
        if video_embedding is not None:
            vid_vec = np.array(video_embedding.embedding, dtype=np.float32)
            for tag, tag_vec in self._get_tag_embeddings().items():
                similarity = float(np.dot(vid_vec, tag_vec))
                if similarity >= CLIP_TAG_THRESHOLD:
                    tags.add(tag)

        # Platform tag
        if video.platform:
            tags.add(f"platform:{video.platform}")

        return tags
