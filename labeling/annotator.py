import logging
from collections import Counter, defaultdict
from typing import List, Set

import numpy as np
from sqlalchemy import select, func

from config.database import (
    get_session, Video, VideoEmbedding, Detection, ActivitySegment,
    Transcript, TranscriptSegment, Annotation, VideoTag, Keyframe,
)
from pipeline.embeddings import CLIPEmbedder

logger = logging.getLogger(__name__)

# CLIP zero-shot tag candidates: tag name → descriptive prompt for CLIP
# Add or remove entries freely — no keyword lists to maintain
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

# ── Annotation quality-filter constants ──────────────────────────
ANNOTATION_MIN_AVG_CONFIDENCE = 0.50
ANNOTATION_MIN_KEYFRAME_COUNT = 2
ANNOTATION_MIN_KEYFRAME_RATIO = 0.05

SCREENCAST_CONFIDENCE_FLOOR = 0.75
SCREENCAST_CATEGORIES = frozenset({"tutorial", "tech", "gaming", "education"})
SCREENCAST_HALLUCINATION_LABELS = frozenset({
    # Animals
    "bird", "cat", "dog", "horse", "sheep", "cow", "elephant", "bear",
    "zebra", "giraffe",
    # Vehicles
    "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    # Outdoor objects
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench",
    # Food / kitchen
    "banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog",
    "pizza", "donut", "cake",
    # Sports / recreation
    "frisbee", "skis", "snowboard", "sports ball", "kite", "baseball bat",
    "baseball glove", "skateboard", "surfboard", "tennis racket",
    # Household / misc
    "potted plant", "bed", "dining table", "toilet", "couch", "oven",
    "toaster", "sink", "refrigerator", "umbrella", "handbag", "suitcase",
    "tie", "wine glass", "vase",
})

TAG_MIN_AVG_CONFIDENCE = 0.55


class AnnotationService:
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
        async with get_session() as session:
            video = await session.get(Video, video_id)
            if not video:
                return

            # Gather all data
            det_result = await session.execute(
                select(Detection).where(Detection.video_id == video_id)
            )
            detections = det_result.scalars().all()

            act_result = await session.execute(
                select(ActivitySegment).where(ActivitySegment.video_id == video_id)
            )
            activities = act_result.scalars().all()

            trans_result = await session.execute(
                select(Transcript).where(Transcript.video_id == video_id)
            )
            transcript = trans_result.scalars().first()

            seg_result = await session.execute(
                select(TranscriptSegment).where(TranscriptSegment.video_id == video_id)
            )
            segments = seg_result.scalars().all()

            # Fetch stored video embedding for CLIP zero-shot tagging
            emb_result = await session.execute(
                select(VideoEmbedding).where(VideoEmbedding.video_id == video_id)
            )
            video_embedding = emb_result.scalars().first()

            # Count keyframes for quality filtering
            kf_count_result = await session.execute(
                select(func.count(Keyframe.id)).where(Keyframe.video_id == video_id)
            )
            total_keyframe_count = kf_count_result.scalar() or 0

        # Build annotation JSONB
        annotation_data = self._build_annotation(
            video, detections, activities, transcript, segments,
            video_embedding, total_keyframe_count,
        )

        # Compute auto-tags
        tags = self._compute_tags(video, video_embedding, detections, activities)

        # Store
        async with get_session() as session:
            annotation = Annotation(
                video_id=video_id,
                data=annotation_data,
                version=1,
            )
            session.add(annotation)

            for tag in tags:
                vt = VideoTag(video_id=video_id, tag=tag, source="auto")
                session.add(vt)

        logger.info(f"Video {video_id}: annotation built, {len(tags)} tags")

    def _classify_content_type(self, video_embedding) -> Set[str]:
        """Use CLIP tag embeddings to determine content categories."""
        if video_embedding is None:
            return set()
        vid_vec = np.array(video_embedding.embedding, dtype=np.float32)
        categories = set()
        for tag, tag_vec in self._get_tag_embeddings().items():
            similarity = float(np.dot(vid_vec, tag_vec))
            if similarity >= CLIP_TAG_THRESHOLD:
                categories.add(tag)
        return categories

    def _filter_detections_for_summary(
        self, detections, total_keyframe_count: int, content_categories: Set[str],
    ) -> List[dict]:
        """Multi-factor quality filter: only labels that pass all gates make it
        into the annotation summary.  Raw detections stay in DB untouched."""
        is_screencast = bool(content_categories & SCREENCAST_CATEGORIES)

        # Group detections by label
        by_label: dict[str, list] = defaultdict(list)
        for d in detections:
            by_label[d.label].append(d)

        results = []
        for label, dets in by_label.items():
            avg_conf = sum(d.confidence for d in dets) / len(dets)
            keyframe_ids = {d.keyframe_id for d in dets}
            kf_count = len(keyframe_ids)

            # Gate 1: minimum average confidence
            if avg_conf < ANNOTATION_MIN_AVG_CONFIDENCE:
                continue

            # Gate 2: minimum keyframe count
            if kf_count < ANNOTATION_MIN_KEYFRAME_COUNT:
                continue

            # Gate 3: minimum keyframe ratio
            if total_keyframe_count > 0 and (kf_count / total_keyframe_count) < ANNOTATION_MIN_KEYFRAME_RATIO:
                continue

            # Gate 4: screencast hallucination check
            if is_screencast and label in SCREENCAST_HALLUCINATION_LABELS:
                if avg_conf < SCREENCAST_CONFIDENCE_FLOOR:
                    continue

            results.append({
                "label": label,
                "count": len(dets),
                "avg_confidence": round(avg_conf, 3),
                "keyframe_count": kf_count,
            })

        # Sort by count descending, cap at 20
        results.sort(key=lambda x: x["count"], reverse=True)
        return results[:20]

    def _build_annotation(
        self, video, detections, activities, transcript, segments,
        video_embedding=None, total_keyframe_count: int = 0,
    ) -> dict:
        # Classify content type for context-aware filtering
        content_categories = self._classify_content_type(video_embedding)

        # Object summary — quality-filtered
        objects = self._filter_detections_for_summary(
            detections, total_keyframe_count, content_categories,
        )

        # Activity summary
        activity_list = [
            {
                "type": a.activity_type,
                "start": a.start_time,
                "end": a.end_time,
                "duration": round(a.end_time - a.start_time, 2),
                "confidence": a.confidence,
            }
            for a in activities
        ]

        # Transcript summary
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

        enrichment = {}
        if video.creator_username:
            enrichment["creator"] = {
                "username": video.creator_username,
                "id": video.creator_id,
                "followers": video.creator_followers,
            }
        if video.hashtags:
            enrichment["hashtags"] = video.hashtags
        if video.mentions:
            enrichment["mentions"] = video.mentions
        if video.engagement:
            enrichment["engagement"] = video.engagement
        if video.music_info:
            enrichment["music_info"] = video.music_info
        if video.language:
            enrichment["language"] = video.language
        if video.sticker_texts:
            enrichment["sticker_texts"] = video.sticker_texts
        if video.is_ad:
            enrichment["is_ad"] = True
        if video.is_ai_generated:
            enrichment["is_ai_generated"] = True
        if video.posted_at:
            enrichment["posted_at"] = video.posted_at.isoformat()

        return {
            "video_id": video.id,
            "title": video.title,
            "platform": video.platform,
            "duration_sec": video.duration_sec,
            "objects": objects,
            "activities": activity_list,
            "transcript": transcript_data,
            "enrichment": enrichment if enrichment else None,
            "content_type_hint": sorted(content_categories) if content_categories else None,
        }

    def _avg_conf(self, detections, label: str) -> float:
        confs = [d.confidence for d in detections if d.label == label]
        return round(sum(confs) / len(confs), 3) if confs else 0.0

    def _compute_tags(self, video, video_embedding, detections, activities) -> Set[str]:
        tags = set()

        # CLIP zero-shot tagging: compare video embedding against tag descriptions
        if video_embedding is not None:
            vid_vec = np.array(video_embedding.embedding, dtype=np.float32)
            for tag, tag_vec in self._get_tag_embeddings().items():
                similarity = float(np.dot(vid_vec, tag_vec))
                if similarity >= CLIP_TAG_THRESHOLD:
                    tags.add(tag)

        # Tags from activity types
        for a in activities:
            tags.add(a.activity_type)

        # Tags from detected objects — count >= 3 AND avg confidence >= threshold
        by_label: dict[str, list] = defaultdict(list)
        for d in detections:
            by_label[d.label].append(d)
        for label, dets in by_label.items():
            if len(dets) >= 3:
                avg_conf = sum(d.confidence for d in dets) / len(dets)
                if avg_conf >= TAG_MIN_AVG_CONFIDENCE:
                    tags.add(f"has_{label}")

        # Platform tag
        if video.platform:
            tags.add(f"platform:{video.platform}")

        return tags
