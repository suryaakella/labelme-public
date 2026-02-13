import logging
from datetime import datetime
from typing import Optional

import numpy as np
from sqlalchemy import select, update, func

from config.database import (
    get_session, Video, VideoEmbedding, Keyframe, Detection,
    Transcript, Annotation, Comment,
)

logger = logging.getLogger(__name__)

# Lazy-loaded VADER singleton
_sentiment_analyzer = None


def _get_sentiment_analyzer():
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
        _sentiment_analyzer = SentimentIntensityAnalyzer()
    return _sentiment_analyzer


# Brand safety keyword lists by severity
_BRAND_SAFETY_KEYWORDS = {
    "high": [
        "terrorism", "extremist", "child abuse", "hate speech", "self harm",
    ],
    "medium": [
        "drugs", "nsfw", "violence", "gore", "gambling", "scam",
    ],
    "low": [
        "alcohol", "smoking", "profanity", "clickbait",
    ],
}

_SEVERITY_PENALTIES = {"high": 0.30, "medium": 0.15, "low": 0.05}

# YOLO/COCO detection labels relevant to brand safety
_DETECTION_SAFETY_FLAGS = {
    "knife": "medium",
    "scissors": "medium",
    "wine glass": "low",
    "bottle": "low",
}

# Engagement tier thresholds (calibrated for Instagram)
_ENGAGEMENT_TIERS = [
    (0.10, "viral"),
    (0.06, "excellent"),
    (0.03, "good"),
    (0.01, "average"),
]


class AnalyticsService:
    def __init__(self):
        self._tag_embeddings_cache: dict[str, np.ndarray] = {}

    async def analyze_video(self, video_id: int):
        """Run all 5 analytics sub-analyses and store results."""
        try:
            sentiment = await self._analyze_comment_sentiment(video_id)
            engagement = await self._compute_engagement_benchmarks(video_id)
            categories = await self._categorize_content(video_id)
            safety = await self._compute_brand_safety(video_id)
            ai_gen = await self._analyze_ai_generated(video_id)

            analytics = {
                "version": 1,
                "computed_at": datetime.utcnow().isoformat() + "Z",
                "comment_sentiment": sentiment,
                "engagement": engagement,
                "content_categories": categories,
                "brand_safety": safety,
                "ai_generated": ai_gen,
            }

            # Write analytics JSONB + update is_ai_generated flag
            async with get_session() as session:
                await session.execute(
                    update(Video).where(Video.id == video_id).values(
                        analytics=analytics,
                        is_ai_generated=ai_gen["is_ai_generated"],
                    )
                )

            # Patch analytics summary into annotation JSONB
            async with get_session() as session:
                ann = (await session.execute(
                    select(Annotation).where(Annotation.video_id == video_id)
                )).scalars().first()
                if ann and ann.data:
                    updated = dict(ann.data)
                    updated["analytics"] = {
                        "comment_sentiment": sentiment,
                        "engagement": engagement,
                        "content_categories": categories,
                        "brand_safety": {"score": safety["score"], "tier": safety["tier"]},
                        "ai_generated": {
                            "is_ai_generated": ai_gen["is_ai_generated"],
                            "confidence": ai_gen["confidence"],
                        },
                    }
                    ann.data = updated
                    ann.version = (ann.version or 1) + 1

            logger.info(f"Video {video_id}: analytics completed")

        except Exception as e:
            logger.error(f"Video {video_id}: analytics failed — {e}")
            raise

    # ── 2A. Comment Sentiment ───────────────────────────────────────

    async def _analyze_comment_sentiment(self, video_id: int) -> dict:
        analyzer = _get_sentiment_analyzer()

        async with get_session() as session:
            result = await session.execute(
                select(Comment).where(Comment.video_id == video_id)
            )
            comments = result.scalars().all()

        if not comments:
            return {
                "total_analyzed": 0,
                "skipped_non_english": 0,
                "distribution": {"positive": 0, "neutral": 0, "negative": 0},
                "avg_compound": 0.0,
                "top_positive": None,
                "top_negative": None,
            }

        analyzed = []
        skipped = 0

        for comment in comments:
            # Language filter: skip non-English
            lang = self._detect_language(comment.text)
            if lang and lang != "en":
                skipped += 1
                continue

            scores = analyzer.polarity_scores(comment.text)
            compound = scores["compound"]

            if compound >= 0.05:
                label = "positive"
            elif compound <= -0.05:
                label = "negative"
            else:
                label = "neutral"

            sentiment_data = {
                "compound": round(compound, 4),
                "positive": round(scores["pos"], 4),
                "negative": round(scores["neg"], 4),
                "neutral": round(scores["neu"], 4),
                "label": label,
            }

            analyzed.append((comment, sentiment_data))

        # Store per-comment sentiment
        if analyzed:
            async with get_session() as session:
                for comment, sentiment_data in analyzed:
                    await session.execute(
                        update(Comment)
                        .where(Comment.id == comment.id)
                        .values(sentiment=sentiment_data)
                    )

        # Aggregate
        distribution = {"positive": 0, "neutral": 0, "negative": 0}
        compounds = []
        top_positive = None
        top_negative = None
        best_pos = -2.0
        best_neg = 2.0

        for comment, sent in analyzed:
            distribution[sent["label"]] += 1
            compounds.append(sent["compound"])
            if sent["compound"] > best_pos:
                best_pos = sent["compound"]
                top_positive = {"text": comment.text[:200], "compound": sent["compound"]}
            if sent["compound"] < best_neg:
                best_neg = sent["compound"]
                top_negative = {"text": comment.text[:200], "compound": sent["compound"]}

        avg_compound = round(float(np.mean(compounds)), 4) if compounds else 0.0

        return {
            "total_analyzed": len(analyzed),
            "skipped_non_english": skipped,
            "distribution": distribution,
            "avg_compound": avg_compound,
            "top_positive": top_positive,
            "top_negative": top_negative,
        }

    def _detect_language(self, text: str) -> Optional[str]:
        if not text or len(text.strip()) < 10:
            return None
        try:
            from langdetect import detect
            return detect(text)
        except Exception:
            return None

    # ── 2B. Engagement Benchmarking ─────────────────────────────────

    async def _compute_engagement_benchmarks(self, video_id: int) -> dict:
        async with get_session() as session:
            video = await session.get(Video, video_id)
            if not video or not video.engagement:
                return {}

        eng = video.engagement or {}
        likes = eng.get("likes", 0) or 0
        comments_count = eng.get("comments", 0) or 0
        shares = eng.get("shares", 0) or 0
        saves = eng.get("saves", 0) or 0
        plays = eng.get("plays", 0) or 0
        followers = video.creator_followers or 0

        # Compute engagement rates
        denominator = plays if plays > 0 else (followers if followers > 0 else 0)
        if denominator == 0:
            return {}

        total_interactions = likes + comments_count + shares + saves
        engagement_rate = round(total_interactions / denominator, 6)
        like_rate = round(likes / denominator, 6)
        comment_rate = round(comments_count / denominator, 6)
        share_rate = round(shares / denominator, 6)
        save_rate = round(saves / denominator, 6)

        rates = {
            "like_rate": like_rate,
            "comment_rate": comment_rate,
            "share_rate": share_rate,
            "save_rate": save_rate,
            "engagement_rate": engagement_rate,
        }

        # Benchmarks
        vs_creator = await self._benchmark_vs_creator(
            video_id, video.creator_username, video.platform, engagement_rate
        )
        vs_platform = await self._benchmark_vs_platform(
            video_id, video.platform, engagement_rate
        )

        # Performance tier
        tier = "below_average"
        for threshold, label in _ENGAGEMENT_TIERS:
            if engagement_rate >= threshold:
                tier = label
                break

        return {
            "rates": rates,
            "benchmarks": {
                "vs_creator": vs_creator,
                "vs_platform": vs_platform,
            },
            "performance_tier": tier,
        }

    async def _benchmark_vs_creator(
        self, video_id: int, creator_username: Optional[str],
        platform: str, engagement_rate: float,
    ) -> dict:
        if not creator_username:
            return {"verdict": "insufficient_data", "sample_size": 0}

        async with get_session() as session:
            result = await session.execute(
                select(Video)
                .where(
                    Video.creator_username == creator_username,
                    Video.platform == platform,
                    Video.id != video_id,
                    Video.engagement.isnot(None),
                    Video.status == "completed",
                )
            )
            other_videos = result.scalars().all()

        if len(other_videos) < 3:
            return {"verdict": "insufficient_data", "sample_size": len(other_videos)}

        other_rates = []
        for v in other_videos:
            e = v.engagement or {}
            total = (e.get("likes", 0) or 0) + (e.get("comments", 0) or 0) + \
                    (e.get("shares", 0) or 0) + (e.get("saves", 0) or 0)
            denom = (e.get("plays", 0) or 0) or (v.creator_followers or 0)
            if denom > 0:
                other_rates.append(total / denom)

        if not other_rates:
            return {"verdict": "insufficient_data", "sample_size": 0}

        creator_avg = float(np.mean(other_rates))
        ratio = round(engagement_rate / creator_avg, 2) if creator_avg > 0 else 0.0

        if ratio >= 1.5:
            verdict = "above_average"
        elif ratio >= 0.75:
            verdict = "average"
        else:
            verdict = "below_average"

        return {
            "ratio": ratio,
            "sample_size": len(other_rates),
            "creator_avg": round(creator_avg, 6),
            "verdict": verdict,
        }

    async def _benchmark_vs_platform(
        self, video_id: int, platform: str, engagement_rate: float,
    ) -> dict:
        async with get_session() as session:
            result = await session.execute(
                select(Video)
                .where(
                    Video.platform == platform,
                    Video.id != video_id,
                    Video.engagement.isnot(None),
                    Video.status == "completed",
                )
            )
            all_videos = result.scalars().all()

        platform_rates = []
        for v in all_videos:
            e = v.engagement or {}
            total = (e.get("likes", 0) or 0) + (e.get("comments", 0) or 0) + \
                    (e.get("shares", 0) or 0) + (e.get("saves", 0) or 0)
            denom = (e.get("plays", 0) or 0) or (v.creator_followers or 0)
            if denom > 0:
                platform_rates.append(total / denom)

        if not platform_rates:
            return {"verdict": "insufficient_data", "sample_size": 0}

        platform_avg = float(np.median(platform_rates))
        ratio = round(engagement_rate / platform_avg, 2) if platform_avg > 0 else 0.0

        if ratio >= 1.5:
            verdict = "above_average"
        elif ratio >= 0.75:
            verdict = "average"
        else:
            verdict = "below_average"

        return {
            "ratio": ratio,
            "sample_size": len(platform_rates),
            "platform_avg": round(platform_avg, 6),
            "verdict": verdict,
        }

    # ── 2C. Content Categorization (CLIP) ───────────────────────────

    async def _categorize_content(self, video_id: int) -> list:
        from labeling.annotator import TAG_CANDIDATES
        from pipeline.embeddings import CLIPEmbedder

        # Load video embedding
        async with get_session() as session:
            emb_result = await session.execute(
                select(VideoEmbedding).where(VideoEmbedding.video_id == video_id)
            )
            video_emb = emb_result.scalars().first()

        if not video_emb:
            return []

        vid_vec = np.array(video_emb.embedding, dtype=np.float32)

        # Get or compute tag embeddings (cached)
        if not self._tag_embeddings_cache:
            embedder = CLIPEmbedder()
            for tag, desc in TAG_CANDIDATES.items():
                self._tag_embeddings_cache[tag] = embedder.embed_text(desc)

        # Compute similarities
        scored = []
        for tag, tag_vec in self._tag_embeddings_cache.items():
            similarity = float(np.dot(vid_vec, tag_vec))
            if similarity > 0.20:
                scored.append({"category": tag, "confidence": round(similarity, 4)})

        # Top 5 by confidence
        scored.sort(key=lambda x: x["confidence"], reverse=True)
        return scored[:5]

    # ── 2D. Brand Safety ────────────────────────────────────────────

    async def _compute_brand_safety(self, video_id: int) -> dict:
        # Gather text sources
        async with get_session() as session:
            video = await session.get(Video, video_id)
            if not video:
                return {"score": 1.0, "tier": "safe", "flags": []}

            trans_result = await session.execute(
                select(Transcript).where(Transcript.video_id == video_id)
            )
            transcript = trans_result.scalars().first()

            comment_result = await session.execute(
                select(Comment.text).where(Comment.video_id == video_id)
            )
            comment_texts = [row[0] for row in comment_result.fetchall()]

            det_result = await session.execute(
                select(Detection.label).where(Detection.video_id == video_id).distinct()
            )
            detection_labels = [row[0] for row in det_result.fetchall()]

        # Build text corpus for keyword scanning
        text_sources = {
            "title": (video.title or "").lower(),
            "description": (video.description or "").lower(),
            "transcript": (transcript.full_text if transcript else "").lower(),
            "stickers": " ".join(video.sticker_texts or []).lower(),
            "comments": " ".join(comment_texts).lower(),
        }

        flags = []
        seen_keywords = set()
        score = 1.0

        # Scan text for keywords
        for severity, keywords in _BRAND_SAFETY_KEYWORDS.items():
            penalty = _SEVERITY_PENALTIES[severity]
            for keyword in keywords:
                if keyword in seen_keywords:
                    continue
                for source_name, text in text_sources.items():
                    if keyword in text:
                        seen_keywords.add(keyword)
                        score -= penalty
                        flags.append({
                            "source": source_name,
                            "keyword": keyword,
                            "severity": severity,
                        })
                        break

        # Check YOLO detections
        for label in detection_labels:
            label_lower = label.lower()
            if label_lower in _DETECTION_SAFETY_FLAGS:
                severity = _DETECTION_SAFETY_FLAGS[label_lower]
                penalty = _SEVERITY_PENALTIES[severity]
                score -= penalty
                flags.append({
                    "source": "detection",
                    "keyword": label_lower,
                    "severity": severity,
                })

        score = round(max(0.0, score), 2)

        if score >= 0.85:
            tier = "safe"
        elif score >= 0.60:
            tier = "low_risk"
        elif score >= 0.30:
            tier = "medium_risk"
        else:
            tier = "high_risk"

        return {"score": score, "tier": tier, "flags": flags}

    # ── 2E. AI-Generated Detection ──────────────────────────────────

    async def _analyze_ai_generated(self, video_id: int) -> dict:
        from labeling.enrichment import AI_GENERATED_SIGNALS

        # Check keyword heuristic
        async with get_session() as session:
            video = await session.get(Video, video_id)
            if not video:
                return {
                    "is_ai_generated": False, "confidence": 0.0,
                    "signals": {"keyword_match": False},
                }

            trans_result = await session.execute(
                select(Transcript).where(Transcript.video_id == video_id)
            )
            transcript = trans_result.scalars().first()

        caption = " ".join(filter(None, [video.title, video.description]))
        hashtags = video.hashtags or []
        transcript_text = transcript.full_text if transcript else ""

        corpus = (
            caption + " " + transcript_text + " " +
            " ".join(f"#{h}" for h in hashtags)
        ).lower()

        keyword_match = any(signal in corpus for signal in AI_GENERATED_SIGNALS)

        # Frame embedding consistency analysis
        async with get_session() as session:
            kf_result = await session.execute(
                select(Keyframe.embedding)
                .where(Keyframe.video_id == video_id, Keyframe.embedding.isnot(None))
            )
            embeddings = [row[0] for row in kf_result.fetchall()]

        mean_cos_sim = None
        embedding_variance = None
        confidence = 0.0

        if keyword_match:
            confidence += 0.50

        if len(embeddings) >= 3:
            vecs = np.array(embeddings, dtype=np.float32)
            # Compute pairwise cosine similarity (vectors are already normalized)
            sim_matrix = vecs @ vecs.T
            n = len(vecs)
            # Upper triangle (excluding diagonal)
            upper = sim_matrix[np.triu_indices(n, k=1)]
            mean_cos_sim = round(float(np.mean(upper)), 4)
            embedding_variance = round(float(np.var(upper)), 6)

            if mean_cos_sim > 0.95:
                confidence += 0.30
            elif mean_cos_sim > 0.90:
                confidence += 0.15

            if embedding_variance < 0.001:
                confidence += 0.20

        is_ai_generated = confidence >= 0.40

        signals = {"keyword_match": keyword_match}
        if embedding_variance is not None:
            signals["embedding_variance"] = embedding_variance
        if mean_cos_sim is not None:
            signals["mean_cosine_similarity"] = mean_cos_sim

        return {
            "is_ai_generated": is_ai_generated,
            "confidence": round(confidence, 2),
            "signals": signals,
        }
