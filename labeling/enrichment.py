import logging
import os
import tempfile
from typing import List, Optional

from sqlalchemy import select, update

from config.database import get_session, Video, Keyframe, Transcript
from config.storage import storage

logger = logging.getLogger(__name__)

# Lazy-loaded singletons
_ocr_reader = None
AI_GENERATED_SIGNALS = [
    "#aiart", "#aigc", "#aigenerated", "#aiimage", "#aivideo",
    "#midjourney", "#stablediffusion", "#dalle", "#sora",
    "made with ai", "ai generated", "ai-generated", "created by ai",
    "generated with", "#deepfake",
]


def _get_ocr_reader():
    global _ocr_reader
    if _ocr_reader is None:
        import easyocr
        _ocr_reader = easyocr.Reader(["en"], gpu=False)
    return _ocr_reader


def _detect_language(text: str) -> Optional[str]:
    if not text or len(text.strip()) < 10:
        return None
    try:
        from langdetect import detect
        return detect(text)
    except Exception:
        return None


class EnrichmentService:
    async def enrich_video(self, video_id: int):
        """Run language detection, OCR on keyframes, and AI-generated flag."""
        async with get_session() as session:
            video = await session.get(Video, video_id)
            if not video:
                return

            caption_text = " ".join(filter(None, [video.title, video.description]))
            platform = video.platform
            hashtags = video.hashtags or []

            # Get transcript text
            trans_result = await session.execute(
                select(Transcript).where(Transcript.video_id == video_id)
            )
            transcript = trans_result.scalars().first()
            transcript_text = transcript.full_text if transcript else ""

            # Get keyframe storage keys
            kf_result = await session.execute(
                select(Keyframe.storage_key).where(Keyframe.video_id == video_id)
            )
            keyframe_keys = [row[0] for row in kf_result.fetchall()]

        # 1. Language detection
        language_info = {}
        caption_lang = _detect_language(caption_text)
        if caption_lang:
            language_info["caption"] = caption_lang
        transcript_lang = _detect_language(transcript_text)
        if transcript_lang:
            language_info["transcript"] = transcript_lang

        # 2. OCR on keyframes for sticker/overlay text
        sticker_texts = []
        if platform == "instagram" and keyframe_keys:
            sticker_texts = await self._ocr_keyframes(keyframe_keys)
            if sticker_texts:
                sticker_lang = _detect_language(" ".join(sticker_texts))
                if sticker_lang:
                    language_info["sticker"] = sticker_lang

        # 3. AI-generated detection (heuristic)
        is_ai_generated = self._detect_ai_generated(
            caption_text, hashtags, transcript_text
        )

        # Update video record
        update_values = {}
        if language_info:
            update_values["language"] = language_info
        if sticker_texts:
            update_values["sticker_texts"] = sticker_texts
        if is_ai_generated:
            update_values["is_ai_generated"] = True

        if update_values:
            async with get_session() as session:
                await session.execute(
                    update(Video).where(Video.id == video_id).values(**update_values)
                )

        logger.info(
            f"Video {video_id}: enrichment done — "
            f"lang={language_info}, stickers={len(sticker_texts)}, ai_gen={is_ai_generated}"
        )

    async def _ocr_keyframes(self, keyframe_keys: List[str], max_frames: int = 5) -> List[str]:
        """Run OCR on a sample of keyframes to extract overlay/sticker text."""
        reader = _get_ocr_reader()
        all_texts = []
        tmpdir = tempfile.mkdtemp(prefix="forgeindex_ocr_")

        try:
            for key in keyframe_keys[:max_frames]:
                local_path = os.path.join(tmpdir, os.path.basename(key))
                try:
                    storage.download_file(key, local_path)
                    results = reader.readtext(local_path, detail=0)
                    # Filter out very short / noisy detections
                    texts = [t.strip() for t in results if len(t.strip()) >= 3]
                    all_texts.extend(texts)
                except Exception as e:
                    logger.debug(f"OCR failed for {key}: {e}")
                finally:
                    if os.path.exists(local_path):
                        os.unlink(local_path)
        finally:
            try:
                os.rmdir(tmpdir)
            except OSError:
                pass

        # Deduplicate while preserving order
        seen = set()
        unique_texts = []
        for t in all_texts:
            normalized = t.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                unique_texts.append(t)
        return unique_texts

    def _detect_ai_generated(
        self, caption: str, hashtags: list, transcript: str
    ) -> bool:
        """Heuristic check for AI-generated content based on text signals."""
        corpus = (caption + " " + transcript + " " + " ".join(
            f"#{h}" for h in hashtags
        )).lower()

        return any(signal in corpus for signal in AI_GENERATED_SIGNALS)
