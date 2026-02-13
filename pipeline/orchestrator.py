import logging
import os
import tempfile
from typing import List

from sqlalchemy import update

from config.database import get_session, Video, Keyframe, VideoEmbedding
from config.storage import storage
from pipeline.frames import KeyframeExtractor
from pipeline.embeddings import CLIPEmbedder
from pipeline.dedup import DeduplicationService
from pipeline.transcription import TranscriptionService

logger = logging.getLogger(__name__)

PIPELINE_STEPS = {
    1: "Extracting keyframes",
    2: "Generating CLIP embeddings",
    3: "Checking duplicates",
    4: "Transcribing audio (Whisper)",
    5: "Enriching metadata (language, OCR, AI detection)",
    6: "Detecting objects (YOLO)",
    7: "Building LLM annotations & tags",
    8: "Scanning for GDPR compliance",
    9: "Computing analytics (sentiment, engagement, brand safety)",
}
TOTAL_STEPS = len(PIPELINE_STEPS)


class PipelineOrchestrator:
    def __init__(self):
        self.frame_extractor = KeyframeExtractor()
        self.embedder = CLIPEmbedder()
        self.dedup = DeduplicationService()
        self.transcription = TranscriptionService()

    async def run(self, video_id: int):
        try:
            await self._set_status(video_id, "processing")

            # Download video from MinIO to temp file
            video_path = await self._download_video(video_id)
            try:
                # Step 1: Extract keyframes
                await self._set_step(video_id, 1)
                frames = self.frame_extractor.extract(video_path, video_id)
                await self._store_keyframes(video_id, frames)

                # Step 2: Generate embeddings
                await self._set_step(video_id, 2)
                await self._generate_embeddings(video_id, frames)

                # Step 3: Check for duplicates
                await self._set_step(video_id, 3)
                await self.dedup.check_duplicates(video_id)

                # Step 4: Transcribe
                await self._set_step(video_id, 4)
                await self.transcription.transcribe_and_store(video_id, video_path)

                # Step 5: Enrich (language detection + OCR + AI-gen flag)
                await self._set_step(video_id, 5)
                await self._run_enrichment(video_id)

                # Step 6: Label (detector + activity)
                await self._set_step(video_id, 6)
                await self._set_status(video_id, "labeling")
                await self._run_labeling(video_id)

                # Step 7: Annotate
                await self._set_step(video_id, 7)
                await self._run_annotation(video_id)

                # Step 8: GDPR PII scan
                await self._set_step(video_id, 8)
                gdpr_result = await self._run_gdpr_scan(video_id)

                # If GDPR blocked, stop pipeline — don't run analytics on quarantined data
                if gdpr_result.get("status") in ("blocked", "unverified", "error"):
                    logger.warning(f"Pipeline stopped for video {video_id}: GDPR {gdpr_result['status']}")
                    return

                # Step 9: Analytics (only runs on GDPR-clean videos)
                await self._set_step(video_id, 9)
                await self._run_analytics(video_id)

                await self._set_status(video_id, "completed", current_step="Pipeline complete")
                logger.info(f"Pipeline completed for video {video_id}")

            finally:
                # Cleanup temp video file
                try:
                    os.unlink(video_path)
                    os.rmdir(os.path.dirname(video_path))
                except OSError:
                    pass

        except Exception as e:
            logger.error(f"Pipeline failed for video {video_id}: {e}")
            await self._set_status(video_id, "failed", error_message=str(e), current_step=None)

    async def _download_video(self, video_id: int) -> str:
        async with get_session() as session:
            video = await session.get(Video, video_id)
            if not video or not video.storage_key:
                raise ValueError(f"Video {video_id} not found or has no storage key")

            tmpdir = tempfile.mkdtemp(prefix="forgeindex_pipeline_")
            local_path = os.path.join(tmpdir, "video.mp4")
            storage.download_file(video.storage_key, local_path)
            return local_path

    async def _store_keyframes(self, video_id: int, frames: list):
        async with get_session() as session:
            for frame in frames:
                kf = Keyframe(
                    video_id=video_id,
                    frame_num=frame.frame_num,
                    timestamp=frame.timestamp,
                    storage_key=frame.storage_key,
                )
                session.add(kf)

    async def _generate_embeddings(self, video_id: int, frames: list):
        frame_paths = [f.local_path for f in frames if os.path.exists(f.local_path)]
        if not frame_paths:
            logger.warning(f"No local frame files for video {video_id}")
            return

        # Generate per-frame embeddings
        frame_embeddings = self.embedder.embed_images_batch(frame_paths)

        # Update keyframes with embeddings
        async with get_session() as session:
            from sqlalchemy import select
            result = await session.execute(
                select(Keyframe).where(Keyframe.video_id == video_id).order_by(Keyframe.frame_num)
            )
            keyframes = result.scalars().all()

            for kf, emb in zip(keyframes, frame_embeddings):
                kf.embedding = emb.tolist()

        # Generate video-level embedding (avg pool)
        video_embedding = self.embedder.embed_video(frame_paths)
        async with get_session() as session:
            ve = VideoEmbedding(
                video_id=video_id,
                embedding=video_embedding.tolist(),
                model_name="ViT-B-32",
            )
            session.add(ve)

    async def _run_enrichment(self, video_id: int):
        from labeling.enrichment import EnrichmentService
        enrichment = EnrichmentService()
        await enrichment.enrich_video(video_id)

    async def _run_labeling(self, video_id: int):
        from labeling.detector import ObjectDetector

        detector = ObjectDetector()
        await detector.detect_for_video(video_id)

    async def _run_annotation(self, video_id: int):
        from labeling.llm_annotator import LLMAnnotationService
        annotator = LLMAnnotationService()
        await annotator.annotate_video(video_id)

    async def _run_gdpr_scan(self, video_id: int) -> dict:
        from labeling.gdpr import GDPRPipelineScanner
        scanner = GDPRPipelineScanner()
        return await scanner.scan_and_redact(video_id)

    async def _run_analytics(self, video_id: int):
        from labeling.analytics import AnalyticsService
        analytics = AnalyticsService()
        await analytics.analyze_video(video_id)

    async def _set_step(self, video_id: int, step_num: int):
        label = PIPELINE_STEPS.get(step_num, "Processing")
        step_text = f"[{step_num}/{TOTAL_STEPS}] {label}"
        async with get_session() as session:
            await session.execute(
                update(Video).where(Video.id == video_id).values(current_step=step_text)
            )

    async def _set_status(self, video_id: int, status: str, error_message: str = None, current_step: object = ...):
        async with get_session() as session:
            values = {"status": status}
            if error_message:
                values["error_message"] = error_message
            if current_step is not ...:
                values["current_step"] = current_step
            await session.execute(
                update(Video).where(Video.id == video_id).values(**values)
            )
