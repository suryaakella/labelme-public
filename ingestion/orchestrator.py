import logging
from datetime import datetime, timezone
from typing import List

from sqlalchemy import select, update

from config.database import get_session, Video, IngestionTask, Comment
from ingestion.discovery import DiscoveryService, VideoInfo
from ingestion.downloader import DownloadService

logger = logging.getLogger(__name__)


class IngestionOrchestrator:
    def __init__(self):
        self.discovery = DiscoveryService()
        self.downloader = DownloadService()

    async def run(self, task_id: int):
        async with get_session() as session:
            task = await session.get(IngestionTask, task_id)
            if not task:
                logger.error(f"Task {task_id} not found")
                return
            query = task.query
            search_topic = task.search_topic or task.query
            platform = task.platform
            max_results = task.max_results

        try:
            await self._update_task(task_id, status="running", current_step="Starting ingestion...")

            # Step 1: Discover videos
            await self._update_task(task_id, current_step=f"Discovering {platform} videos for '{query}'...")
            videos = await self.discovery.discover(
                query=search_topic, platform=platform, max_results=max_results,
            )

            # Step 2: Deduplicate URLs
            await self._update_task(task_id, current_step=f"Found {len(videos)} videos, checking for duplicates...")
            videos = await self._deduplicate_urls(videos)
            total = len(videos)
            await self._update_task(task_id, total_videos=total,
                                    current_step=f"{total} new video(s) to process")

            if total == 0:
                await self._update_task(task_id, status="completed", progress=1.0,
                                        current_step="No new videos found")
                return

            # Step 3: Download and process each video
            gdpr_blocked_count = 0
            completed_count = 0
            failed_count = 0

            for i, video_info in enumerate(videos):
                video_label = video_info.title or video_info.url
                if len(video_label) > 60:
                    video_label = video_label[:57] + "..."

                try:
                    await self._update_task(task_id,
                        current_step=f"[{i+1}/{total}] Downloading: {video_label}")
                    video_id = await self._ingest_video(video_info, task_id=task_id)

                    await self._update_task(task_id,
                        current_step=f"[{i+1}/{total}] Processing pipeline: {video_label}")
                    await self._trigger_pipeline(video_id)

                    # Check final status
                    async with get_session() as session:
                        video = await session.get(Video, video_id)
                        if video and video.status == "gdpr_blocked":
                            gdpr_blocked_count += 1
                        elif video and video.status == "completed":
                            completed_count += 1
                        else:
                            failed_count += 1
                except Exception as e:
                    logger.error(f"Failed to ingest {video_info.url}: {e}", exc_info=True)
                    failed_count += 1
                    # Mark the video as failed if it was already created
                    try:
                        async with get_session() as session:
                            result = await session.execute(
                                select(Video).where(Video.url == video_info.url)
                            )
                            video = result.scalars().first()
                            if video and video.status not in ("completed", "gdpr_blocked"):
                                video.status = "failed"
                                video.error_message = str(e)[:500]
                    except Exception:
                        pass

                progress = (i + 1) / total
                await self._update_task(
                    task_id, progress=progress, processed_videos=i + 1,
                )

            # Build summary
            parts = []
            if completed_count:
                parts.append(f"{completed_count} video(s) processed successfully")
            if gdpr_blocked_count:
                parts.append(f"{gdpr_blocked_count} video(s) blocked (GDPR)")
            if failed_count:
                parts.append(f"{failed_count} video(s) failed")

            summary = ". ".join(parts) + "." if parts else "No videos processed."

            if completed_count == 0 and total > 0:
                await self._update_task(
                    task_id, status="completed", progress=1.0,
                    error_message=summary, current_step=summary,
                )
            else:
                await self._update_task(
                    task_id, status="completed", progress=1.0,
                    error_message=summary if gdpr_blocked_count or failed_count else None,
                    current_step=summary,
                )

        except Exception as e:
            logger.error(f"Ingestion task {task_id} failed: {e}", exc_info=True)
            await self._update_task(task_id, status="failed", error_message=str(e))

    async def _deduplicate_urls(self, videos: List[VideoInfo]) -> List[VideoInfo]:
        if not videos:
            return []
        urls = [v.url for v in videos]
        async with get_session() as session:
            result = await session.execute(
                select(Video.url).where(Video.url.in_(urls))
            )
            existing = {row[0] for row in result.fetchall()}
        deduped = [v for v in videos if v.url not in existing]
        logger.info(f"Deduplicated: {len(videos)} → {len(deduped)} new videos")
        return deduped

    async def _ingest_video(self, video_info: VideoInfo, *, task_id: int = None) -> int:
        # Create video record with rich metadata and commit so pipeline can see it
        async with get_session() as session:
            video = Video(
                url=video_info.url,
                platform=video_info.platform,
                title=video_info.title,
                description=video_info.description,
                duration_sec=video_info.duration,
                status="downloading",
                task_id=task_id,
                # Rich Instagram metadata
                creator_username=video_info.creator_username or None,
                creator_id=video_info.creator_id or None,
                creator_followers=video_info.creator_followers or None,
                creator_avatar_url=video_info.creator_avatar_url or None,
                hashtags=video_info.hashtags or [],
                mentions=video_info.mentions or [],
                music_info=video_info.music_info,
                is_ad=video_info.is_ad,
                is_ai_generated=video_info.is_ai_generated,
                posted_at=self._parse_timestamp(video_info.posted_at),
                engagement=video_info.engagement or {},
            )
            session.add(video)
            await session.flush()
            video_id = video.id
        # session commits here via context manager

        # Insert platform comments
        if video_info.comments:
            async with get_session() as session:
                for c in video_info.comments:
                    comment = Comment(
                        video_id=video_id,
                        comment_id=c.get("comment_id"),
                        username=c.get("username"),
                        text=c.get("text", ""),
                        like_count=c.get("like_count", 0),
                        reply_count=c.get("reply_count", 0),
                        posted_at=c.get("posted_at"),
                    )
                    session.add(comment)

        # Download
        result = self.downloader.download(video_info.url, video_id)

        # Update video with download results — commit so pipeline sees it
        async with get_session() as session:
            await session.execute(
                update(Video).where(Video.id == video_id).values(
                    storage_key=result.storage_key,
                    thumbnail_key=result.thumbnail_key,
                    title=result.title or video_info.title,
                    description=result.description or video_info.description,
                    duration_sec=result.duration_sec or video_info.duration,
                    width=result.width,
                    height=result.height,
                    fps=result.fps,
                    file_size_bytes=result.file_size_bytes,
                    status="processing",
                )
            )
        # session commits here

        return video_id

    async def _trigger_pipeline(self, video_id: int):
        from pipeline.orchestrator import PipelineOrchestrator
        orchestrator = PipelineOrchestrator()
        await orchestrator.run(video_id)

    @staticmethod
    def _parse_timestamp(value) -> datetime | None:
        if value is None:
            return None
        if isinstance(value, datetime):
            return value
        if isinstance(value, str):
            for fmt in ("%Y-%m-%dT%H:%M:%S.%fZ", "%Y-%m-%dT%H:%M:%SZ", "%Y-%m-%dT%H:%M:%S%z"):
                try:
                    dt = datetime.strptime(value, fmt)
                    if dt.tzinfo is None:
                        dt = dt.replace(tzinfo=timezone.utc)
                    return dt
                except ValueError:
                    continue
        return None

    async def _update_task(self, task_id: int, **kwargs):
        async with get_session() as session:
            await session.execute(
                update(IngestionTask).where(IngestionTask.id == task_id).values(**kwargs)
            )
