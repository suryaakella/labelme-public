import logging
from typing import List, Tuple

from sqlalchemy import text

from config import settings
from config.database import get_session, DuplicatePair

logger = logging.getLogger(__name__)


class DeduplicationService:
    def __init__(self):
        self.threshold = settings.dedup_threshold

    async def check_duplicates(self, video_id: int) -> List[Tuple[int, float]]:
        async with get_session() as session:
            # Find similar videos using pgvector cosine similarity
            result = await session.execute(
                text("""
                    SELECT ve2.video_id,
                           1 - (ve1.embedding <=> ve2.embedding) AS similarity
                    FROM video_embeddings ve1
                    JOIN video_embeddings ve2 ON ve1.video_id != ve2.video_id
                    WHERE ve1.video_id = :video_id
                      AND 1 - (ve1.embedding <=> ve2.embedding) > :threshold
                    ORDER BY similarity DESC
                """),
                {"video_id": video_id, "threshold": self.threshold},
            )
            duplicates = [(row[0], row[1]) for row in result.fetchall()]

            # Insert duplicate pairs
            for other_id, sim in duplicates:
                a, b = min(video_id, other_id), max(video_id, other_id)
                try:
                    pair = DuplicatePair(video_id_a=a, video_id_b=b, similarity=sim)
                    session.add(pair)
                    await session.flush()
                except Exception:
                    await session.rollback()
                    # Pair already exists
                    pass

            if duplicates:
                logger.info(f"Video {video_id}: found {len(duplicates)} duplicates")

            return duplicates
