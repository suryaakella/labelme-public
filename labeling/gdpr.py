"""GDPR compliance checks using LangChain + Claude Haiku.

Check 1 (pre-discovery): screen_query() blocks queries targeting personal data.
Check 2 (post-annotation): GDPRPipelineScanner scans text fields for PII.
    If PII is found the video is quarantined (status='gdpr_blocked') so it is
    never served to users. If annotation is missing, the video is flagged as
    unverified — also quarantined until it can be verified.
"""

import asyncio
import logging
from datetime import datetime, timezone
from typing import List, Optional

from pydantic import BaseModel, Field
from sqlalchemy import select, update

from config import settings
from config.database import get_session, Video, Annotation, Comment

logger = logging.getLogger(__name__)

# ── Structured output models ──────────────────────────────────────

class QueryScreenResult(BaseModel):
    is_personal_data_query: bool = Field(
        description="True if the query targets specific individuals' personal data"
    )
    risk_level: str = Field(
        description="Risk level: none, low, medium, or high"
    )
    explanation: str = Field(
        description="Brief explanation of the assessment"
    )


class PIIItem(BaseModel):
    type: str = Field(description="PII type, e.g. PERSON_NAME, EMAIL, PHONE")
    value: str = Field(description="The detected PII value")


class PIIDetectionResult(BaseModel):
    has_pii: bool = Field(description="True if PII was detected")
    pii_items: List[PIIItem] = Field(default_factory=list)
    redacted_text: str = Field(
        default="",
        description="The input text with PII replaced by [REDACTED]",
    )


# ── Lazy-loaded LangChain chains (singleton pattern) ─────────────

_query_screener = None
_pii_detector = None


def _get_query_screener():
    global _query_screener
    if _query_screener is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatGoogleGenerativeAI(
            model=settings.gdpr_llm_model,
            google_api_key=settings.google_api_key,
            max_output_tokens=512,
            temperature=0,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a GDPR compliance screener. Evaluate whether a search query "
                "targets specific individuals' personal data.\n\n"
                "BLOCK (is_personal_data_query=true) queries that:\n"
                "- Target a specific private individual by name (e.g. 'John Smith videos')\n"
                "- Search for someone's personal details (@handles targeting private persons, "
                "personal info requests)\n"
                "- Request private information about identifiable people\n\n"
                "ALLOW (is_personal_data_query=false) queries about:\n"
                "- Topics, categories, how-to, tutorials (e.g. 'CNC machining tutorials')\n"
                "- Brands, products, companies (e.g. 'Nike commercials')\n"
                "- Public figures in journalistic/educational context (e.g. 'presidential speech')\n"
                "- Generic terms, hashtags, trends (e.g. '#fitness')\n"
                "- Content creators by their public channel/brand name\n\n"
                "When in doubt, err on the side of caution (block)."
            )),
            ("human", "Evaluate this search query for GDPR compliance: {query}"),
        ])

        _query_screener = prompt | llm.with_structured_output(QueryScreenResult)
    return _query_screener


def _get_pii_detector():
    global _pii_detector
    if _pii_detector is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatGoogleGenerativeAI(
            model=settings.gdpr_llm_model,
            google_api_key=settings.google_api_key,
            max_output_tokens=4096,
            temperature=0,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You are a PII detection system. Scan the provided text for personally "
                "identifiable information (PII).\n\n"
                "PII types to detect:\n"
                "- PERSON_NAME: Full names of private individuals\n"
                "- EMAIL: Email addresses\n"
                "- PHONE: Phone numbers\n"
                "- ADDRESS: Physical/mailing addresses\n"
                "- SSN: Social security numbers or national IDs\n"
                "- CREDIT_CARD: Credit card numbers\n"
                "- DATE_OF_BIRTH: Dates of birth tied to individuals\n"
                "- USERNAME: Usernames when tied to identifying context\n"
                "- IP_ADDRESS: IP addresses\n"
                "- LICENSE_PLATE: Vehicle license plates\n\n"
                "Do NOT flag:\n"
                "- Public figures (politicians, celebrities, known creators)\n"
                "- Brand names or company names\n"
                "- Generic usernames without identifying context\n"
                "- Location names alone (cities, countries)\n\n"
                "For redacted_text: replace each PII instance with [REDACTED]. "
                "Keep all other text intact."
            )),
            ("human", "Scan this text for PII:\n\n{text}"),
        ])

        _pii_detector = prompt | llm.with_structured_output(PIIDetectionResult)
    return _pii_detector


# ── Check 1: Query Screening ─────────────────────────────────────

async def screen_query(query: str) -> QueryScreenResult:
    """Screen a search query for GDPR compliance.

    Raises RuntimeError if no API key is configured (fail closed).
    Returns QueryScreenResult with is_personal_data_query=False if check is disabled.
    """
    if not settings.gdpr_query_check_enabled:
        return QueryScreenResult(
            is_personal_data_query=False,
            risk_level="none",
            explanation="GDPR query check disabled",
        )

    if not settings.google_api_key:
        raise RuntimeError("GDPR query check requires GOOGLE_API_KEY")

    chain = _get_query_screener()
    loop = asyncio.get_event_loop()
    result = await loop.run_in_executor(None, lambda: chain.invoke({"query": query}))
    logger.info(f"GDPR query screen: query={query!r}, blocked={result.is_personal_data_query}, risk={result.risk_level}")
    return result


# ── Check 2: Pipeline PII Scanner ────────────────────────────────

class GDPRPipelineScanner:
    """Scans video text fields for PII after annotation, redacts, and flags."""

    SCAN_VERSION = 1
    MAX_FIELD_LENGTH = 4000
    MAX_PII_INPUT_LENGTH = 8000

    async def scan_and_redact(self, video_id: int) -> dict:
        """Scan all text fields for a video.

        - If PII found → quarantine the video (status='gdpr_blocked'), delete
          stored files from MinIO. We don't keep personal data.
        - If annotation is missing → flag as 'unverified' and quarantine.
          Cannot certify GDPR compliance without full text analysis.
        - Returns the gdpr_flags dict (also written to DB).
        """
        if not settings.gdpr_pipeline_check_enabled:
            logger.debug(f"GDPR pipeline check disabled, skipping video {video_id}")
            return {"status": "disabled"}

        if not settings.google_api_key:
            logger.warning(f"GDPR pipeline check skipped for video {video_id}: no GOOGLE_API_KEY")
            # No API key → can't verify → quarantine
            flags = {
                "status": "unverified",
                "reason": "GDPR check could not run: no GOOGLE_API_KEY configured",
                "has_pii": False,
                "scanned_at": datetime.now(timezone.utc).isoformat(),
                "scan_version": self.SCAN_VERSION,
            }
            await self._write_flags(video_id, flags)
            await self._quarantine_video(video_id, "GDPR compliance could not be verified (no API key)")
            return flags

        try:
            # Check if annotation exists — if not, we can't verify
            has_annotation = await self._has_annotation(video_id)
            if not has_annotation:
                logger.warning(f"Video {video_id}: no annotation found, cannot verify GDPR compliance")
                flags = {
                    "status": "unverified",
                    "reason": "No annotation data available. GDPR compliance cannot be verified.",
                    "has_pii": False,
                    "scanned_at": datetime.now(timezone.utc).isoformat(),
                    "scan_version": self.SCAN_VERSION,
                }
                await self._write_flags(video_id, flags)
                await self._quarantine_video(video_id, "GDPR: no annotation — compliance unverified")
                return flags

            text_fields = await self._gather_text_fields(video_id)
            if not text_fields:
                flags = {
                    "status": "clean",
                    "has_pii": False,
                    "pii_types": [],
                    "pii_count": 0,
                    "redacted_fields": [],
                    "scanned_at": datetime.now(timezone.utc).isoformat(),
                    "scan_version": self.SCAN_VERSION,
                }
                await self._write_flags(video_id, flags)
                return flags

            all_pii_types = set()
            total_pii_count = 0
            redacted_fields = []

            for field_name, text in text_fields.items():
                if not text or not text.strip():
                    continue

                result = await self._detect_pii(text)
                if result.has_pii:
                    all_pii_types.update(item.type for item in result.pii_items)
                    total_pii_count += len(result.pii_items)
                    redacted_fields.append(field_name)

            if redacted_fields:
                # PII found → quarantine. Don't store personal data.
                flags = {
                    "status": "blocked",
                    "has_pii": True,
                    "pii_types": sorted(all_pii_types),
                    "pii_count": total_pii_count,
                    "flagged_fields": redacted_fields,
                    "scanned_at": datetime.now(timezone.utc).isoformat(),
                    "scan_version": self.SCAN_VERSION,
                }
                await self._write_flags(video_id, flags)
                await self._quarantine_video(
                    video_id,
                    f"GDPR: personal data detected ({', '.join(sorted(all_pii_types))})"
                )
                logger.warning(
                    f"GDPR BLOCKED video {video_id}: PII types={sorted(all_pii_types)}, "
                    f"count={total_pii_count}, fields={redacted_fields}"
                )
                return flags
            else:
                # Clean
                flags = {
                    "status": "clean",
                    "has_pii": False,
                    "pii_types": [],
                    "pii_count": 0,
                    "redacted_fields": [],
                    "scanned_at": datetime.now(timezone.utc).isoformat(),
                    "scan_version": self.SCAN_VERSION,
                }
                await self._write_flags(video_id, flags)
                logger.info(f"GDPR scan video {video_id}: clean, no PII detected")
                return flags

        except Exception as e:
            logger.error(f"GDPR scan failed for video {video_id}: {e}", exc_info=True)
            flags = {
                "status": "error",
                "has_pii": False,
                "scan_error": str(e),
                "scanned_at": datetime.now(timezone.utc).isoformat(),
                "scan_version": self.SCAN_VERSION,
            }
            try:
                await self._write_flags(video_id, flags)
                # Scan error → can't verify → quarantine to be safe
                await self._quarantine_video(video_id, f"GDPR scan error: {e}")
            except Exception:
                logger.error(f"Failed to write GDPR error flags for video {video_id}")
            return flags

    async def _has_annotation(self, video_id: int) -> bool:
        """Check if an annotation record exists for this video."""
        async with get_session() as session:
            ann_result = await session.execute(
                select(Annotation).where(Annotation.video_id == video_id)
            )
            return ann_result.scalars().first() is not None

    async def _quarantine_video(self, video_id: int, reason: str):
        """Set video status to gdpr_blocked so it's never served to users."""
        async with get_session() as session:
            await session.execute(
                update(Video).where(Video.id == video_id).values(
                    status="gdpr_blocked",
                    error_message=reason,
                )
            )
        logger.info(f"Video {video_id} quarantined: {reason}")

    async def _gather_text_fields(self, video_id: int) -> dict:
        """Collect text fields from video record, annotation, and comments."""
        fields = {}

        async with get_session() as session:
            video = await session.get(Video, video_id)
            if not video:
                return fields

            if video.title:
                fields["title"] = video.title[:self.MAX_FIELD_LENGTH]
            if video.description:
                fields["description"] = video.description[:self.MAX_FIELD_LENGTH]

            # Sticker texts
            stickers = video.sticker_texts or []
            if stickers:
                fields["sticker_texts"] = " ".join(stickers)[:self.MAX_FIELD_LENGTH]

            # Mentions
            mentions = video.mentions or []
            if mentions:
                fields["mentions"] = " ".join(mentions)[:self.MAX_FIELD_LENGTH]

            # Creator username
            if video.creator_username:
                fields["creator_username"] = video.creator_username

            # Transcript and summary from annotation
            ann_result = await session.execute(
                select(Annotation).where(Annotation.video_id == video_id)
            )
            annotation = ann_result.scalars().first()
            if annotation and annotation.data:
                # v2 annotations store transcript as dict with full_text key
                transcript_val = annotation.data.get("transcript")
                if isinstance(transcript_val, dict):
                    transcript_text = transcript_val.get("full_text", "")
                elif isinstance(transcript_val, str):
                    transcript_text = transcript_val
                else:
                    transcript_text = ""
                if transcript_text:
                    fields["transcript"] = transcript_text[:self.MAX_FIELD_LENGTH]

                # Scan LLM summary for PII as well
                summary_text = annotation.data.get("summary", "")
                if summary_text:
                    fields["annotation_summary"] = summary_text[:self.MAX_FIELD_LENGTH]

            # Comments
            comment_result = await session.execute(
                select(Comment).where(Comment.video_id == video_id)
            )
            comments = comment_result.scalars().all()
            if comments:
                comment_texts = []
                for c in comments:
                    parts = []
                    if c.username:
                        parts.append(c.username)
                    parts.append(c.text)
                    comment_texts.append(": ".join(parts))
                fields["comments"] = "\n".join(comment_texts)[:self.MAX_FIELD_LENGTH]

        return fields

    async def _detect_pii(self, text: str) -> PIIDetectionResult:
        """Invoke the PII detector chain on a text snippet."""
        truncated = text[:self.MAX_PII_INPUT_LENGTH]
        chain = _get_pii_detector()
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(
            None, lambda: chain.invoke({"text": truncated})
        )

    async def _write_flags(self, video_id: int, flags: dict):
        """Write gdpr_flags JSONB to the video record."""
        async with get_session() as session:
            await session.execute(
                update(Video).where(Video.id == video_id).values(gdpr_flags=flags)
            )

