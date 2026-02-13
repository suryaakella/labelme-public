import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from config import settings
from config.database import get_session, IngestionTask
from config.task_queue import task_queue
from ingestion.orchestrator import IngestionOrchestrator
from ingestion.query_parser import parse_ingest_query

logger = logging.getLogger(__name__)
router = APIRouter(tags=["ingest"])


class IngestRequest(BaseModel):
    query: str = Field(..., min_length=1, description="Natural language query, e.g. '20 cooking videos from instagram'")


class IngestResponse(BaseModel):
    task_id: int
    status: str
    message: str


@router.post("/ingest", response_model=IngestResponse)
async def create_ingestion(req: IngestRequest):
    # Parse natural language query into structured intent
    parsed = await parse_ingest_query(req.query)

    # GDPR Check (only if API key is configured)
    if settings.gdpr_query_check_enabled and settings.google_api_key:
        try:
            from labeling.gdpr import screen_query
            result = await screen_query(parsed.search_topic)
            if result.is_personal_data_query:
                raise HTTPException(status_code=422, detail={
                    "error": "gdpr_query_blocked",
                    "message": (
                        "This query is not GDPR compliant. Queries targeting specific "
                        "individuals' personal data cannot be processed. Please use "
                        "topic-based, brand, or category searches instead."
                    ),
                    "risk_level": result.risk_level,
                    "explanation": result.explanation,
                })
        except HTTPException:
            raise
        except Exception as e:
            logger.warning(f"GDPR query check failed ({e}), skipping — allowing ingestion to proceed")

    async with get_session() as session:
        task = IngestionTask(
            query=req.query,
            platform=parsed.platform,
            max_results=parsed.max_results,
            status="pending",
        )
        session.add(task)
        await session.flush()
        task_id = task.id

    # Submit to background queue
    orchestrator = IngestionOrchestrator()
    task_queue.submit_async(orchestrator.run, task_id, task_id=str(task_id))

    return IngestResponse(
        task_id=task_id,
        status="pending",
        message=f"Ingestion task created: {parsed.max_results} '{parsed.search_topic}' videos from {parsed.platform}",
    )
