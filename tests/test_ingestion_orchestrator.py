from unittest.mock import AsyncMock, patch, MagicMock

import pytest

from ingestion.orchestrator import IngestionOrchestrator
from ingestion.discovery import VideoInfo


@pytest.mark.asyncio
async def test_deduplicate_urls():
    orch = IngestionOrchestrator()

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = [("https://existing.com",)]
    mock_session.execute.return_value = mock_result

    videos = [
        VideoInfo(url="https://existing.com", title="Old"),
        VideoInfo(url="https://new.com", title="New"),
    ]

    deduped = await orch._deduplicate_urls(mock_session, videos)
    assert len(deduped) == 1
    assert deduped[0].url == "https://new.com"


@pytest.mark.asyncio
async def test_deduplicate_empty():
    orch = IngestionOrchestrator()
    result = await orch._deduplicate_urls(AsyncMock(), [])
    assert result == []
