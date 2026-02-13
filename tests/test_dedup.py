import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from pipeline.dedup import DeduplicationService


def test_dedup_threshold():
    svc = DeduplicationService()
    assert svc.threshold == 0.95


@pytest.mark.asyncio
async def test_check_duplicates_empty():
    svc = DeduplicationService()

    mock_session = AsyncMock()
    mock_result = MagicMock()
    mock_result.fetchall.return_value = []
    mock_session.execute.return_value = mock_result

    with patch("pipeline.dedup.get_session") as mock_gs:
        mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)
        duplicates = await svc.check_duplicates(1)

    assert duplicates == []
