import pytest
from unittest.mock import AsyncMock, patch, MagicMock

from pipeline.orchestrator import PipelineOrchestrator


def test_pipeline_orchestrator_init():
    orch = PipelineOrchestrator()
    assert orch.frame_extractor is not None
    assert orch.embedder is not None
    assert orch.dedup is not None
    assert orch.transcription is not None


@pytest.mark.asyncio
async def test_set_status():
    orch = PipelineOrchestrator()
    mock_session = AsyncMock()

    with patch("pipeline.orchestrator.get_session") as mock_gs:
        mock_gs.return_value.__aenter__ = AsyncMock(return_value=mock_session)
        mock_gs.return_value.__aexit__ = AsyncMock(return_value=False)
        await orch._set_status(1, "processing")

    mock_session.execute.assert_called_once()
