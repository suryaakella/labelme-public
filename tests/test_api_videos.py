from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("api.main.storage"):
        from api.main import app
        return TestClient(app)


def test_list_videos_empty(client):
    with patch("api.routes.videos.get_session") as mock_gs:
        mock_session = AsyncMock()

        # Count query
        mock_count_result = MagicMock()
        mock_count_result.scalar.return_value = 0

        # Video query
        mock_video_result = MagicMock()
        mock_video_result.scalars.return_value = MagicMock(all=MagicMock(return_value=[]))

        mock_session.execute = AsyncMock(side_effect=[mock_count_result, mock_video_result])

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_gs.return_value = mock_ctx

        resp = client.get("/api/videos")

    assert resp.status_code == 200
    data = resp.json()
    assert data["total"] == 0
    assert data["videos"] == []


def test_get_video_not_found(client):
    with patch("api.routes.videos.get_session") as mock_gs:
        mock_session = AsyncMock()
        mock_session.get.return_value = None

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_gs.return_value = mock_ctx

        resp = client.get("/api/videos/999")

    assert resp.status_code == 404
