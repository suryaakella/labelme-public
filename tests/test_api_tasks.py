from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("api.main.storage"):
        from api.main import app
        return TestClient(app)


def test_list_tasks_empty(client):
    with patch("api.routes.tasks.get_session") as mock_gs:
        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.scalars.return_value = MagicMock(all=MagicMock(return_value=[]))
        mock_session.execute.return_value = mock_result

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_gs.return_value = mock_ctx

        resp = client.get("/api/tasks")

    assert resp.status_code == 200
    assert resp.json() == []


def test_get_task_not_found(client):
    with patch("api.routes.tasks.get_session") as mock_gs:
        mock_session = AsyncMock()
        mock_session.get.return_value = None

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_gs.return_value = mock_ctx

        resp = client.get("/api/tasks/999")

    assert resp.status_code == 404
