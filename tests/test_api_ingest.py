from unittest.mock import AsyncMock, patch, MagicMock

import pytest
from fastapi.testclient import TestClient

from ingestion.query_parser import ParsedQuery


@pytest.fixture
def client():
    with patch("api.main.storage"):
        from api.main import app
        return TestClient(app)


def test_ingest_endpoint(client):
    mock_session = AsyncMock()
    mock_task = MagicMock()
    mock_task.id = 1

    mock_parsed = ParsedQuery(
        platform="youtube",
        max_results=5,
        search_topic="CNC machining",
    )

    with patch("api.routes.ingest.get_session") as mock_gs, \
         patch("api.routes.ingest.task_queue") as mock_tq, \
         patch("api.routes.ingest.parse_ingest_query", new_callable=AsyncMock, return_value=mock_parsed):
        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_gs.return_value = mock_ctx
        mock_session.flush = AsyncMock()

        # Mock task id assignment
        def add_side_effect(task):
            task.id = 1
        mock_session.add.side_effect = add_side_effect

        resp = client.post("/api/ingest", json={
            "query": "5 CNC machining videos from youtube",
        })

    assert resp.status_code == 200
    data = resp.json()
    assert data["task_id"] == 1
    assert data["status"] == "pending"


def test_ingest_validation(client):
    resp = client.post("/api/ingest", json={
        "query": "",
    })
    assert resp.status_code == 422
