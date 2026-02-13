from unittest.mock import AsyncMock, patch, MagicMock

import pytest
import numpy as np
from fastapi.testclient import TestClient


@pytest.fixture
def client():
    with patch("api.main.storage"):
        from api.main import app
        return TestClient(app)


def test_search_requires_query(client):
    resp = client.get("/api/search")
    assert resp.status_code == 422


def test_search_endpoint(client):
    fake_embedding = np.random.randn(512).astype(np.float32)
    fake_embedding = fake_embedding / np.linalg.norm(fake_embedding)

    with patch("api.routes.search.CLIPEmbedder") as mock_clip, \
         patch("api.routes.search.get_session") as mock_gs:
        mock_embedder = MagicMock()
        mock_embedder.embed_text.return_value = fake_embedding
        mock_clip.return_value = mock_embedder

        mock_session = AsyncMock()
        mock_result = MagicMock()
        mock_result.fetchall.return_value = []
        mock_session.execute.return_value = mock_result

        mock_ctx = AsyncMock()
        mock_ctx.__aenter__ = AsyncMock(return_value=mock_session)
        mock_ctx.__aexit__ = AsyncMock(return_value=False)
        mock_gs.return_value = mock_ctx

        resp = client.get("/api/search?q=welding+tutorial")

    assert resp.status_code == 200
    data = resp.json()
    assert data["query"] == "welding tutorial"
    assert isinstance(data["results"], list)
