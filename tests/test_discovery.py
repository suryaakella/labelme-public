from unittest.mock import patch, AsyncMock

import pytest

from ingestion.discovery import DiscoveryService, VideoInfo


@patch("ingestion.discovery.settings")
def test_init_requires_apify_token(mock_settings):
    mock_settings.apify_api_token = ""
    with pytest.raises(RuntimeError, match="APIFY_API_TOKEN is required"):
        DiscoveryService()


@patch("ingestion.discovery.settings")
def test_init_succeeds_with_token(mock_settings):
    mock_settings.apify_api_token = "test_token"
    svc = DiscoveryService()
    assert svc.apify_token == "test_token"


@patch("ingestion.discovery.settings")
def test_unsupported_platform(mock_settings):
    mock_settings.apify_api_token = "test_token"
    svc = DiscoveryService()
    with pytest.raises(ValueError, match="Unsupported platform"):
        import asyncio
        asyncio.get_event_loop().run_until_complete(svc.discover("query", platform="vimeo"))


@patch("ingestion.discovery.settings")
def test_build_payload_youtube(mock_settings):
    mock_settings.apify_api_token = "test_token"
    svc = DiscoveryService()
    payload = svc._build_payload("test", "youtube", 5)
    assert payload == {"searchQueries": ["test"], "maxResults": 5}


@patch("ingestion.discovery.settings")
def test_build_payload_instagram(mock_settings):
    mock_settings.apify_api_token = "test_token"
    svc = DiscoveryService()
    payload = svc._build_payload("test", "instagram", 5)
    assert payload == {"search": "test", "resultsLimit": 5, "resultsType": "posts"}


@patch("ingestion.discovery.settings")
def test_build_payload_tiktok(mock_settings):
    mock_settings.apify_api_token = "test_token"
    svc = DiscoveryService()
    payload = svc._build_payload("test", "tiktok", 5)
    assert payload == {"searchQueries": ["test"], "maxResults": 5}


@patch("ingestion.discovery.settings")
@pytest.mark.asyncio
async def test_discover_calls_apify(mock_settings):
    mock_settings.apify_api_token = "test_token"
    svc = DiscoveryService()

    mock_response = AsyncMock()
    mock_response.json.return_value = [
        {"url": "https://youtube.com/watch?v=abc", "title": "Test", "duration": 120},
    ]
    mock_response.raise_for_status = lambda: None

    with patch("ingestion.discovery.httpx.AsyncClient") as mock_client_cls:
        mock_client = AsyncMock()
        mock_client.post.return_value = mock_response
        mock_client.__aenter__ = AsyncMock(return_value=mock_client)
        mock_client.__aexit__ = AsyncMock(return_value=False)
        mock_client_cls.return_value = mock_client

        results = await svc.discover("CNC machining", "youtube", 5)

    assert len(results) == 1
    assert results[0].url == "https://youtube.com/watch?v=abc"
    assert results[0].platform == "youtube"
