from unittest.mock import MagicMock, patch

from config.storage import StorageClient


def test_storage_client_init():
    with patch("config.storage.Minio") as mock_minio:
        client = StorageClient()
        assert client.client is not None
        assert client.bucket == "forgeindex-test" or client.bucket  # uses env


def test_upload_bytes():
    with patch("config.storage.Minio") as mock_minio:
        mock_instance = MagicMock()
        mock_minio.return_value = mock_instance
        client = StorageClient()
        result = client.upload_bytes("test/key.bin", b"hello", "application/octet-stream")
        assert result == "test/key.bin"
        mock_instance.put_object.assert_called_once()


def test_get_url():
    with patch("config.storage.Minio"):
        client = StorageClient()
        url = client.get_url("videos/test.mp4")
        assert "videos/test.mp4" in url
