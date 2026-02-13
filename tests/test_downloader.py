import os
import tempfile
from unittest.mock import patch, MagicMock
from pathlib import Path

from ingestion.downloader import DownloadService


def test_download_service_init():
    svc = DownloadService()
    assert svc.max_resolution == 720


@patch("ingestion.downloader.storage")
@patch("ingestion.downloader.yt_dlp.YoutubeDL")
def test_download_uploads_to_minio(mock_ydl_cls, mock_storage):
    # Create a temp dir with a fake video file
    tmpdir = tempfile.mkdtemp()
    fake_video = Path(tmpdir) / "test123.mp4"
    fake_video.write_bytes(b"\x00" * 1024)

    mock_info = {
        "title": "Test Video",
        "description": "A test",
        "duration": 60,
        "width": 1280,
        "height": 720,
        "fps": 30,
    }

    mock_instance = MagicMock()
    mock_instance.__enter__ = lambda s: s
    mock_instance.__exit__ = MagicMock(return_value=False)
    mock_instance.extract_info.return_value = mock_info
    mock_ydl_cls.return_value = mock_instance

    svc = DownloadService()

    with patch("tempfile.mkdtemp", return_value=tmpdir):
        with patch("pathlib.Path.iterdir", return_value=[fake_video]):
            result = svc.download("https://youtube.com/watch?v=test", 1)

    assert result.title == "Test Video"
    assert result.duration_sec == 60.0
    assert mock_storage.upload_file.called
