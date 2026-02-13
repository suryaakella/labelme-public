from unittest.mock import patch, MagicMock
from pathlib import Path

from pipeline.frames import KeyframeExtractor


def test_extractor_init():
    ext = KeyframeExtractor()
    assert ext.interval == 30
    assert ext.scene_threshold == 0.3


@patch("pipeline.frames.storage")
@patch("pipeline.frames.subprocess.run")
def test_get_fps_fallback(mock_run, mock_storage):
    ext = KeyframeExtractor()
    mock_run.side_effect = Exception("ffprobe not found")
    fps = ext._get_fps("/fake/video.mp4")
    assert fps == 30.0


@patch("pipeline.frames.storage")
@patch("pipeline.frames.subprocess.run")
def test_get_fps_fractional(mock_run, mock_storage):
    ext = KeyframeExtractor()
    mock_run.return_value = MagicMock(stdout="30000/1001\n", returncode=0)
    fps = ext._get_fps("/fake/video.mp4")
    assert abs(fps - 29.97) < 0.1
