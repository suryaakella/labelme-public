from unittest.mock import patch, MagicMock

from pipeline.transcription import TranscriptionService


@patch("pipeline.transcription.subprocess.run")
def test_extract_audio(mock_run):
    mock_run.return_value = MagicMock(returncode=0)
    svc = TranscriptionService()
    path = svc.extract_audio("/fake/video.mp4")
    assert path.endswith("audio.wav")
    mock_run.assert_called_once()


@patch("pipeline.transcription._get_whisper")
@patch("pipeline.transcription.TranscriptionService.extract_audio")
def test_transcribe(mock_extract, mock_get_whisper):
    mock_extract.return_value = "/tmp/fake/audio.wav"

    mock_model = MagicMock()
    mock_model.transcribe.return_value = {
        "text": "Hello world",
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 2.0, "text": " Hello world"},
        ],
    }
    mock_get_whisper.return_value = mock_model

    with patch("os.unlink"), patch("os.rmdir"):
        svc = TranscriptionService()
        result = svc.transcribe("/fake/video.mp4")

    assert result.full_text == "Hello world"
    assert result.language == "en"
    assert len(result.segments) == 1
