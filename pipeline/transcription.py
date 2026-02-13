import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from typing import List, Optional

from config import settings
from config.database import get_session, Transcript, TranscriptSegment

logger = logging.getLogger(__name__)

_whisper_model = None


def _get_whisper():
    global _whisper_model
    if _whisper_model is None:
        import whisper
        logger.info(f"Loading Whisper model: {settings.whisper_model}")
        _whisper_model = whisper.load_model(settings.whisper_model)
    return _whisper_model


@dataclass
class TranscriptResult:
    full_text: str
    language: str
    segments: List[dict]


class TranscriptionService:
    def extract_audio(self, video_path: str) -> str:
        tmpdir = tempfile.mkdtemp(prefix="forgeindex_audio_")
        audio_path = os.path.join(tmpdir, "audio.wav")

        cmd = [
            "ffmpeg", "-i", video_path,
            "-vn", "-acodec", "pcm_s16le",
            "-ar", "16000", "-ac", "1",
            audio_path,
            "-y", "-loglevel", "warning",
        ]
        subprocess.run(cmd, check=True, capture_output=True, text=True)
        return audio_path

    def transcribe(self, video_path: str) -> TranscriptResult:
        audio_path = self.extract_audio(video_path)
        try:
            model = _get_whisper()
            result = model.transcribe(audio_path, language=None)

            segments = [
                {
                    "start": seg["start"],
                    "end": seg["end"],
                    "text": seg["text"].strip(),
                }
                for seg in result.get("segments", [])
            ]

            return TranscriptResult(
                full_text=result.get("text", "").strip(),
                language=result.get("language", "en"),
                segments=segments,
            )
        finally:
            try:
                os.unlink(audio_path)
                os.rmdir(os.path.dirname(audio_path))
            except OSError:
                pass

    async def transcribe_and_store(self, video_id: int, video_path: str):
        try:
            result = self.transcribe(video_path)
        except subprocess.CalledProcessError:
            logger.warning(f"Video {video_id}: no audio track or unsupported codec — skipping transcription")
            return

        async with get_session() as session:
            transcript = Transcript(
                video_id=video_id,
                full_text=result.full_text,
                language=result.language,
                model_name=settings.whisper_model,
            )
            session.add(transcript)
            await session.flush()

            for seg in result.segments:
                segment = TranscriptSegment(
                    transcript_id=transcript.id,
                    video_id=video_id,
                    start_time=seg["start"],
                    end_time=seg["end"],
                    text=seg["text"],
                )
                session.add(segment)

            logger.info(f"Stored transcript for video {video_id}: {len(result.segments)} segments")
