import logging
import os
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import List

from config import settings
from config.storage import storage

logger = logging.getLogger(__name__)


@dataclass
class ExtractedFrame:
    frame_num: int
    timestamp: float
    storage_key: str
    local_path: str


class KeyframeExtractor:
    def __init__(self):
        self.interval = settings.keyframe_interval
        self.scene_threshold = settings.scene_threshold

    def extract(self, video_path: str, video_id: int) -> List[ExtractedFrame]:
        tmpdir = tempfile.mkdtemp(prefix="forgeindex_frames_")
        try:
            return self._do_extract(video_path, video_id, tmpdir)
        except Exception:
            # Cleanup on error
            self._cleanup(tmpdir)
            raise

    def _do_extract(self, video_path: str, video_id: int, tmpdir: str) -> List[ExtractedFrame]:
        output_pattern = os.path.join(tmpdir, "frame_%06d.jpg")

        # Use ffmpeg with scene detection + interval sampling
        vf = f"select='not(mod(n\\,{self.interval}))+gt(scene\\,{self.scene_threshold})',setpts=N/TB"

        cmd = [
            "ffmpeg", "-i", video_path,
            "-vf", vf,
            "-vsync", "vfr",
            "-q:v", "2",
            output_pattern,
            "-y", "-loglevel", "warning",
        ]

        subprocess.run(cmd, check=True, capture_output=True, text=True)

        # Get frame timestamps via ffprobe
        frames = self._collect_frames(tmpdir, video_id, video_path)
        return frames

    def _collect_frames(self, tmpdir: str, video_id: int, video_path: str) -> List[ExtractedFrame]:
        frames = []
        frame_files = sorted(Path(tmpdir).glob("frame_*.jpg"))

        # Get video fps for timestamp calculation
        fps = self._get_fps(video_path)

        for i, frame_file in enumerate(frame_files):
            frame_num = i * self.interval  # approximate
            timestamp = frame_num / fps if fps > 0 else 0.0

            storage_key = f"keyframes/{video_id}/frame_{i:06d}.jpg"
            storage.upload_file(storage_key, str(frame_file), content_type="image/jpeg")

            frames.append(ExtractedFrame(
                frame_num=frame_num,
                timestamp=timestamp,
                storage_key=storage_key,
                local_path=str(frame_file),
            ))

        logger.info(f"Extracted {len(frames)} keyframes for video {video_id}")
        return frames

    def _get_fps(self, video_path: str) -> float:
        cmd = [
            "ffprobe", "-v", "error",
            "-select_streams", "v:0",
            "-show_entries", "stream=r_frame_rate",
            "-of", "csv=p=0",
            video_path,
        ]
        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            fps_str = result.stdout.strip()
            if "/" in fps_str:
                num, den = fps_str.split("/")
                return float(num) / float(den)
            return float(fps_str)
        except Exception:
            return 30.0

    def _cleanup(self, tmpdir: str):
        for f in Path(tmpdir).iterdir():
            try:
                f.unlink()
            except OSError:
                pass
        try:
            os.rmdir(tmpdir)
        except OSError:
            pass
