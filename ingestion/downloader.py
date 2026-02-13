import logging
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Optional
from urllib.parse import urlparse

import httpx
import yt_dlp

from config import settings
from config.storage import storage

logger = logging.getLogger(__name__)

# Direct CDN domains that serve raw video files (no yt-dlp needed)
DIRECT_DOWNLOAD_DOMAINS = {"cdninstagram.com", "fbcdn.net", "akamaihd.net"}


@dataclass
class DownloadResult:
    storage_key: str
    thumbnail_key: Optional[str]
    title: str
    description: str
    duration_sec: float
    width: int
    height: int
    fps: float
    file_size_bytes: int


class DownloadService:
    def __init__(self):
        self.max_resolution = settings.video_max_resolution

    def download(self, url: str, video_id: int) -> DownloadResult:
        tmpdir = tempfile.mkdtemp(prefix="forgeindex_")
        try:
            if self._is_direct_url(url):
                return self._download_direct(url, video_id, tmpdir)
            return self._do_download(url, video_id, tmpdir)
        finally:
            # Cleanup temp files
            for f in Path(tmpdir).iterdir():
                try:
                    f.unlink()
                except OSError:
                    pass
            try:
                os.rmdir(tmpdir)
            except OSError:
                pass

    def _is_direct_url(self, url: str) -> bool:
        """Check if URL is a direct video file (CDN link) rather than a page URL."""
        parsed = urlparse(url)
        hostname = parsed.hostname or ""
        # Check if domain matches known CDN patterns
        return any(hostname.endswith(d) for d in DIRECT_DOWNLOAD_DOMAINS)

    def _download_direct(self, url: str, video_id: int, tmpdir: str) -> DownloadResult:
        """Download a direct video URL (CDN link) using httpx."""
        logger.info(f"Direct download: {url[:100]}...")
        video_path = Path(tmpdir) / f"{video_id}.mp4"

        with httpx.Client(timeout=120, follow_redirects=True) as client:
            with client.stream("GET", url) as resp:
                resp.raise_for_status()
                with open(video_path, "wb") as f:
                    for chunk in resp.iter_bytes(chunk_size=65536):
                        f.write(chunk)

        if not video_path.exists() or video_path.stat().st_size == 0:
            raise RuntimeError(f"Direct download failed — empty file for {url[:100]}")

        file_size = video_path.stat().st_size

        # Upload video to MinIO
        storage_key = f"videos/{video_id}/{video_path.name}"
        storage.upload_file(storage_key, str(video_path), content_type="video/mp4")

        return DownloadResult(
            storage_key=storage_key,
            thumbnail_key=None,
            title="",
            description="",
            duration_sec=0.0,
            width=0,
            height=0,
            fps=0.0,
            file_size_bytes=file_size,
        )

    def _do_download(self, url: str, video_id: int, tmpdir: str) -> DownloadResult:
        output_template = os.path.join(tmpdir, "%(id)s.%(ext)s")

        ydl_opts = {
            "format": f"bestvideo[height<={self.max_resolution}]+bestaudio/best[height<={self.max_resolution}]",
            "outtmpl": output_template,
            "merge_output_format": "mp4",
            "quiet": True,
            "no_warnings": True,
            "writethumbnail": True,
        }

        # Use cookies file for sites that require auth (e.g. Instagram)
        if settings.ytdlp_cookies_file and os.path.isfile(settings.ytdlp_cookies_file):
            ydl_opts["cookiefile"] = settings.ytdlp_cookies_file

        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=True)

        # Find the downloaded file
        video_file = None
        thumb_file = None
        for f in Path(tmpdir).iterdir():
            if f.suffix in (".mp4", ".mkv", ".webm"):
                video_file = f
            elif f.suffix in (".jpg", ".jpeg", ".png", ".webp"):
                thumb_file = f

        if not video_file:
            raise RuntimeError(f"Download failed — no video file found for {url}")

        # Upload video to MinIO
        storage_key = f"videos/{video_id}/{video_file.name}"
        storage.upload_file(storage_key, str(video_file), content_type="video/mp4")

        # Upload thumbnail
        thumbnail_key = None
        if thumb_file:
            thumbnail_key = f"thumbnails/{video_id}/{thumb_file.name}"
            storage.upload_file(thumbnail_key, str(thumb_file), content_type=f"image/{thumb_file.suffix.lstrip('.')}")

        file_size = video_file.stat().st_size

        return DownloadResult(
            storage_key=storage_key,
            thumbnail_key=thumbnail_key,
            title=info.get("title", ""),
            description=info.get("description", ""),
            duration_sec=float(info.get("duration", 0) or 0),
            width=int(info.get("width", 0) or 0),
            height=int(info.get("height", 0) or 0),
            fps=float(info.get("fps", 0) or 0),
            file_size_bytes=file_size,
        )
