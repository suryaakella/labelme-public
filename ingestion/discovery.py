import logging
from dataclasses import dataclass, field
from typing import List, Optional

from apify_client import ApifyClientAsync

from config import settings

logger = logging.getLogger(__name__)


@dataclass
class VideoInfo:
    url: str
    title: str = ""
    platform: str = "youtube"
    duration: float = 0.0
    description: str = ""
    metadata: dict = field(default_factory=dict)
    # Rich Instagram fields
    creator_username: str = ""
    creator_id: str = ""
    creator_followers: int = 0
    creator_avatar_url: str = ""
    hashtags: List[str] = field(default_factory=list)
    mentions: List[str] = field(default_factory=list)
    music_info: Optional[dict] = None
    is_ad: bool = False
    is_ai_generated: bool = False
    posted_at: Optional[str] = None
    engagement: dict = field(default_factory=dict)
    comments: list = field(default_factory=list)
    sticker_texts: List[str] = field(default_factory=list)
    language: Optional[dict] = None


class DiscoveryService:
    APIFY_ACTORS = {
        "youtube": "streamers/youtube-scraper",
        "instagram": "apify/instagram-hashtag-scraper",
        "tiktok": "clockworks/tiktok-scraper",
    }

    def __init__(self):
        self.apify_token = settings.apify_api_token
        if not self.apify_token:
            raise RuntimeError("APIFY_API_TOKEN is required for video discovery")

    async def discover(
        self, query: str, platform: str = "youtube", max_results: int = 10
    ) -> List[VideoInfo]:
        if platform not in self.APIFY_ACTORS:
            raise ValueError(f"Unsupported platform: {platform}. Must be one of: {list(self.APIFY_ACTORS.keys())}")

        actor_id = self.APIFY_ACTORS[platform]
        run_input = self._build_payload(query, platform, max_results)

        logger.info(f"Apify request: actor={actor_id}, input={run_input}")

        client = ApifyClientAsync(self.apify_token)
        run = await client.actor(actor_id).call(run_input=run_input, timeout_secs=300)
        dataset_id = run["defaultDatasetId"]
        dataset = await client.dataset(dataset_id).list_items()
        items = dataset.items

        logger.info(f"Apify raw response: {len(items)} items for '{query}' on {platform}")
        if items and platform == "instagram":
            sample = items[0]
            logger.info(f"Instagram sample item keys: {list(sample.keys())[:20]}")
            # Log all post types to understand what the scraper returns
            type_counts = {}
            for it in items:
                t = it.get("type", "unknown")
                type_counts[t] = type_counts.get(t, 0) + 1
            logger.info(f"Instagram post types: {type_counts}")
            logger.info(f"Instagram sample: shortCode={sample.get('shortCode')}, type={sample.get('type')}, videoUrl={sample.get('videoUrl', 'N/A')[:80] if sample.get('videoUrl') else 'None'}")

        results = []
        for item in items[:max_results]:
            video_url = self._extract_video_url(item, platform)
            if not video_url:
                logger.debug(f"Skipped item (no video URL): keys={list(item.keys())[:10]}")
                continue

            info = VideoInfo(
                url=video_url,
                title=item.get("title", ""),
                platform=platform,
                duration=self._parse_duration(item.get("duration", 0)),
                description=item.get("description", "") or item.get("caption", ""),
                metadata=item,
            )

            # Parse Instagram-specific rich metadata
            if platform == "instagram":
                info = self._parse_instagram_metadata(info, item)

            results.append(info)

        logger.info(f"Apify discovered {len(results)} videos for '{query}' on {platform}")
        return results

    def _build_payload(self, query: str, platform: str, max_results: int) -> dict:
        if platform == "youtube":
            return {"searchQueries": [query], "maxResults": max_results}
        elif platform == "instagram":
            # Over-fetch because we filter out non-video (image) posts
            return {
                "hashtags": [query],
                "resultsType": "reels",
                "resultsLimit": max(max_results * 10, 20),
            }
        elif platform == "tiktok":
            return {"searchQueries": [query], "maxResults": max_results}
        return {"searchQueries": [query], "maxResults": max_results}

    @staticmethod
    def _parse_duration(raw) -> float:
        """Parse duration from various formats: seconds (int/float), 'HH:MM:SS', 'MM:SS'."""
        if not raw:
            return 0.0
        if isinstance(raw, (int, float)):
            return float(raw)
        s = str(raw)
        try:
            return float(s)
        except ValueError:
            pass
        # Parse HH:MM:SS or MM:SS
        parts = s.split(":")
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + int(parts[2])
            elif len(parts) == 2:
                return int(parts[0]) * 60 + int(parts[1])
        except ValueError:
            pass
        return 0.0

    def _extract_video_url(self, item: dict, platform: str) -> str:
        """Extract a direct video/post URL from an Apify result item, skipping non-video pages."""
        if platform == "instagram":
            # Skip non-video posts (images, carousels)
            post_type = (item.get("type") or "").lower()
            if post_type and post_type not in ("video", "reel"):
                return ""
            # Prefer direct video URL (CDN link, no auth needed) over reel page URL
            video_url = item.get("videoUrl") or ""
            if video_url:
                return video_url
            short_code = item.get("shortCode") or item.get("code")
            if short_code:
                return f"https://www.instagram.com/reel/{short_code}/"
            url = item.get("shortUrl") or item.get("url", "")
            # Skip explore/tag pages — they aren't individual videos
            if "/explore/" in url or "/tags/" in url:
                return ""
            return url
        elif platform == "tiktok":
            return item.get("videoUrl") or item.get("url") or item.get("webVideoUrl", "")
        else:
            # YouTube
            return item.get("url") or item.get("videoUrl", "")

    def _parse_instagram_metadata(self, info: VideoInfo, item: dict) -> VideoInfo:
        """Extract rich metadata from an Apify Instagram scraper response item."""
        info.creator_username = item.get("ownerUsername", "") or item.get("owner", {}).get("username", "")
        info.creator_id = str(item.get("ownerId", "")) or str(item.get("owner", {}).get("id", ""))
        info.creator_followers = int(item.get("ownerFollowerCount", 0) or 0)
        info.creator_avatar_url = item.get("ownerProfilePicUrl", "") or item.get("owner", {}).get("profile_pic_url", "")

        # Hashtags and mentions
        info.hashtags = item.get("hashtags", []) or []
        info.mentions = item.get("mentions", []) or []

        # Engagement metrics
        info.engagement = {
            "plays": item.get("videoPlayCount", 0) or item.get("videoViewCount", 0) or 0,
            "likes": item.get("likesCount", 0) or 0,
            "comments": item.get("commentsCount", 0) or 0,
            "shares": item.get("sharesCount", 0) or 0,
            "saves": item.get("savesCount", 0) or 0,
        }

        # Sponsored / ad detection
        info.is_ad = bool(item.get("isSponsored", False) or item.get("isPaidPartnership", False))

        # Timestamp
        ts = item.get("timestamp")
        if ts:
            info.posted_at = ts if isinstance(ts, str) else None

        # Music info
        music = item.get("musicInfo") or item.get("music", {})
        if music:
            info.music_info = {
                "artist": music.get("artist_name", "") or music.get("artist", ""),
                "title": music.get("song_name", "") or music.get("title", ""),
                "album": music.get("album", ""),
                "url": music.get("url", ""),
            }

        # Comments
        raw_comments = item.get("latestComments", []) or []
        info.comments = [
            {
                "comment_id": str(c.get("id", "")),
                "username": c.get("ownerUsername", "") or c.get("username", ""),
                "text": c.get("text", ""),
                "like_count": int(c.get("likesCount", 0) or c.get("likeCount", 0) or 0),
                "reply_count": int(c.get("repliesCount", 0) or c.get("replyCount", 0) or 0),
                "posted_at": c.get("timestamp"),
            }
            for c in raw_comments
            if c.get("text")
        ]

        return info
