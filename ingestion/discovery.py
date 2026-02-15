import asyncio
import logging
from dataclasses import dataclass, field
from typing import List, Optional, Set

from apify_client import ApifyClientAsync

from config import settings

logger = logging.getLogger(__name__)

# ── LLM chain for suggesting alternative hashtags (singleton) ────

_hashtag_suggester = None


def _get_hashtag_suggester():
    global _hashtag_suggester
    if _hashtag_suggester is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate
        from pydantic import BaseModel, Field

        class HashtagSuggestions(BaseModel):
            hashtags: List[str] = Field(
                description="5 related Instagram hashtags (no # prefix, no spaces, lowercase)"
            )

        llm = ChatGoogleGenerativeAI(
            model=settings.gdpr_llm_model,
            google_api_key=settings.google_api_key,
            max_output_tokens=256,
            temperature=0.7,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You suggest Instagram hashtags for video discovery.\n"
                "Given a search topic and hashtags that returned no results, "
                "suggest 5 alternative hashtags that are popular on Instagram "
                "and closely related to the topic.\n\n"
                "Rules:\n"
                "- No # prefix, no spaces, all lowercase\n"
                "- Use hashtags that actually exist and are popular on Instagram\n"
                "- Be creative but stay relevant to the topic\n"
                "- Prefer single-word or common compound hashtags (e.g. 'streetfood', 'robotics')\n"
                "- Do NOT repeat any of the already-tried hashtags"
            )),
            ("human",
             "Topic: {topic}\n"
             "Already tried (no results): {tried}\n"
             "Suggest 5 related Instagram hashtags:"),
        ])

        _hashtag_suggester = prompt | llm.with_structured_output(HashtagSuggestions)
    return _hashtag_suggester


@dataclass
class VideoInfo:
    url: str
    title: str = ""
    platform: str = "instagram"
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

    MAX_DISCOVERY_ROUNDS = 3  # initial + 2 LLM-suggested retries

    def __init__(self):
        self.apify_token = settings.apify_api_token
        if not self.apify_token:
            raise RuntimeError("APIFY_API_TOKEN is required for video discovery")

    async def discover(
        self, query: str, platform: str = "instagram", max_results: int = 10
    ) -> List[VideoInfo]:
        if platform not in self.APIFY_ACTORS:
            raise ValueError(f"Unsupported platform: {platform}. Must be one of: {list(self.APIFY_ACTORS.keys())}")

        # For Instagram, use smart retry with LLM-suggested hashtags
        if platform == "instagram":
            return await self._discover_instagram_smart(query, max_results)

        # YouTube / TikTok — single search
        return await self._run_apify_search(query, platform, max_results)

    async def _discover_instagram_smart(
        self, query: str, max_results: int,
    ) -> List[VideoInfo]:
        """Search Instagram with LLM-powered retry.

        Round 1: Search initial hashtag variants (combined + individual words).
        Round 2+: If not enough results, ask LLM for related hashtags and search those.
        Repeat until we have enough videos or exhaust retries.
        """
        all_results: List[VideoInfo] = []
        seen_urls: Set[str] = set()
        tried_hashtags: Set[str] = set()

        # Round 1: initial hashtags from the query
        initial_hashtags = self._build_instagram_hashtags(query)
        logger.info(f"Instagram discovery round 1: hashtags={initial_hashtags}")

        new_results = await self._search_instagram_hashtags(
            initial_hashtags, max_results, seen_urls,
        )
        all_results.extend(new_results)
        tried_hashtags.update(initial_hashtags)

        if len(all_results) >= max_results:
            return all_results[:max_results]

        # Rounds 2+: LLM-suggested alternatives
        if not settings.google_api_key:
            logger.info("No API key for LLM hashtag suggestions, returning what we have")
            return all_results

        for round_num in range(2, self.MAX_DISCOVERY_ROUNDS + 1):
            still_need = max_results - len(all_results)
            logger.info(
                f"Instagram discovery round {round_num}: have {len(all_results)}/{max_results}, "
                f"need {still_need} more"
            )

            # Ask LLM for alternative hashtags
            suggested = await self._suggest_hashtags(query, tried_hashtags)
            new_hashtags = [h for h in suggested if h not in tried_hashtags]

            if not new_hashtags:
                logger.info("LLM suggested no new hashtags, stopping")
                break

            logger.info(f"Instagram discovery round {round_num}: trying LLM-suggested hashtags={new_hashtags}")

            new_results = await self._search_instagram_hashtags(
                new_hashtags, still_need, seen_urls,
            )
            all_results.extend(new_results)
            tried_hashtags.update(new_hashtags)

            if len(all_results) >= max_results:
                break

        logger.info(
            f"Instagram smart discovery complete: {len(all_results)} videos found "
            f"across {len(tried_hashtags)} hashtags"
        )
        return all_results[:max_results]

    async def _suggest_hashtags(self, topic: str, tried: Set[str]) -> List[str]:
        """Ask LLM for related hashtags."""
        try:
            chain = _get_hashtag_suggester()
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                lambda: chain.invoke({"topic": topic, "tried": ", ".join(sorted(tried))}),
            )
            # Sanitize suggestions
            suggestions = []
            for h in result.hashtags:
                clean = self._sanitize_hashtag(h)
                if clean and len(clean) >= 3:
                    suggestions.append(clean)
            logger.info(f"LLM suggested hashtags for '{topic}': {suggestions}")
            return suggestions
        except Exception as e:
            logger.warning(f"LLM hashtag suggestion failed: {e}")
            return []

    async def _search_instagram_hashtags(
        self,
        hashtags: List[str],
        max_results: int,
        seen_urls: Set[str],
    ) -> List[VideoInfo]:
        """Run a single Apify Instagram hashtag search and parse results."""
        run_input = {
            "hashtags": hashtags,
            "resultsType": "reels",
            "resultsLimit": max(max_results * 10, 20),
        }

        client = ApifyClientAsync(self.apify_token)
        actor_id = self.APIFY_ACTORS["instagram"]
        logger.info(f"Apify request: actor={actor_id}, hashtags={hashtags}")

        run = await client.actor(actor_id).call(run_input=run_input, timeout_secs=300)
        dataset_id = run["defaultDatasetId"]
        dataset = await client.dataset(dataset_id).list_items()
        items = dataset.items

        logger.info(f"Apify returned {len(items)} items for hashtags {hashtags}")

        # Log post types for debugging
        if items:
            type_counts = {}
            for it in items:
                t = it.get("type", "NO_TYPE")
                type_counts[t] = type_counts.get(t, 0) + 1
            logger.info(f"Instagram post types: {type_counts}")
            sample = items[0]
            logger.info(f"Sample item keys: {sorted(sample.keys())[:15]}")
            logger.info(f"Sample: type={sample.get('type')}, videoUrl={bool(sample.get('videoUrl'))}, shortCode={sample.get('shortCode')}")

        results = []
        for item in items:
            if len(results) >= max_results:
                break

            video_url = self._extract_video_url(item, "instagram")
            if not video_url or video_url in seen_urls:
                continue

            seen_urls.add(video_url)
            info = VideoInfo(
                url=video_url,
                title=item.get("title", ""),
                platform="instagram",
                duration=self._parse_duration(item.get("duration", 0)),
                description=item.get("description", "") or item.get("caption", ""),
                metadata=item,
            )
            info = self._parse_instagram_metadata(info, item)
            results.append(info)

        return results

    async def _run_apify_search(
        self, query: str, platform: str, max_results: int,
    ) -> List[VideoInfo]:
        """Run a single Apify search for YouTube/TikTok."""
        actor_id = self.APIFY_ACTORS[platform]
        run_input = self._build_payload(query, platform, max_results)

        logger.info(f"Apify request: actor={actor_id}, input={run_input}")

        client = ApifyClientAsync(self.apify_token)
        run = await client.actor(actor_id).call(run_input=run_input, timeout_secs=300)
        dataset_id = run["defaultDatasetId"]
        dataset = await client.dataset(dataset_id).list_items()
        items = dataset.items

        logger.info(f"Apify raw response: {len(items)} items for '{query}' on {platform}")

        results = []
        for item in items[:max_results]:
            video_url = self._extract_video_url(item, platform)
            if not video_url:
                continue

            info = VideoInfo(
                url=video_url,
                title=item.get("title", ""),
                platform=platform,
                duration=self._parse_duration(item.get("duration", 0)),
                description=item.get("description", "") or item.get("caption", ""),
                metadata=item,
            )
            results.append(info)

        logger.info(f"Apify discovered {len(results)} videos for '{query}' on {platform}")
        return results

    @classmethod
    def _build_instagram_hashtags(cls, query: str) -> list:
        """Build hashtag variants from a multi-word query.

        'robot walking' → ['robotwalking', 'robot', 'walking']
        'cooking'       → ['cooking']
        """
        words = query.strip().split()
        hashtags = []

        # Combined form (all words joined)
        combined = cls._sanitize_hashtag(query)
        if combined:
            hashtags.append(combined)

        # Individual words (only if multi-word query)
        if len(words) > 1:
            for word in words:
                tag = cls._sanitize_hashtag(word)
                if tag and tag not in hashtags and len(tag) >= 3:
                    hashtags.append(tag)

        return hashtags

    @staticmethod
    def _sanitize_hashtag(raw: str) -> str:
        """Strip leading #, whitespace, and any characters Apify's hashtag regex rejects."""
        import re
        tag = raw.strip().lstrip("#").strip()
        # Remove chars forbidden by Apify: !?.,:;-+=*&%$#@/\~^|<>()[]{}\"'`  and whitespace
        tag = re.sub(r'[!?.,:;\-+=*&%$#@/\\~^|<>()\[\]{}"\'`\s]+', '', tag)
        return tag

    def _build_payload(self, query: str, platform: str, max_results: int) -> dict:
        """Build Apify payload for YouTube/TikTok (Instagram handled separately)."""
        if platform == "youtube":
            return {"searchQueries": [query], "maxResults": max_results}
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
