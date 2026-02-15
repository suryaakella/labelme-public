"""Parse natural language ingest queries using LangChain + Claude.

Extracts platform, max_results, and search topic from freeform text like:
  "20 cooking videos from instagram"
  "5 TikTok videos about skateboarding tricks"
  "youtube videos of woodworking"
"""

import logging

from pydantic import BaseModel, Field

from config import settings

logger = logging.getLogger(__name__)

_parser_chain = None


class ParsedQuery(BaseModel):
    platform: str = Field(
        description="Platform to search: youtube, instagram, or tiktok"
    )
    max_results: int = Field(
        description="Number of videos requested"
    )
    search_topic: str = Field(
        description="The actual search topic with platform/count stripped out"
    )


def _get_parser_chain():
    global _parser_chain
    if _parser_chain is None:
        from langchain_google_genai import ChatGoogleGenerativeAI
        from langchain_core.prompts import ChatPromptTemplate

        llm = ChatGoogleGenerativeAI(
            model=settings.gdpr_llm_model,
            google_api_key=settings.google_api_key,
            max_output_tokens=256,
            temperature=0,
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", (
                "You parse video search queries. Extract:\n"
                "- platform: youtube, instagram, or tiktok (default: instagram)\n"
                "- max_results: number of videos requested (default: 10)\n"
                "- search_topic: the actual search topic with platform name and count removed\n\n"
                "Examples:\n"
                "  '20 cooking videos from instagram' -> platform=instagram, max_results=20, search_topic=cooking\n"
                "  '5 TikTok videos about skateboarding' -> platform=tiktok, max_results=5, search_topic=skateboarding\n"
                "  'cat videos' -> platform=instagram, max_results=10, search_topic=cat videos\n"
                "  '100 reels of street food in tokyo' -> platform=instagram, max_results=100, search_topic=street food in tokyo\n"
                "  'charizard on insta' -> platform=instagram, max_results=10, search_topic=charizard\n"
                "  '3 fitness clips from yt' -> platform=youtube, max_results=3, search_topic=fitness\n"
                "  'dancing on ig' -> platform=instagram, max_results=10, search_topic=dancing\n"
                "  '2 prank videos tt' -> platform=tiktok, max_results=2, search_topic=prank\n"
            )),
            ("human", "{query}"),
        ])

        _parser_chain = prompt | llm.with_structured_output(ParsedQuery)

    return _parser_chain


def _fallback_parse(query: str) -> ParsedQuery:
    """Simple regex fallback when no API key is available."""
    import re
    q = query.lower()

    # Detect platform (default: instagram)
    platform = "instagram"
    if any(w in q for w in ["youtube", "yt "]):
        platform = "youtube"
    elif any(w in q for w in ["tiktok", "tik tok", " tt ", "#tt"]):
        platform = "tiktok"
    elif any(w in q for w in ["instagram", "insta", " ig ", "#ig"]):
        platform = "instagram"
    if any(w in q for w in ["reel", "reels"]):
        platform = "instagram"

    # Detect count
    max_results = 10
    match = re.search(r'(\d+)', query)
    if match:
        max_results = min(int(match.group(1)), 50)

    # Strip platform/count words to get topic
    topic = re.sub(r'\d+', '', query)
    for word in ["videos", "video", "reels", "reel", "from", "of", "about",
                  "on", "youtube", "yt", "instagram", "insta", "ig",
                  "tiktok", "tik tok", "tt"]:
        topic = re.sub(rf'\b{word}\b', '', topic, flags=re.IGNORECASE)
    topic = re.sub(r'\s+', ' ', topic).strip()
    if not topic:
        topic = query

    return ParsedQuery(platform=platform, max_results=max_results, search_topic=topic)


async def parse_ingest_query(query: str) -> ParsedQuery:
    if not settings.google_api_key:
        logger.info("No GOOGLE_API_KEY, using fallback parser")
        result = _fallback_parse(query)
    else:
        try:
            chain = _get_parser_chain()
            result = await chain.ainvoke({"query": query})
        except Exception as e:
            logger.warning(f"LLM query parser failed ({e}), using fallback regex parser")
            result = _fallback_parse(query)
    logger.info(f"Parsed query '{query}' -> platform={result.platform}, max_results={result.max_results}, topic='{result.search_topic}'")
    return result
