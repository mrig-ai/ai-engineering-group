"""
attractions_tools.py
--------------------
LangChain tools used by the Attractions Agent.

search_attractions queries Google via SerpAPI and returns the
top_sights block for a destination. Each sight contains:

  IDENTITY     title, description (may include open/closed status)
  RATINGS      rating (0-5), reviews (count)
  PRICING      price string, extracted_price (float) — absent if free
  MEDIA        thumbnail (image URL)
"""

import os
import json
import time
from dotenv import load_dotenv
from langchain.tools import tool
from serpapi import GoogleSearch
from src.logger import get_logger, log_api_request, log_api_response
load_dotenv()  # Load environment variables from .env file

_logger = get_logger("api.serpapi.attractions")

@tool
def search_attractions(query: str) -> str:
    """
    Search for top attractions and sights using Google via SerpAPI.

    Returns top sights for a destination. Each result contains:
      - title           → attraction name
      - description     → may include "Open" / "Closed" status
      - rating          → float 0-5
      - reviews         → integer review count
      - price           → price string e.g. "$17.42" (absent if free)
      - extracted_price → float price (absent if free or unknown)
      - thumbnail       → image URL string

    Args:
        query: Free-text search e.g. "Top sights in Madeira"
    """
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        return json.dumps({"error": "SERPAPI_KEY environment variable not set."})

    try:
        params = {"engine": "google", "q": query, "api_key": api_key}
        log_api_request(_logger, "serpapi.google_attractions", params)
        t0 = time.perf_counter()
        search = GoogleSearch(params)
        results = search.get_dict()
        duration_ms = (time.perf_counter() - t0) * 1000
        sights = results.get("top_sights", {}).get("sights", [])

        if not sights:
            return json.dumps({"error": "No attractions found.", "raw": results})
        
        # remove thumbnails
        for sight in sights:
            sight.pop("thumbnail", None)

        log_api_response(_logger, "serpapi.google_attractions", sights, duration_ms)
        return json.dumps(sights, indent=2)

    except Exception as exc:
        _logger.error("Attraction search failed: %s", exc, exc_info=True)
        return json.dumps({"error": str(exc)})