"""
hotel_tools.py
--------------
LangChain tools used by the Hotel Agent.

search_hotels returns everything needed to rank and recommend hotels
in a single call (based on real SerpAPI google_hotels response):

  IDENTITY     name, type, description, hotel_class (star rating)
  LOCATION     gps_coordinates (lat/lng), nearby_places with
               transport type and walking/taxi/transit durations
  PRICING      rate_per_night.extracted_lowest, total_rate.extracted_lowest
               deal, deal_description (e.g. "18% less than usual")
  RATINGS      overall_rating (0-5), location_rating, reviews count
               ratings: full star distribution [{stars, count}]
               reviews_breakdown: per-category sentiment with
               total_mentioned, positive, negative, neutral counts
               (Service, Pool, Breakfast, Fitness, Property, Nature,
                Bar, Room, Bathroom, Atmosphere, Dining, etc.)
  AMENITIES    amenities: flat string list
               ("Free Wi-Fi", "Outdoor pool", "Spa", "Free breakfast"...)
               excluded_amenities: what's explicitly missing
  POLICIES     check_in_time, check_out_time
"""

import os
import json
import time
from dotenv import load_dotenv
from langchain.tools import tool
from serpapi import GoogleSearch
from src.logger import get_logger, log_api_request, log_api_response
load_dotenv()  # Load environment variables from .env file

_logger = get_logger("api.serpapi.hotels")

@tool
def search_hotels(
    query: str,
    check_in_date: str,
    check_out_date: str,
    adults: int = 1,
    currency: str = "EUR",
) -> str:
    """
    Search for hotels using Google Hotels via SerpAPI.

    Returns up to 10 properties. Each result contains everything needed
    for ranking and recommendation:
      - name, type, description, hotel_class (star rating)
      - gps_coordinates, nearby_places (transport type + duration)
      - rate_per_night, total_rate, deal / deal_description
      - overall_rating, location_rating, review count
      - ratings: full 1-5 star distribution
      - reviews_breakdown: per-category sentiment (Service, Pool,
        Breakfast, Fitness, Property, Nature, Bar, etc.) with
        positive / negative / neutral counts
      - amenities (flat list), excluded_amenities
      - check_in_time, check_out_time

    Args:
        query:          Free-text search  e.g. "Bali resort"
        check_in_date:  ISO date string   e.g. "2026-03-23"
        check_out_date: ISO date string   e.g. "2026-03-31"
        adults:         Number of adult guests (default 1)
        currency:       3-letter currency code (default "EUR")
    """
    api_key = os.getenv("SERPAPI_KEY")
    if not api_key:
        return json.dumps({"error": "SERPAPI_KEY environment variable not set."})

    params = {
        "engine": "google_hotels",
        "q": query,
        "check_in_date": check_in_date,
        "check_out_date": check_out_date,
        "api_key": api_key,
        "currency": "EUR",
        "adults": adults,
    }

    try:
        log_api_request(_logger, "serpapi.google_hotels", params)
        t0 = time.perf_counter()
        search = GoogleSearch(params)
        results = search.get_dict()
        duration_ms = (time.perf_counter() - t0) * 1000
        properties = results.get("properties", [])

        if not properties:
            return json.dumps({"error": "No hotel results found.", "raw": results})
        
        # Keys the model actually uses
        allowed_keys = {
            "type",
            "name",
            "description",
            "gps_coordinates",
            "check_in_time",
            "check_out_time",
            "rate_per_night",
            "total_rate",
            "deal",              
            "deal_description",
            "nearby_places",
            "overall_rating",
            "reviews",
            "ratings",
            "location_rating",
            "reviews_breakdown",
            "amenities",
            "excluded_amenities",
        }

        # Keep only allowed fields
        clean_properties = [
            {k: v for k, v in prop.items() if k in allowed_keys}
            for prop in properties[:20]
]

        # Top 20
        log_api_response(_logger, "serpapi.google_hotels", clean_properties, duration_ms)
        return json.dumps(clean_properties, indent=2)

    except Exception as exc:
        _logger.error("Hotel search failed: %s", exc, exc_info=True)
        return json.dumps({"error": str(exc)})