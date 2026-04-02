"""
hotel_agent.py
--------------
Hotel recommendation agent — minimal LLM output approach.

WHAT THE LLM DOES (only):
  Receives indexed hotel summaries → returns status, recommended_index,
  alternative_indices, reasoning.

WHAT PYTHON DOES (zero LLM tokens):
  Fetch + hydrate all HotelOption fields, parse user preferences,
  compute price/rating ranges, assemble search_criteria.

Real SerpAPI shape notes (verified against live data):
  - deal          → "18% less than usual"  (parse discount % from here)
  - deal_description → "Deal"              (just a label, not useful)
  - nearby_places entries may omit "transportations" key entirely
  - description   → absent on some properties (no key at all)
  - type          → "vacation rental" (with space) for vacation rentals
"""

import json
import re
from datetime import date
from typing import Optional

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from src.logger import get_logger, log_agent_run
from src.models import (
    Amenity,
    HotelOption,
    HotelRankingResponse,
    HotelsAgentResponse,
    NearbyPlace,
    RatingDistribution,
    ReviewBreakdown,
)
from src.tools.hotel_tools import search_hotels


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _slugify(name: str) -> str:
    """'ORA Resort Avia' → 'ora-resort-avia'"""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def _parse_duration_minutes(duration: str) -> int:
    """
    '12 min'        → 12
    '1 hour'        → 60
    '1 hr 37 min'   → 97
    '4 hr 8 min'    → 248
    """
    hours   = re.search(r"(\d+)\s*h(?:r|our)?", duration, re.IGNORECASE)
    minutes = re.search(r"(\d+)\s*min",           duration, re.IGNORECASE)
    total   = 0
    if hours:
        total += int(hours.group(1)) * 60
    if minutes:
        total += int(minutes.group(1))
    return total or 0


def _parse_nearby_places(raw_places: list[dict]) -> list[NearbyPlace]:
    """
    Each entry:  {"name": "...", "transportations": [{"type": "Taxi", "duration": "7 min"}]}
    Some entries omit "transportations" entirely — skip those (no useful distance).
    """
    result = []
    for place in raw_places:
        transportations = place.get("transportations") or []
        if not transportations:
            continue

        durations = [
            (_parse_duration_minutes(t.get("duration", "")), t.get("type", ""))
            for t in transportations
            if t.get("duration")
        ]
        if not durations:
            continue

        min_minutes, _ = min(durations, key=lambda x: x[0])
        transport_types = [t["type"] for t in transportations if t.get("type")]

        result.append(NearbyPlace(
            name=place.get("name", ""),
            distance_minutes=min_minutes,
            transportation_types=transport_types,
        ))
    return result



def _parse_deal_discount(prop: dict) -> Optional[float]:
    """
    SerpAPI puts the human-readable string in the 'deal' key:
      prop["deal"]             = "18% less than usual"
      prop["deal_description"] = "Deal"   ← just a label, not useful

    Returns the discount percentage as a float, or None.
    """
    deal_str = prop.get("deal")
    if not deal_str:
        return None
    match = re.search(r"(\d+(?:\.\d+)?)\s*%", deal_str)
    return float(match.group(1)) if match else None


def _hotel_type(raw_type: Optional[str]) -> str:
    """Map SerpAPI 'type' string to HotelOption Literal."""
    mapping = {
        "hotel":           "hotel",
        "resort":          "resort",
        "vacation rental": "vacation_rental",
        "vacation_rental": "vacation_rental",
        "villa":           "villa",
        "hostel":          "hostel",
    }
    return mapping.get((raw_type or "").lower(), "hotel")


def _nights_between(check_in: str, check_out: str) -> int:
    """ISO date strings → number of nights (minimum 1)."""
    try:
        return max((date.fromisoformat(check_out) - date.fromisoformat(check_in)).days, 1)
    except ValueError:
        return 1


def _extract_location(query: str) -> str:
    """
    Best-effort: strip leading/trailing hotel type qualifiers.
      "Hotels in Paris"      → "Paris"
      "Bali resort"          → "Bali"
      "luxury hotel Madeira" → "Madeira"
    """
    # Remove leading qualifiers like "hotels in", "resorts in", "luxury hotel", etc.
    cleaned = re.sub(
        r"^(hotels?\s+in|hotel\s+in|resorts?\s+in|luxury\s+hotel|budget\s+hotel)\s+",
        "", query.strip(), flags=re.IGNORECASE,
    )
    # Remove trailing qualifiers like "hotel", "resort", "hostel", "villa", etc.
    cleaned = re.sub(
        r"\s+(hotel|resort|hostel|villa|accommodation)s?$",
        "", cleaned.strip(), flags=re.IGNORECASE,
    )
    return cleaned.strip().title() or query.title()


# ── Full hotel hydration (pure Python) ───────────────────────────────────────

def _hydrate_hotel(
    prop: dict,
    check_in_date: str,
    check_out_date: str,
    currency: str,
) -> HotelOption:
    """Build a complete HotelOption from a raw SerpAPI property dict."""
    amenity_strings: list[str] = prop.get("amenities") or []
    # Amenity.coerce_from_string fires on each string: strips '($)', sets included=False
    amenities     = [Amenity.model_validate(a) for a in amenity_strings]
    amenity_lower = [a.lower() for a in amenity_strings]

    pet_friendly          = any("pet-friendly" in a for a in amenity_lower)
    kid_friendly          = any("kid-friendly" in a for a in amenity_lower)
    wheelchair_accessible = any("accessible"   in a for a in amenity_lower)

    rating_distribution = [
        RatingDistribution(stars=r["stars"], count=r["count"])
        for r in (prop.get("ratings") or [])
        if "stars" in r and "count" in r
    ]

    review_breakdown = [
        ReviewBreakdown(
            category=r["name"],  # field name; alias is 'name'
            positive=r.get("positive", 0),
            negative=r.get("negative", 0),
            neutral=r.get("neutral", 0),
            total_mentioned=r.get("total_mentioned", 0),
        )
        for r in (prop.get("reviews_breakdown") or [])
        if "name" in r
    ]

    price_per_night = (prop.get("rate_per_night") or {}).get("extracted_lowest") or 0.0
    total_price     = (prop.get("total_rate")     or {}).get("extracted_lowest") or 0.0
    nights          = _nights_between(check_in_date, check_out_date)

    # 'deal' key holds the human-readable string e.g. "18% less than usual"
    deal_active       = "deal" in prop
    deal_discount_pct = _parse_deal_discount(prop)

    gps = prop.get("gps_coordinates") or {}

    return HotelOption(
        option_id             = _slugify(prop.get("name", "unknown")),
        name                  = prop.get("name", "Unknown"),
        type                  = _hotel_type(prop.get("type")),
        description           = prop.get("description"),   # absent on some properties → None
        location              = "",                        # backfilled by caller
        latitude              = gps.get("latitude", 0.0),
        longitude             = gps.get("longitude", 0.0),
        nearby_places         = _parse_nearby_places(prop.get("nearby_places") or []),
        currency              = currency,
        price_per_night       = price_per_night,
        total_price           = total_price,
        nights                = nights,
        check_in_date         = check_in_date,
        check_out_date        = check_out_date,
        check_in_time         = prop.get("check_in_time"),
        check_out_time        = prop.get("check_out_time"),
        deal_active           = deal_active,
        deal_discount_percentage = deal_discount_pct,
        overall_rating        = prop.get("overall_rating"),
        total_reviews         = prop.get("reviews", 0),
        location_rating       = prop.get("location_rating"),
        rating_distribution   = rating_distribution,
        review_breakdown      = review_breakdown,
        amenities             = amenities,
        excluded_amenities    = prop.get("excluded_amenities") or [],
        pet_friendly          = pet_friendly,
        kid_friendly          = kid_friendly,
        wheelchair_accessible = wheelchair_accessible,
    )


# ── Compact indexed summary for LLM ──────────────────────────────────────────

def _build_hotel_summary(prop: dict, index: int, budget: float) -> dict:
    """
    Compact dict sent to the LLM — only ranking-relevant fields.
    Boolean amenity flags are pre-computed so the LLM never scans raw lists.
    """
    amenity_lower = [a.lower() for a in (prop.get("amenities") or [])]
    price = (prop.get("rate_per_night") or {}).get("extracted_lowest") or 0.0


    return {
        "i":               index,
        "name":            prop.get("name", ""),
        "type":            prop.get("type", ""),
        "price_per_night": price,
        "total_price":     (prop.get("total_rate") or {}).get("extracted_lowest") or 0.0,
        "within_budget":   budget <= 0 or (price is not None and price <= budget),
        "rating":          prop.get("overall_rating"),
        "location_rating": prop.get("location_rating"),
        "reviews":         prop.get("reviews", 0),
        "deal":            prop.get("deal"),
        # pre-computed boolean flags
        "has_pool":           any("pool"         in a for a in amenity_lower),
        "has_spa":            any("spa"           in a for a in amenity_lower),
        "has_gym":            any("fitness" in a or "gym" in a for a in amenity_lower),
        "has_free_breakfast": any(a == "free breakfast" for a in amenity_lower),
        "has_breakfast":      any("breakfast"    in a for a in amenity_lower),
        "has_free_wifi":      any("free wi-fi" in a or "free wifi" in a for a in amenity_lower),
        "is_pet_friendly":    any("pet-friendly" in a for a in amenity_lower),
        "is_kid_friendly":    any("kid-friendly" in a for a in amenity_lower),
        "is_beachfront":      any("beach"        in a for a in amenity_lower)
                              or "beach" in (prop.get("description") or "").lower(),
        "is_accessible":      any("accessible"   in a for a in amenity_lower),
    }


# ── Agent ─────────────────────────────────────────────────────────────────────

class HotelAgent:
    """Hotel Agent — fetches, hydrates, and LLM-ranks hotels with minimal LLM output."""

    def __init__(
        self,
        model_name: str = "gpt-5-nano",
        temperature: float = 0.0,
        verbose: bool = True,
    ):
        self.llm     = ChatOpenAI(model=model_name, temperature=temperature)
        self.verbose = verbose
        self.logger  = get_logger("agent.hotel")

        # No tools — raw data fetched in Python, injected via prompt.
        self.agent = create_agent(
            model=self.llm,
            tools=[],
            system_prompt=self._create_system_prompt(),
            response_format=HotelRankingResponse,
            name="hotel_ranking_agent",
        )

    # ── System prompt ─────────────────────────────────────────────────────────
    @staticmethod
    def _create_system_prompt() -> str:
        return """You are a Hotel Ranking Agent.

You receive an indexed list of pre-processed hotel summaries and must rank them.

OUTPUT — return ONLY a HotelRankingResponse with these fields:
  status              "success" | "over_budget" | "no_results"
  recommended_index   index of the single best hotel (None only if no_results)
  alternative_indices up to 4 runner-up indices (ordered best-first)
  reasoning           2-4 sentences: why this hotel was chosen over others

DO NOT output any other fields. Python handles all field mapping.

RANKING RULES:
  - Budget is a HARD constraint (within_budget=true required).
    If ALL hotels have within_budget=false → status="over_budget", pick cheapest.
  - Match pre-computed boolean flags to user preferences:
      "pool"             → has_pool
      "spa"              → has_spa
      "gym" / "fitness"  → has_gym
      "free breakfast"   → has_free_breakfast (stricter than has_breakfast)
      "breakfast"        → has_breakfast
      "beach"            → is_beachfront
      "pet-friendly"     → is_pet_friendly
      "family" / "kids"  → is_kid_friendly
      "accessible"       → is_accessible
      "deal"             → prefer hotels where deal is non-null
      "location"/"central" → weight location_rating higher
  - When no preference stated: balance price, rating, location_rating, deal.
  - reasoning must reference specific hotel names and concrete data points (e.g. rating, price).
"""

    # ── Raw fetch ─────────────────────────────────────────────────────────────
    def _fetch_raw_properties(
        self,
        query: str,
        check_in_date: str,
        check_out_date: str,
        adults: int,
        currency: str,
    ) -> list[dict]:
        raw = search_hotels.invoke({
            "query": query,
            "check_in_date": check_in_date,
            "check_out_date": check_out_date,
            "adults": adults,
            "currency": currency,
        })
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []

    # ── Main entry point ──────────────────────────────────────────────────────
    @log_agent_run("HotelAgent")
    def search_and_recommend(
        self,
        query: str,
        check_in_date: str,
        check_out_date: str,
        adults: int = 1,
        currency: str = "EUR",
        budget_per_night: float = 0,
        user_preferences: str = "",
    ) -> HotelsAgentResponse:
        """
        Search Google Hotels and return a ranked HotelsAgentResponse.

        Args:
            query:             Free-text search e.g. "Bali resort"
            check_in_date:     ISO date  e.g. "2026-03-23"
            check_out_date:    ISO date  e.g. "2026-03-31"
            adults:            Number of guests
            currency:          3-letter code (default "EUR")
            budget_per_night:  Hard price ceiling per night. 0 = no limit.
            user_preferences:  Free-text e.g. "pool, free breakfast, rating above 4"
        """
        self.logger.info(
            "Searching hotels: %s | %s → %s | budget=%s/night | adults=%d",
            query, check_in_date, check_out_date,
            f"{budget_per_night} {currency}" if budget_per_night > 0 else "unlimited",
            adults,
        )
        if user_preferences:
            self.logger.debug("Hotel preferences: %s", user_preferences)

        # ── 1. Fetch raw properties ───────────────────────────────────────────
        raw_properties = self._fetch_raw_properties(
            query, check_in_date, check_out_date, adults, currency
        )
        if not raw_properties:
            return HotelsAgentResponse(
                status="no_results",
                error_message="No hotels returned from SerpAPI.",
                search_criteria={
                    "query": query, "check_in": check_in_date,
                    "check_out": check_out_date, "guests": adults,
                },
            )

        # ── 2. Hydrate all hotels in Python (no LLM) ─────────────────────────
        location   = _extract_location(query)
        all_hotels: list[HotelOption] = []
        for prop in raw_properties:
            hotel = _hydrate_hotel(prop, check_in_date, check_out_date, currency)
            hotel.location = location
            all_hotels.append(hotel)
        # ── 2b. Drop hotels with no usable price ─────────────────────────────
        paired = [
            (prop, hotel)
            for prop, hotel in zip(raw_properties, all_hotels)
            if hotel.price_per_night and hotel.price_per_night > 0   
        ]
        if not paired:
            return HotelsAgentResponse(
                status="no_results",
                error_message="No hotels with a known price were found.",
                search_criteria={
                    "query": query, "check_in": check_in_date,
                    "check_out": check_out_date, "guests": adults,
                },
            )
        raw_properties, all_hotels = map(list, zip(*paired)) 

        # ── 3. Parse user preferences in Python (no LLM) ─────────────────────
        detected_prefs = user_preferences

        # ── 4. Build compact indexed summaries for LLM ────────────────────────
        indexed = [
            _build_hotel_summary(prop, i, budget_per_night)
            for i, prop in enumerate(raw_properties)
        ]
        budget_str = (
            f"Budget: up to {budget_per_night} {currency}/night"
            if budget_per_night > 0 else "Budget: no constraint"
        )
        input_msg = (
            f"Rank these {len(indexed)} hotels.\n\n"
            f"Query: {query}\n"
            f"{budget_str}\n"
            f"{'Preferences: ' + user_preferences if user_preferences else 'No preferences stated.'}\n\n"
            f"{json.dumps(indexed, indent=2)}\n\n"
            "Return: status, recommended_index, alternative_indices (up to 4, best-first), reasoning."
        )

        # ── 5. LLM ranking call ───────────────────────────────────────────────
        result  = self.agent.invoke({"messages": [{"role": "user", "content": input_msg}]})
        ranking: HotelRankingResponse = result["structured_response"]

        if ranking.status == "no_results":
            return HotelsAgentResponse(
                status="no_results",
                error_message=ranking.error_message,
                total_results=len(all_hotels),
                all_hotels=all_hotels,
                search_criteria={
                    "query": query, "check_in": check_in_date,
                    "check_out": check_out_date, "guests": adults,
                },
            )

        # ── 6. Assemble recommended + alternatives ────────────────────────────
        rec_idx = ranking.recommended_index
        recommended_hotel = (
            all_hotels[rec_idx]
            if rec_idx is not None and rec_idx < len(all_hotels)
            else None
        )
        # Get up to 4 alternatives, excluding the recommended hotel and any out-of-range indices
        alternative_options = [
            all_hotels[i]
            for i in (ranking.alternative_indices or [])
            if i < len(all_hotels) and i != rec_idx
        ][:4]

        # ── 7. Market summary — pure Python ──────────────────────────────────
        prices  = [h.price_per_night for h in all_hotels if h.price_per_night]
        ratings = [h.overall_rating  for h in all_hotels if h.overall_rating]
        nights  = _nights_between(check_in_date, check_out_date)

        price_range = {
            "min":     round(min(prices),               2),
            "max":     round(max(prices),               2),
            "average": round(sum(prices) / len(prices), 2),
        } if prices else None

        rating_range = {
            "min":     round(min(ratings),                2),
            "max":     round(max(ratings),                2),
            "average": round(sum(ratings) / len(ratings), 2),
        } if ratings else None

        response = HotelsAgentResponse(
            status                    = ranking.status,
            recommended_hotel         = recommended_hotel,
            alternative_options       = alternative_options,
            all_hotels                = all_hotels,
            total_results             = len(all_hotels),
            search_criteria           = {
                "location":  location,
                "check_in":  check_in_date,
                "check_out": check_out_date,
                "nights":    nights,
                "guests":    adults,
                "currency":  currency,
            },
            detected_user_preferences = detected_prefs,
            price_range               = price_range,
            rating_range              = rating_range,
            reasoning                 = ranking.reasoning,
        )

        if response:
            h = response.recommended_hotel
            self.logger.info(
                "HotelAgent done: status=%s  recommended=%s  alts=%d  total=%d",
                response.status,
                f"{h.name} ★{h.overall_rating} €{h.price_per_night}/night" if h else "none",
                len(response.alternative_options),
                response.total_results,
            )
            self.logger.debug("Hotel reasoning: %s", response.reasoning)

        return response