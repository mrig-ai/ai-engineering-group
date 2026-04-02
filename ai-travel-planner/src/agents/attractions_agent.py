"""
attractions_agent.py
--------------------
Attractions recommendation agent — minimal LLM output approach.

REAL SerpAPI description field shapes (from observed data):
  - "Open"             → open status only, no descriptive content
  - "Closed"           → closed status only, no descriptive content
  - "Garden"           → descriptive label only, no status
  - "Opera house"      → descriptive label only, no status
  - "Cultural center"  → descriptive label only, no status
  - "Mountain peak"    → descriptive label only, no status
  - "Neighborhood"     → descriptive label only, no status
  - absent / ""        → neither

  i.e. description is EITHER a status word OR a short category label — never both.

WHAT THE LLM DOES:
  Receives indexed list of {i, title, description, rating, reviews, free, price?}.
  Outputs ONLY: status, recommended_indices, alternative_indices, reasoning.

WHAT PYTHON DOES (zero LLM tokens):
  - name, rating, reviews_count, price, is_free  ← direct field mapping
  - open_status                                  ← exact match on "Open"/"Closed"
  - description                                  ← raw description when NOT a status word
  - category                                     ← keyword table on title + description label
  - distance_minutes, distance_source            ← fuzzy match against hotel nearby_places
  - estimated_total_cost, hotel_context_used     ← derived from above
"""

import json
import re
from typing import Optional

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from src.logger import get_logger, log_agent_run
from src.models import (
    Attraction,
    AttractionAgentResponse,
    AttractionRankingResponse,
    HotelOption,
)
from src.tools.attraction_tools import search_attractions


# ── Status parsing ────────────────────────────────────────────────────────────

def _parse_description(raw: Optional[str]) -> tuple[Optional[str], Optional[str]]:
    """
    Split the SerpAPI description field into (open_status, description).

    SerpAPI description can take many forms in the wild:
      "Open"                       → status only
      "Closed"                     → status only
      "Garden"                     → label only
      "Historic site · Open"       → label + status mixed
      "Open · Closes 6 PM"         → status + detail
      "Natural reserve · Closed"   → label + status mixed
      "Waterfall in the mountains" → pure description, no status

    Strategy:
      1. Detect "open" or "closed" as whole words anywhere in the string.
      2. Strip all status-bearing tokens (handles "· Open", "Open ·", etc.)
      3. Whatever remains is the description label; None if empty.
    """
    if not raw or not raw.strip():
        return None, None

    text = raw.strip()

    # Detect status — whole-word match, case-insensitive
    open_status: Optional[str] = None
    if re.search(r"\bclosed\b", text, re.IGNORECASE):
        open_status = "Closed"
    elif re.search(r"\bopen\b", text, re.IGNORECASE):
        open_status = "Open"

    # Strip status tokens and surrounding separators (·, •, –, -, |, comma)
    # Also strip trailing time details like "· Closes 6 PM" that follow the status word.
    clean = re.sub(
        r"[\s·•–\-|,]*\b(?:open|closed)\b(?:\s*[·•–\-|,]?\s*[^\s·•–\-|,]+)*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip(" ·•–-|,")

    return open_status, clean or None


# ── Category keyword table ────────────────────────────────────────────────────
# Checked in order — first match wins.
# Sources: attraction title AND the description label (when it's not a status word).

_CATEGORY_RULES: list[tuple[str, list[str]]] = [
    ("Beach",        ["beach", "bay", "swimming", "praia", "plage"]),
    ("Viewpoint",    ["miradouro", "viewpoint", "belvedere", "skywalk", "cliff",
                      "lookout", "mountain peak", "peak"]),
    ("Adventure",    ["levada", "hike", "hiking", "cable car", "zip", "trail", "trek"]),
    ("Nature",       ["park", "garden", "forest", "waterfall", "lagoon", "natural pool",
                      "reserve", "botanical"]),
    ("Culture",      ["museum", "musée", "musee", "market", "church", "cathedral",
                      "basilica", "chapel", "historic", "heritage", "wine", "lodge",
                      "palace", "fort", "château", "opera", "opera house", "cemetery",
                      "neighbourhood", "neighborhood", "cultural center", "cultural centre",
                      "library", "bookshop", "bookstore"]),
    ("Food & Drink", ["restaurant", "food", "taste", "tasting", "cafe", "bar", "winery",
                      "bistro", "brasserie"]),
]

def _infer_category(title: str, description_label: Optional[str]) -> Optional[str]:
    """
    Keyword-match against title + description label.
    description_label is the non-status description (e.g. "Garden", "Opera house").
    """
    text = (title + " " + (description_label or "")).lower()
    for category, keywords in _CATEGORY_RULES:
        if any(kw in text for kw in keywords):
            return category
    return None


# ── Hotel distance matching ───────────────────────────────────────────────────

def _match_distance(
    attraction_name: str,
    hotel: Optional[HotelOption],
) -> tuple[Optional[int], Optional[str]]:
    if not hotel or not hotel.nearby_places:
        return None, None
    for place in hotel.nearby_places:
        if place.name.lower() == attraction_name.lower():
            source = place.transportation_types[0] if place.transportation_types else None
            return place.distance_minutes, source
    return None, None


# ── Agent ─────────────────────────────────────────────────────────────────────

class AttractionsAgent:
    """
    Attractions Agent — searches and recommends sights for a destination.

    Usage:
        agent = AttractionsAgent()
        response = agent.find_attractions(
            location="Paris",
            days=3,
            user_preferences="Art and culture. Some free attractions.",
            hotel=hotel_response.recommended_hotel,
        )
    """

    def __init__(
        self,
        model_name: str = "gpt-5-nano",
        temperature: float = 0.0,
        verbose: bool = True,
    ):
        self.llm = ChatOpenAI(model=model_name, temperature=temperature)
        self.verbose = verbose
        self.logger  = get_logger("agent.attractions")

        self.agent = create_agent(
            model=self.llm,
            tools=[],
            system_prompt=self._create_system_prompt(),
            response_format=AttractionRankingResponse,
            name="attractions_ranking_agent",
        )

    # ── System prompt ─────────────────────────────────────────────────────────
    def _create_system_prompt(self) -> str:
        return """You are a Travel Attractions Ranking Agent.

You receive an indexed list of attractions. Your ONLY job is to decide which
indices to recommend and which to leave as alternatives.

OUTPUT — four fields only:
  status:               "success" or "no_results"
  recommended_indices:  ordered list of indices, best-first, ~3-4 per day
  alternative_indices:  all remaining indices (every index must appear in one list)
  reasoning:            2-4 sentences referencing days, preferences, ratings

DO NOT output names, descriptions, categories, prices, or any other fields.
Python handles all field mapping — your only value-add is the ranking decision.

RANKING RULES:
  - Prefer rating >= 4.3 with > 1000 reviews.
  - User preferences:
    • "art" / "culture" / "history"  → favour museums, cathedrals, galleries
    • "nature" / "outdoors"          → favour parks, gardens, natural sites
    • "budget" / "free"              → favour attractions with free=true in the data
    • "views"                        → favour viewpoints, hilltops
    • "food"                         → favour markets, food areas
  - Balance across types when no preference stated.
  - Palace of Versailles and similar day-trip destinations: include but note
    they consume a full day (count as ~4 slots).

SCALING:
  1 day → up to 4 | 3 days → up to 12 | 5 days → up to 16 | 8 days → up to 20
"""

    # ── Fetch raw sights ──────────────────────────────────────────────────────
    def _fetch_raw_sights(self, location: str) -> list[dict]:
        raw = search_attractions.invoke(f"Top sights in {location}")
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            return []
        return data if isinstance(data, list) else []

    # ── Hydrate one Attraction (pure Python) ──────────────────────────────────
    def _hydrate_attraction(
        self,
        sight: dict,
        hotel: Optional[HotelOption],
    ) -> Attraction:
        raw_desc = sight.get("description", "")
        open_status, description_label = _parse_description(raw_desc)

        distance_minutes, distance_source = _match_distance(
            sight.get("title", ""), hotel
        )

        return Attraction(
            name=sight.get("title", "Unknown"),
            rating=sight.get("rating"),
            reviews_count=sight.get("reviews"),
            price=sight.get("extracted_price"),       # None when key absent (free/unknown)
            is_free="extracted_price" not in sight,
            # thumbnail=sight.get("thumbnail"),
            open_status=open_status,
            description=description_label,            # None when description was a status word
            category=_infer_category(
                sight.get("title", ""), description_label
            ),
            distance_minutes=distance_minutes,
            distance_source=distance_source,
        )

    # ── Main entry point ──────────────────────────────────────────────────────
    @log_agent_run("AttractionsAgent")
    def find_attractions(
        self,
        location: str,
        days: int = 1,
        user_preferences: str = "",
        hotel: Optional[HotelOption] = None,
        currency: str = "EUR",
    ) -> AttractionAgentResponse:
        """
        Search and recommend attractions for a destination.

        Args:
            location:         Destination name e.g. "Paris"
            days:             Number of days visiting. Scales recommendation count.
            user_preferences: Free-text preferences e.g. "art, budget conscious"
            hotel:            Optional HotelOption whose nearby_places enrich distances.
            currency:         Currency code for cost estimates (default "EUR")

        Returns:
            AttractionAgentResponse — public interface unchanged.
        """

        # ── 1. Fetch + hydrate all attractions (no LLM yet) ───────────────────
        self.logger.info("Fetching attractions: %s", location)

        raw_sights = self._fetch_raw_sights(location)

        if not raw_sights:
            return AttractionAgentResponse(
                status="no_results",
                error_message="No attractions found from SerpAPI.",
                days=days,
            )

        all_attractions = [
            self._hydrate_attraction(sight, hotel)
            for sight in raw_sights
        ]

        # ── 2. Build compact indexed prompt ──────────────────────────────────
        # Send only ranking-relevant fields — nothing the LLM would just copy.
        indexed = [
            {
                "i": i,
                "title": s.get("title", ""),
                "rating": s.get("rating"),
                "reviews": s.get("reviews"),
                "free": "extracted_price" not in s,
                **({"price": s["extracted_price"]} if "extracted_price" in s else {}),
            }
            for i, s in enumerate(raw_sights)
        ]

        input_msg = (
            f"Rank these {len(indexed)} attractions.\n\n"
            f"Location: {location} | Days: {days} | Target: ~{days * 4} recommended\n"
            f"{'Preferences: ' + user_preferences if user_preferences else 'No preferences stated.'}\n\n"
            f"{json.dumps(indexed, indent=2)}\n\n"
            f"Return: status, recommended_indices (best-first, ~{days * 4}), "
            f"alternative_indices (all remaining), reasoning."
        )

        self.logger.info(
            "Ranking %d sights | days=%d%s%s",
            len(indexed), days,
            f" | hotel={hotel.name}" if hotel else "",
            f" | prefs={user_preferences}" if user_preferences else "",
        )

        # ── 3. LLM ranking call ───────────────────────────────────────────────
        result = self.agent.invoke({
            "messages": [{"role": "user", "content": input_msg}]
        })
        ranking: AttractionRankingResponse = result["structured_response"]

        if ranking.status == "no_results":
            return AttractionAgentResponse(
                status="no_results",
                error_message=ranking.error_message,
                days=days,
            )

        # ── 4. Partition into recommended / alternative ───────────────────────
        ranked_set = set(ranking.recommended_indices) | set(ranking.alternative_indices)

        recommended = [
            all_attractions[i]
            for i in ranking.recommended_indices
            if i < len(all_attractions)
        ]
        alternative = [
            all_attractions[i]
            for i in ranking.alternative_indices
            if i < len(all_attractions)
        ]
        # Safety net: any index the LLM silently dropped → alternative
        for i in range(len(all_attractions)):
            if i not in ranked_set:
                alternative.append(all_attractions[i])

        # ── 5. Derived fields ─────────────────────────────────────────────────
        known_prices = [a.price for a in recommended if a.price is not None]
        estimated_cost = round(sum(known_prices), 2) if known_prices else None
        hotel_context_used = any(a.distance_minutes is not None for a in all_attractions)

        response = AttractionAgentResponse(
            status="success",
            recommended_attractions=recommended,
            alternative_attractions=alternative,
            all_attractions=all_attractions,
            total_results=len(all_attractions),
            days=days,
            estimated_total_cost=estimated_cost,
            hotel_context_used=hotel_context_used,
            reasoning=ranking.reasoning,
        )

        self.logger.info(
            "AttractionsAgent done: status=%s  recommended=%d  alts=%d  total=%d",
            response.status,
            len(response.recommended_attractions),
            len(response.alternative_attractions),
            response.total_results,
        )
        self.logger.debug("Attractions reasoning: %s", response.reasoning)

        return response