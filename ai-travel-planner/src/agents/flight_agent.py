"""
flight_agent.py
---------------
Flight recommendation agent — two-phase LLM ranking using create_agent.

WORKFLOW:
  Phase 1 — Outbound:
    1. Python fetches outbound flights via search_flights
    2. Python hydrates all FlightOption objects
    3. LLM call 1 → ranks outbound → outputs recommended_index
    4. Python retrieves departure_token from the chosen outbound flight

  Phase 2 — Return (round-trip only):
    5. Python fetches return flights via search_return_flights using that token
    6. Python hydrates all return FlightOption objects
    7. LLM call 2 → ranks return → outputs recommended_return_index

  Phase 3 — Assemble:
    8. Python builds FlightAgentResponse from both sets of indices

WHAT THE LLM DOES (only):
  Receives compact indexed summaries → outputs status, recommended_index,
  alternative_indices, reasoning. FlightRankingResponse is reused for both
  calls; return fields are unused in the outbound call.

WHAT PYTHON DOES (zero LLM tokens):
  - All SerpAPI calls (_parse() in flight_tools handles field mapping)
  - FlightOption hydration
  - departure_token lookup and return API call
  - Final response assembly
  - Safety-net fallback (sorted by price if LLM skips recommendations)
"""

import json
from typing import Optional

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from src.logger import get_logger, log_agent_run
from src.models import (
    FlightAgentResponse,
    FlightOption,
    FlightRankingResponse,
)
from src.tools.flight_tools import search_flights, search_return_flights

logger = get_logger(__name__)


# ── Hydration ─────────────────────────────────────────────────────────────────

def _hydrate_flights(raw: list[dict]) -> list[FlightOption]:
    """
    Convert _parse() dicts to FlightOption objects.
    _parse() in flight_tools already handles all SerpAPI field mapping.
    """
    result = []
    for d in raw:
        try:
            result.append(FlightOption(**d))
        except Exception:
            continue
    return result


# ── Compact prompt summary ────────────────────────────────────────────────────

def _build_flight_summary(flight: FlightOption, i: int, budget: float) -> dict:
    """
    Minimal dict for LLM ranking prompt.
    Pre-computes layover total so LLM does not scan nested legs.
    """
    layover_min = sum(lay.duration_minutes or 0 for lay in flight.layovers)
    summary: dict = {
        "i":             i,
        "airline":       flight.airline,
        "type":          flight.type,           # "direct" | "1_stop" | "2_stop"
        "price":         flight.price,
        "duration_h":    flight.total_duration_hours,
        "dep":           flight.departure_time,
        "arr":           flight.arrival_time,
        "within_budget": budget <= 0 or flight.price <= budget,
    }
    if layover_min:          summary["layover_min"] = layover_min
    if flight.legroom:       summary["legroom"]     = flight.legroom
    if flight.amenities:     summary["amenities"]   = flight.amenities
    return summary


# ── Agent ─────────────────────────────────────────────────────────────────────

class FlightAgent:
    """
    Flight Agent — searches and recommends flights using two-phase LLM ranking.

    LLM call 1 ranks outbound flights. Python then uses the chosen flight's
    departure_token to fetch return options. LLM call 2 ranks those returns.

    Usage:
        agent = FlightAgent()
        response = agent.search_and_recommend(
            departure_airport="LIS",
            arrival_airport="FNC",
            departure_date="2026-03-23",
            return_date="2026-03-31",
            budget=600,
            adults=2,
            user_query="Direct flight preferred. Morning departure.",
        )
    """

    _OUTBOUND_SYSTEM_PROMPT = """You are a Flight Ranking Agent — Outbound Leg.

You receive compact indexed summaries of outbound flights.
Your ONLY job is to rank them and pick the best outbound.

OUTPUT — these fields only:
  status:              "success" | "over_budget" | "no_results"
  recommended_index:   single best outbound index (None if no_results)
  alternative_indices: up to 4 runner-up indices, best-first
  reasoning:           2-3 sentences on why this outbound was chosen
  error_message:       set only when status="no_results"

Leave recommended_return_index and alternative_return_indices empty.
DO NOT output airline names, prices, times, or any other flight fields.

RANKING RULES:
  - Budget here is the outbound share only (half of combined budget).
    If all outbound flights exceed budget → status "over_budget", recommend cheapest anyway.
  - If no flights returned → status "no_results".
  - Otherwise → status "success".
  - Direct flight is worth paying €200-300 more than 1-stop.
  - User preferences (honour all explicitly stated):
    • "no <airline>" / "avoid <airline>" → NEVER recommend that airline
    • "morning"       → prefer dep 06:00-11:59
    • "afternoon"     → prefer dep 12:00-17:59
    • "evening"       → prefer dep 18:00-23:59
    • "direct only"   → exclude all non-direct
    • "cheapest"      → price is king
    • "fastest"       → duration is king
    • "short layover" → prefer lower layover_min
    • "extra legroom" → prefer flights with legroom field present
  - Short layovers (<60 min) are risky; long layovers (>240 min) are inconvenient.
  - When no preference stated: balance price, duration, stops.
"""

    _RETURN_SYSTEM_PROMPT = """You are a Flight Ranking Agent — Return Leg.

You receive compact indexed summaries of return flights.
Your ONLY job is to rank them and pick the best return.

OUTPUT — these fields only:
  status:                     "success" | "over_budget" | "no_results"
  recommended_return_index:   single best return index (None if no_results)
  alternative_return_indices: up to 4 runner-up indices, best-first
  reasoning:                  2-3 sentences on why this return was chosen
  error_message:              set only when status="no_results"

Leave recommended_index and alternative_indices empty.
DO NOT output airline names, prices, times, or any other flight fields.

RANKING RULES:
  - Budget here is the remaining budget after the chosen outbound price.
    If all return flights exceed remaining budget → status "over_budget", recommend cheapest anyway.
  - If no flights returned → status "no_results".
  - Otherwise → status "success".
  - Apply the same airline/time/stop preferences as for outbound.
  - "same airline" preference → prefer same airline as the outbound (stated in prompt).
  - Short layovers (<60 min) are risky; long layovers (>240 min) are inconvenient.
  - When no preference stated: balance price, duration, stops.
"""

    def __init__(
        self,
        model_name: str = "gpt-4o-mini",
        temperature: float = 0.0,
        verbose: bool = True,
    ):
        self.verbose = verbose
        llm = ChatOpenAI(model=model_name, temperature=temperature)

        # Two separate agents — one per ranking phase, each with its own system prompt
        self._outbound_agent = create_agent(
            model=llm,
            tools=[],
            system_prompt=self._OUTBOUND_SYSTEM_PROMPT,
            response_format=FlightRankingResponse,
            name="flight_outbound_ranking_agent",
        )
        self._return_agent = create_agent(
            model=llm,
            tools=[],
            system_prompt=self._RETURN_SYSTEM_PROMPT,
            response_format=FlightRankingResponse,
            name="flight_return_ranking_agent",
        )

    # ── LLM calls ─────────────────────────────────────────────────────────────

    def _rank_outbound(self, input_msg: str) -> FlightRankingResponse:
        result = self._outbound_agent.invoke(
            {"messages": [{"role": "user", "content": input_msg}]}
        )
        return result["structured_response"]

    def _rank_return(self, input_msg: str) -> FlightRankingResponse:
        result = self._return_agent.invoke(
            {"messages": [{"role": "user", "content": input_msg}]}
        )
        return result["structured_response"]

    # ── Fetch helpers ─────────────────────────────────────────────────────────

    def _fetch_outbound(
        self,
        departure_airport: str,
        arrival_airport: str,
        departure_date: str,
        return_date: str,
        travel_class: str,
        adults: int,
    ) -> list[dict]:
        raw = search_flights.invoke({
            "departure_airport": departure_airport,
            "arrival_airport":   arrival_airport,
            "departure_date":    departure_date,
            "return_date":       return_date,
            "travel_class":      travel_class,
            "adults":            adults,
        })
        if isinstance(raw, str) or not raw:
            return []
        return raw

    def _fetch_return(
        self,
        departure_airport: str,
        arrival_airport: str,
        departure_date: str,
        return_date: str,
        departure_token: str,
        adults: int,
    ) -> list[dict]:
        raw = search_return_flights.invoke({
            "departure_airport": departure_airport,
            "arrival_airport":   arrival_airport,
            "outbound_date":     departure_date,
            "return_date":       return_date,
            "departure_token":   departure_token,
            "adults":            adults,
        })
        if isinstance(raw, str) or not raw:
            return []
        return raw

    # ── Main entry point ──────────────────────────────────────────────────────

    @log_agent_run("FlightAgent")
    def search_and_recommend(
        self,
        departure_airport: str,
        arrival_airport: str,
        departure_date: str,
        return_date: str = "",
        budget: float = 0,
        preference: str = "balanced",
        travel_class: str = "economy",
        adults: int = 1,
        user_query: str = "",
    ) -> FlightAgentResponse:
        """
        Search and recommend flights using two-phase LLM ranking.

        Args:
            departure_airport: 3-letter IATA code e.g. "LIS"
            arrival_airport:   3-letter IATA code e.g. "FNC"
            departure_date:    ISO date e.g. "2026-03-23"
            return_date:       ISO date for round-trip, "" for one-way
            budget:            Total budget for outbound + return combined. 0 = unlimited.
            preference:        "balanced" | "cheapest" | "fastest" | "direct"
            travel_class:      "economy" | "premium_economy" | "business" | "first"
            adults:            Number of adult passengers
            user_query:        Free-text user preferences

        Returns:
            FlightAgentResponse
        """
        is_round_trip = bool(return_date)
        prefs_line    = f"Preference: {preference}" + (f", {user_query}" if user_query else "")

        logger.info(
            "Searching flights: %s → %s | dep=%s ret=%s | budget=€%s class=%s adults=%d",
            departure_airport, arrival_airport,
            departure_date, return_date or "one-way",
            budget if budget > 0 else "unlimited",
            travel_class, adults,
        )
        if user_query:
            logger.debug("Flight preferences: %s", user_query)

        # ── Phase 1a: Fetch outbound flights ──────────────────────────────────
        raw_outbound = self._fetch_outbound(
            departure_airport, arrival_airport,
            departure_date, return_date,
            travel_class, adults,
        )

        if not raw_outbound:
            return FlightAgentResponse(
                status="no_results",
                reasoning="Flight search returned no outbound results.",
            )

        all_flights = _hydrate_flights(raw_outbound)
        logger.info("Outbound flights found: %d", len(all_flights))

        # ── Phase 1b: LLM call 1 — rank outbound ─────────────────────────────
        per_leg_budget = (budget / 2) if (is_round_trip and budget > 0) else budget

        outbound_summaries = [
            _build_flight_summary(f, i, per_leg_budget)
            for i, f in enumerate(all_flights)
        ]

        trip_type   = "round-trip" if is_round_trip else "one-way"
        budget_line = (
            f"Budget: €{per_leg_budget:.0f} for outbound leg"
            if budget > 0 else "Budget: unlimited"
        )

        outbound_msg = (
            f"Rank these outbound flights ({trip_type}).\n\n"
            f"Route: {departure_airport} → {arrival_airport} | Dep: {departure_date}\n"
            f"{budget_line} | {prefs_line}\n\n"
            f"OUTBOUND ({len(outbound_summaries)} flights):\n"
            f"{json.dumps(outbound_summaries, indent=2)}\n\n"
            f"Return: status, recommended_index, alternative_indices (up to 4), reasoning."
        )

        logger.info("Sending %d outbound summaries to LLM...", len(outbound_summaries))
        outbound_ranking: FlightRankingResponse = self._rank_outbound(outbound_msg)

        if outbound_ranking.status == "no_results":
            return FlightAgentResponse(
                status="no_results",
                all_flights=all_flights,
                reasoning=outbound_ranking.reasoning,
                error_message=outbound_ranking.error_message,
            )

        # ── Phase 1c: Resolve recommended outbound ────────────────────────────
        recommended_flight: Optional[FlightOption] = None
        alternative_options: list[FlightOption]    = []

        rec_idx = outbound_ranking.recommended_index
        if rec_idx is not None and rec_idx < len(all_flights):
            recommended_flight = all_flights[rec_idx]

        for i in outbound_ranking.alternative_indices:
            if i < len(all_flights) and i != rec_idx:
                alternative_options.append(all_flights[i])

        # Safety net: LLM skipped recommended — fall back to cheapest
        if not recommended_flight and all_flights:
            sorted_out         = sorted(all_flights, key=lambda f: f.price)
            recommended_flight = sorted_out[0]
            alternative_options = sorted_out[1:5]

        if recommended_flight:
            logger.info(
                "Outbound selected: %s %s — €%s (%s)",
                recommended_flight.airline,
                recommended_flight.flight_number,
                recommended_flight.price,
                recommended_flight.type,
            )

        # ── Phase 2a: Fetch return using chosen outbound's departure_token ─────
        all_return_flights: list[FlightOption]       = []
        recommended_return_flight: Optional[FlightOption] = None
        alternative_return_options: list[FlightOption]    = []
        return_reasoning = ""
        return_status    = "success"

        if is_round_trip:
            token = recommended_flight.departure_token if recommended_flight else None

            if token:
                raw_return         = self._fetch_return(
                    departure_airport, arrival_airport,
                    departure_date, return_date,
                    token, adults,
                )
                all_return_flights = _hydrate_flights(raw_return)
                logger.info("Return flights found: %d", len(all_return_flights))
            else:
                logger.warning("No departure_token on recommended outbound — skipping return fetch.")

            # ── Phase 2b: LLM call 2 — rank return ───────────────────────────
            if all_return_flights:
                remaining_budget = (
                    budget - recommended_flight.price
                    if (budget > 0 and recommended_flight)
                    else budget
                )
                return_summaries = [
                    _build_flight_summary(f, i, remaining_budget)
                    for i, f in enumerate(all_return_flights)
                ]

                outbound_airline = recommended_flight.airline if recommended_flight else "N/A"
                ret_budget_line  = (
                    f"Budget: €{remaining_budget:.0f} remaining after outbound "
                    f"(€{recommended_flight.price:.0f})"
                    if budget > 0 else "Budget: unlimited"
                )

                return_msg = (
                    f"Rank these return flights.\n\n"
                    f"Route: {arrival_airport} → {departure_airport} | Ret: {return_date}\n"
                    f"{ret_budget_line} | {prefs_line}\n"
                    f"Chosen outbound airline: {outbound_airline} "
                    f"(relevant for 'same airline' preference)\n\n"
                    f"RETURN ({len(return_summaries)} flights):\n"
                    f"{json.dumps(return_summaries, indent=2)}\n\n"
                    f"Return: status, recommended_return_index, "
                    f"alternative_return_indices (up to 4), reasoning."
                )

                logger.info("Sending %d return summaries to LLM...", len(return_summaries))
                return_ranking: FlightRankingResponse = self._rank_return(return_msg)
                return_reasoning = return_ranking.reasoning
                return_status    = return_ranking.status

                ret_idx = return_ranking.recommended_return_index
                if ret_idx is not None and ret_idx < len(all_return_flights):
                    recommended_return_flight = all_return_flights[ret_idx]

                for i in return_ranking.alternative_return_indices:
                    if i < len(all_return_flights) and i != ret_idx:
                        alternative_return_options.append(all_return_flights[i])

                # Safety net
                if not recommended_return_flight:
                    sorted_ret                 = sorted(all_return_flights, key=lambda f: f.price)
                    recommended_return_flight  = sorted_ret[0]
                    alternative_return_options = sorted_ret[1:5]

                if recommended_return_flight:
                    logger.info(
                        "Return selected: %s %s — €%s (%s)",
                        recommended_return_flight.airline,
                        recommended_return_flight.flight_number,
                        recommended_return_flight.price,
                        recommended_return_flight.type,
                    )

        # ── Phase 3: Assemble final response ──────────────────────────────────
        combined_reasoning = outbound_ranking.reasoning
        if return_reasoning:
            combined_reasoning += f" {return_reasoning}"

        final_status = outbound_ranking.status
        if is_round_trip and return_status == "over_budget":
            final_status = "over_budget"

        response = FlightAgentResponse(
            status=final_status,
            recommended_flight=recommended_flight,
            alternative_options=alternative_options,
            all_flights=all_flights,
            recommended_return_flight=recommended_return_flight,
            alternative_return_options=alternative_return_options,
            all_return_flights=all_return_flights,
            reasoning=combined_reasoning,
        )

        logger.info(
            "FlightAgent done: status=%s  outbound_alts=%d  return_alts=%d",
            response.status,
            len(response.alternative_options),
            len(response.alternative_return_options),
        )
        return response