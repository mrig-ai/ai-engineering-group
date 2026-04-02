"""
Flight Search Tools - Outbound + Return flight search via SerpAPI.
"""
from typing import List, Dict
from langchain.tools import tool
import os
import re
import time
from dotenv import load_dotenv
from src.logger import get_logger, log_api_request, log_api_response
load_dotenv()

_logger = get_logger("api.serpapi.flights")

try:
    from serpapi import GoogleSearch
    SERPAPI_AVAILABLE = True
except ImportError:
    SERPAPI_AVAILABLE = False

# Mapping from readable names to SerpAPI's integer codes
TRAVEL_CLASS_MAP = {
    "economy": 1,
    "premium_economy": 2,
    "business": 3,
    "first": 4,
}


@tool
def search_flights(
    departure_airport: str,
    arrival_airport: str,
    departure_date: str,
    return_date: str = "",
    travel_class: str = "economy",
    max_results: int = 10,
    adults: int = 1,
) -> List[Dict]:
    """
    Search OUTBOUND flights using Google Flights via SerpAPI.

    For round-trip searches (return_date provided), each result includes
    a `departure_token` field. Pass that token to `search_return_flights`
    to get matching return flights for a specific outbound.

    Args:
        departure_airport: 3-letter IATA code (e.g., "JFK")
        arrival_airport: 3-letter IATA code (e.g., "CDG")
        departure_date: YYYY-MM-DD format
        return_date: YYYY-MM-DD for round-trip, or "" for one-way
        travel_class: "economy", "premium_economy", "business", or "first"
        max_results: Max flights to return (default: 10)
        adults: Number of adult passengers (default: 1)

    Returns:
        List of flight dicts, each with a `departure_token` for round-trips.
    """
    if not SERPAPI_AVAILABLE or not os.environ.get("SERPAPI_KEY"):
        return "SerpAPI not available or API key missing. Please set SERPAPI_KEY."

    try:
        params = {
            "engine": "google_flights",
            "departure_id": departure_airport.upper(),
            "arrival_id": arrival_airport.upper(),
            "outbound_date": departure_date,
            "currency": "EUR",
            "api_key": os.environ.get("SERPAPI_KEY"),
            "type": "1" if return_date else "2",
            "adults": adults,
            "travel_class": TRAVEL_CLASS_MAP.get(travel_class.lower(), 1),
        }
        if return_date:
            params["return_date"] = return_date

        log_api_request(_logger, "serpapi.google_flights", params)
        t0 = time.perf_counter()
        search = GoogleSearch(params)
        results = search.get_dict()
        duration_ms = (time.perf_counter() - t0) * 1000

        flights = []
        if "best_flights" in results:
            flights.extend(_parse(results["best_flights"]))
        if "other_flights" in results and len(flights) < max_results:
            flights.extend(_parse(results["other_flights"]))

        parsed = flights[:max_results]
        log_api_response(_logger, "serpapi.google_flights", parsed, duration_ms)
        return parsed
    except Exception as e:
        _logger.error("Flight search failed: %s", e, exc_info=True)
        return f"Flight search failed: {str(e)}"


@tool
def search_return_flights(
    departure_airport: str,
    arrival_airport: str,
    outbound_date: str,
    return_date: str,
    departure_token: str,
    max_results: int = 10,
    adults: int = 1,
) -> List[Dict]:
    """
    Search RETURN flights for a specific outbound selection.

    Use the `departure_token` from a flight returned by `search_flights`
    along with the original route and dates to get matching return flights.

    Args:
        departure_airport: Original departure 3-letter IATA code (e.g., "JFK")
        arrival_airport: Original arrival 3-letter IATA code (e.g., "CDG")
        outbound_date: Original outbound date in YYYY-MM-DD format
        return_date: Return date in YYYY-MM-DD format
        departure_token: Token from the chosen outbound flight's
                         `departure_token` field.
        max_results: Max return flights to return (default: 10)

    Returns:
        List of return flight dicts.
    """
    if not SERPAPI_AVAILABLE or not os.environ.get("SERPAPI_KEY"):
        return "SerpAPI not available or API key missing. Please set SERPAPI_KEY."

    try:
        params = {
            "engine": "google_flights",
            "departure_id": departure_airport.upper(),
            "arrival_id": arrival_airport.upper(),
            "outbound_date": outbound_date,
            "return_date": return_date,
            "type": "1",
            "departure_token": departure_token,
            "currency": "EUR",
            "adults": adults,
            "api_key": os.environ.get("SERPAPI_KEY"),
        }

        log_api_request(_logger, "serpapi.google_flights.return", params)
        t0 = time.perf_counter()
        search = GoogleSearch(params)
        results = search.get_dict()
        duration_ms = (time.perf_counter() - t0) * 1000

        flights = []
        if "best_flights" in results:
            flights.extend(_parse(results["best_flights"]))
        if "other_flights" in results and len(flights) < max_results:
            flights.extend(_parse(results["other_flights"]))

        parsed = flights[:max_results]
        log_api_response(_logger, "serpapi.google_flights.return", parsed, duration_ms)
        return parsed
    except Exception as e:
        _logger.error("Return flight search failed: %s", e, exc_info=True)
        return f"Return flight search failed: {str(e)}"


# ── Shared parser ────────────────────────────────────────────────────────

def _parse(flights: List[Dict]) -> List[Dict]:
    """Parse SerpAPI flight results with full leg, layover, and transit details."""
    parsed = []
    for flight in flights:
        try:
            legs_raw = flight.get("flights", [])
            if not legs_raw:
                continue

            first, last = legs_raw[0], legs_raw[-1]
            stops = len(legs_raw) - 1

            dep = first.get("departure_airport", {})
            arr = last.get("arrival_airport", {})

            # Total duration covers all legs + layovers
            total_min = flight.get("total_duration", 0)
            if not total_min:
                total_min = sum(l.get("duration", 0) for l in legs_raw)
            duration_h = round(total_min / 60, 1) if total_min else 0

            # ── Parse each leg ───────────────────────────────────────
            legs_parsed = []
            for leg in legs_raw:
                leg_dep = leg.get("departure_airport", {})
                leg_arr = leg.get("arrival_airport", {})

                leg_exts = leg.get("extensions", [])
                leg_amenities = []
                leg_legroom = leg.get("legroom", "")

                for ext in leg_exts:
                    ext_lower = str(ext).lower()
                    if "carbon" in ext_lower:
                        continue
                    elif "legroom" in ext_lower and not leg_legroom:
                        match = re.search(r'(\d+\s*in)', ext)
                        if match:
                            leg_legroom = match.group(1)
                    else:
                        leg_amenities.append(ext)

                legs_parsed.append({
                    "departure_airport_code": leg_dep.get("id", ""),
                    "departure_airport_name": leg_dep.get("name", ""),
                    "departure_time": leg_dep.get("time", ""),
                    "arrival_airport_code": leg_arr.get("id", ""),
                    "arrival_airport_name": leg_arr.get("name", ""),
                    "arrival_time": leg_arr.get("time", ""),
                    "duration_minutes": leg.get("duration"),
                    "airline": leg.get("airline", ""),
                    "flight_number": leg.get("flight_number", ""),
                    "airplane": leg.get("airplane", ""),
                    "travel_class": leg.get("travel_class", "Economy"),
                    "legroom": leg_legroom,
                    "amenities": leg_amenities,
                    "overnight": leg.get("overnight", False),
                })

            # ── Parse layovers ───────────────────────────────────────
            layovers_parsed = []
            for layover in flight.get("layovers", []):
                layovers_parsed.append({
                    "airport_code": layover.get("id", ""),
                    "airport_name": layover.get("name", ""),
                    "duration_minutes": layover.get("duration"),
                })

            # ── Carbon emissions (flight-level) ──────────────────────
            carbon_raw = flight.get("carbon_emissions", {})
            if isinstance(carbon_raw, dict):
                carbon_kg = carbon_raw.get("this_flight")
                carbon = f"{round(carbon_kg / 1000)} kg" if carbon_kg else None
            else:
                carbon = None

            # ── First-leg amenities for top-level fields ─────────────
            first_exts = first.get("extensions", [])
            amenities, legroom = [], first.get("legroom", "")
            for ext in first_exts:
                ext_lower = str(ext).lower()
                if "carbon" in ext_lower:
                    continue
                elif "legroom" in ext_lower and not legroom:
                    match = re.search(r'(\d+\s*in)', ext)
                    if match:
                        legroom = match.group(1)
                else:
                    amenities.append(ext)

            flight_number = first.get("flight_number", "")
            option_id = (
                f"{first.get('airline', 'XX')[:2].upper()}"
                f"{flight_number.replace(' ', '') or str(len(parsed) + 1)}"
            )
            # Skip results with no price, SerpAPI returned incomplete data
            price = float(flight.get("price", 0))
            if not price:
                continue
            parsed.append({
                "option_id": option_id,
                "airline": first.get("airline", "Unknown"),
                "price": float(flight.get("price", 0)),
                "currency": "EUR",
                "type": {0: "direct", 1: "1_stop"}.get(stops, "2_stop"),
                "total_duration_hours": float(duration_h),
                "departure_airport_code": dep.get("id", ""),
                "departure_airport_name": dep.get("name", ""),
                "departure_time": dep.get("time", ""),
                "arrival_airport_code": arr.get("id", ""),
                "arrival_airport_name": arr.get("name", ""),
                "arrival_time": arr.get("time", ""),
                "flight_number": flight_number,
                "airplane": first.get("airplane", ""),
                "travel_class": first.get("travel_class", "Economy"),
                "legroom": legroom,
                "amenities": amenities,
                "legs": legs_parsed,
                "layovers": layovers_parsed,
                "ticket_also_sold_by": first.get("ticket_also_sold_by", []),
                "carbon_emissions": carbon,
                "departure_token": flight.get("departure_token"),
            })
        except:
            continue
    return parsed