"""
itinerary_agent.py
------------------
Itinerary agent built with create_agent.
Takes the single recommended flight, hotel, and attractions list
plus user preferences and builds a structured day-by-day itinerary.

The agent does NOT call any tools, it purely reasons over the
structured data passed to it from the other agents.
"""

from typing import Optional, List

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from src.logger import get_logger, log_agent_run

from src.models import (
    ItineraryAgentResponse,
    FlightOption,
    HotelOption,
    Attraction,
)


class ItineraryAgent:
    """
    Itinerary Agent — builds a day-by-day trip plan from agent outputs.

    Usage:
        agent = ItineraryAgent()
        itinerary = agent.build(
            outbound_flight=flight_response.recommended_flight,
            return_flight=flight_response.recommended_return_flight,
            hotel=hotel_response.recommended_hotel,
            attractions=attractions_response.recommended_attractions,
            destination="Madeira",
            user_preferences="Hiking and nature. Budget conscious.",
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
        self.logger  = get_logger("agent.itinerary")

        self.agent = create_agent(
            model=self.llm,
            tools=[],
            system_prompt=self._create_system_prompt(),
            response_format=ItineraryAgentResponse,
            name="itinerary_agent",
        )

    # ── System prompt ────────────────────────────────────────────────────
    def _create_system_prompt(self) -> str:
        return """You are a Travel Itinerary Planning Agent.

You receive a recommended flight, hotel, and attractions list.
Your job is to synthesise this into a logical, enjoyable day-by-day itinerary.

STRUCTURE RULES:
- Day 1 is always arrival day:
  • Include the outbound flight details (departure → arrival time).
  • Add hotel check-in as an activity (use hotel check_in_time).
  • Only add light activities after arrival — respect travel fatigue.
- Last day is always departure day:
  • Include hotel check-out (use hotel check_out_time).
  • Only add activities that fit before the return flight departure.
  • Include the return flight as the final activity.
- Middle days are full activity days:
  • Schedule 3-4 attractions per day from the provided attractions list.
  • Spread attraction categories across days — don't cluster all hikes on one day.
  • Prefer walkable/nearby attractions (low distance_minutes) on days after long travel.
  • Respect open_status — do not schedule "Closed" attractions without noting the risk.

ACTIVITY RULES:
- Assign realistic times. A typical day runs 09:00–18:00 for activities.
- Allow 30-60 min travel time between attractions that aren't walkable.
- Assign duration_minutes per attraction type:
    Viewpoint      → 30-45 min
    Nature/Hike    → 120-180 min
    Culture/Museum → 60-90 min
    Beach/Pool     → 120+ min
    Cable Car      → 60 min including travel
- Include a lunch break (type: "free_time") around 13:00 on full days.
- Add practical notes where relevant:
    "Book tickets in advance", "Bring sunscreen", "Check weather before hiking".

FLIGHT ACTIVITY RULES:
- Use outbound flight for Day 1 activity.
- Use return flight for last day final activity.
- Flight activity title: "<Airline> <Flight Number> — <Origin> → <Destination>"
- Include departure_time and arrival_time in description.

HOTEL ACTIVITY RULES:
- Day 1: type="hotel", title="Check-in: <hotel name>", time=check_in_time.
- Last day: type="hotel", title="Check-out: <hotel name>", time=check_out_time.

COST RULES:
- estimated_daily_cost = sum of known activity costs per day.
- cost_breakdown = {
    'flights': outbound price ONLY — for round-trips this is already
               the total round-trip price. Do NOT add return price again.
    'hotel': hotel total_price,
    'attractions': sum of attraction entry prices,
    'total': all three combined
  }
- total_estimated_cost = cost_breakdown total.

PREFERENCE RULES:
- "relaxed" / "slow pace" → max 2-3 attractions per day, add more free_time blocks.
- "budget" → prefer is_free attractions, note paid ones clearly.
- "hiking" / "nature" → front-load nature/adventure days in the middle of the trip.
- "culture" → include at least one culture-focused day in the middle days.
- If no preference stated, balance pace and category variety across all days.

OUTPUT RULES:
- trip_title should be catchy and specific e.g. "8-Day Madeira Nature & Culture Escape".
- highlights: pick 3-5 standout moments from the itinerary.
- practical_tips: 3-5 general tips relevant to the destination and activities chosen.
- reasoning: explain how flight times, hotel location, days, and preferences
  shaped the structure.
"""

    # ── Helper: format flight context ────────────────────────────────────
    def _format_flight_context(
        self,
        outbound: Optional[FlightOption],
        returning: Optional[FlightOption],
    ) -> str:
        if not outbound:
            return "No flight data provided."

        outbound_str = (
            f"Outbound: {outbound.airline} {outbound.flight_number or ''} | "
            f"{outbound.departure_airport_code} → {outbound.arrival_airport_code} | "
            f"Departs: {outbound.departure_time} | Arrives: {outbound.arrival_time} | "
            f"Price: {outbound.price} {outbound.currency}"
        )

        return_str = "Return: Not provided."
        if returning:
            return_str = (
                f"Return:   {returning.airline} {returning.flight_number or ''} | "
                f"{returning.departure_airport_code} → {returning.arrival_airport_code} | "
                f"Departs: {returning.departure_time} | Arrives: {returning.arrival_time} | "
                f"Price: INCLUDED in outbound price"
            )

        return f"{outbound_str}\n{return_str}"

    # ── Helper: format hotel context ─────────────────────────────────────
    def _format_hotel_context(self, hotel: Optional[HotelOption]) -> str:
        if not hotel:
            return "No hotel data provided."

        return (
            f"Name: {hotel.name} | Location: {hotel.location} | "
            f"Check-in: {hotel.check_in_date} at {hotel.check_in_time} | "
            f"Check-out: {hotel.check_out_date} at {hotel.check_out_time} | "
            f"Total: {hotel.total_price} {hotel.currency} ({hotel.nights} nights)"
        )

    # ── Helper: format attractions context ───────────────────────────────
    def _format_attractions_context(self, attractions: List[Attraction]) -> str:
        if not attractions:
            return "No attractions data provided."

        lines = []
        for a in attractions:
            price_str = f"${a.price}" if a.price else "Free/Unknown"
            dist_str = (
                f"{a.distance_minutes} min by {a.distance_source}"
                if a.distance_minutes else "Distance unknown"
            )
            lines.append(
                f"  - [{a.category}] {a.name} | "
                f"Rating: {a.rating} ({a.reviews_count} reviews) | "
                f"Price: {price_str} | {dist_str} | "
                f"Status: {a.open_status or 'Unknown'}"
            )

        return "\n".join(lines)

    # ── Main entry point ─────────────────────────────────────────────────
    @log_agent_run("ItineraryAgent")
    def build(
        self,
        destination: str,
        outbound_flight: Optional[FlightOption] = None,
        return_flight: Optional[FlightOption] = None,
        hotel: Optional[HotelOption] = None,
        attractions: Optional[List[Attraction]] = None,
        user_preferences: str = "",
        currency: str = "EUR",
    ) -> ItineraryAgentResponse:
        """
        Build a day-by-day itinerary from recommended options.

        Args:
            destination:      Primary destination e.g. "Madeira"
            outbound_flight:  flight_response.recommended_flight
            return_flight:    flight_response.recommended_return_flight
            hotel:            hotel_response.recommended_hotel
            attractions:      attractions_response.recommended_attractions
            user_preferences: Free-text preferences e.g. "relaxed pace, budget"
            currency:         Currency code for cost display (default "EUR")

        Returns:
            ItineraryAgentResponse with a full day-by-day itinerary.
        """
        attractions = attractions or []
        days = hotel.nights if hotel else len(attractions) // 4 or 1

        input_msg = f"""Build a day-by-day itinerary.

Destination: {destination}
Total days:  {days}
Currency:    {currency}
{("User preferences: " + user_preferences) if user_preferences else "No specific preferences stated."}

── FLIGHT DATA ──────────────────────────────────────────
{self._format_flight_context(outbound_flight, return_flight)}

── HOTEL DATA ───────────────────────────────────────────
{self._format_hotel_context(hotel)}

── RECOMMENDED ATTRACTIONS ──────────────────────────────
{self._format_attractions_context(attractions)}

Instructions:
- Build one ItineraryDay per day (Day 1 = arrival, Day {days} = departure).
- Assign activities in logical time order within each day.
- Return a complete ItineraryAgentResponse.
"""

        self.logger.info(
            "Building itinerary: %s | %d days | %d attractions%s%s",
            destination, days, len(attractions),
            f" | flight={outbound_flight.airline} {outbound_flight.flight_number or ''}" if outbound_flight else "",
            f" | hotel={hotel.name}" if hotel else "",
        )
        if user_preferences:
            self.logger.debug("Itinerary preferences: %s", user_preferences)

        result = self.agent.invoke({
            "messages": [{"role": "user", "content": input_msg}]
        })

        response: ItineraryAgentResponse = result["structured_response"]

        if response:
            self.logger.info(
                "ItineraryAgent done: status=%s | title=%s | days=%d%s",
                response.status, response.trip_title, len(response.days),
                f" | cost={response.total_estimated_cost} {currency}" if response.total_estimated_cost else "",
            )
            self.logger.debug("Itinerary reasoning: %s", response.reasoning)

        return response