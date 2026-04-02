"""
Pydantic models for the Travel Planning System.
Defines all data structures used across agents.
"""
from typing import List, Dict, Optional, Literal, Any
from pydantic import BaseModel, Field, field_validator, ConfigDict, model_validator


# ============================================================================
# EXPENSE & BUDGET MODELS
# ============================================================================

class Expense(BaseModel):
    """Individual expense item."""
    category: Literal["flights", "hotels", "activities", "other"]
    amount: float
    description: str


class BudgetAllocation(BaseModel):
    """Budget allocation across categories."""
    flights: float = 0.0
    hotels: float = 0.0
    activities: float = 0.0
    buffer: float = 0.0
    
    @property
    def total(self) -> float:
        """Calculate total allocated budget."""
        return self.flights + self.hotels + self.activities + self.buffer


# ============================================================================
# FLIGHT MODELS
# ============================================================================

class FlightLeg(BaseModel):
    """Single segment of a flight (e.g., JFK → KEF is one leg)."""
    departure_airport_code: Optional[str] = None
    departure_airport_name: Optional[str] = None
    departure_time: Optional[str] = None
    arrival_airport_code: Optional[str] = None
    arrival_airport_name: Optional[str] = None
    arrival_time: Optional[str] = None
    duration_minutes: Optional[int] = None
    airline: Optional[str] = None
    flight_number: Optional[str] = None
    airplane: Optional[str] = None
    travel_class: Optional[str] = None
    legroom: Optional[str] = None
    amenities: List[str] = Field(default_factory=list)
    overnight: bool = False


class Layover(BaseModel):
    """Layover between two flight legs."""
    airport_code: Optional[str] = None
    airport_name: Optional[str] = None
    duration_minutes: Optional[int] = None


class FlightOption(BaseModel):
    """
    Flight option with comprehensive details from SerpAPI.
    Includes pricing, duration, aircraft, amenities, legs, and layovers.
    """
    # Required fields
    option_id: str
    airline: str
    price: float
    currency: str = "EUR"
    type: Literal["direct", "1_stop", "2_stop", "return"]
    total_duration_hours: float
    
    # Overall departure / arrival (first leg depart → last leg arrive)
    departure_airport_code: Optional[str] = None
    departure_airport_name: Optional[str] = None
    departure_time: Optional[str] = None
    arrival_airport_code: Optional[str] = None
    arrival_airport_name: Optional[str] = None
    arrival_time: Optional[str] = None
    
    # Primary flight details (from first leg, for backward compatibility)
    flight_number: Optional[str] = None
    airplane: Optional[str] = None
    travel_class: Optional[str] = None
    legroom: Optional[str] = None
    amenities: List[str] = Field(default_factory=list)
    
    # Full leg-by-leg breakdown and layovers
    legs: List[FlightLeg] = Field(
        default_factory=list,
        description="All segments of this flight in order."
    )
    layovers: List[Layover] = Field(
        default_factory=list,
        description="Layover details between legs. Empty for direct flights."
    )
    
    # Additional info
    ticket_also_sold_by: List[str] = Field(default_factory=list)
    carbon_emissions: Optional[str] = None

    # Round-trip token
    departure_token: Optional[str] = Field(
        None,
        description=(
            "SerpAPI departure_token for this outbound flight. "
            "Pass to search_return_flights to get matching return options."
        )
    )
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "option_id": "ICFI614",
                "airline": "Icelandair",
                "price": 388,
                "currency": "EUR",
                "type": "1_stop",
                "total_duration_hours": 10.5,
                "departure_airport_code": "JFK",
                "departure_airport_name": "John F. Kennedy International Airport",
                "departure_time": "2026-04-15 20:30",
                "arrival_airport_code": "CDG",
                "arrival_airport_name": "Paris Charles de Gaulle Airport",
                "arrival_time": "2026-04-16 13:00",
                "flight_number": "FI 614",
                "airplane": "Airbus A321neo",
                "travel_class": "Economy",
                "legroom": "31 in",
                "amenities": ["Wi-Fi for a fee", "In-seat USB outlet", "On-demand video"],
                "legs": [
                    {
                        "departure_airport_code": "JFK",
                        "arrival_airport_code": "KEF",
                        "airline": "Icelandair",
                        "flight_number": "FI 614",
                        "duration_minutes": 350,
                    },
                    {
                        "departure_airport_code": "KEF",
                        "arrival_airport_code": "CDG",
                        "airline": "Icelandair",
                        "flight_number": "FI 542",
                        "duration_minutes": 205,
                    }
                ],
                "layovers": [
                    {
                        "airport_code": "KEF",
                        "airport_name": "Keflavík International Airport",
                        "duration_minutes": 75
                    }
                ],
                "carbon_emissions": "476 kg",
                "departure_token": "WyJDbUl..."
            }
        }



# ============================================================================
# HOTEL MODELS
# ============================================================================

class Amenity(BaseModel):
    """Hotel amenity with cost information."""
    name: str
    included: bool = True
    cost: Optional[float] = None

    @model_validator(mode="before")
    @classmethod
    def coerce_from_string(cls, v: Any) -> Any:
        """Allow the LLM to pass a plain string like 'Free Wi-Fi' or 'Parking ($)'."""
        if isinstance(v, str):
            return {
                "name": v.replace(" ($)", ""),
                "included": "($)" not in v,
            }
        return v

class ReviewBreakdown(BaseModel):
    model_config = ConfigDict(populate_by_name=True)
    category: str = Field(alias="name")
    positive: int
    negative: int
    neutral: int
    total_mentioned: int


class NearbyPlace(BaseModel):
    """Nearby attraction or location."""
    name: str
    distance_minutes: int
    transportation_types: List[str]  # e.g., ["Walking", "Taxi", "Public transport"]

class RatingDistribution(BaseModel):
    """Rating count distribution."""
    stars: int  # 1-5
    count: int

class CancellationPolicy(BaseModel):
    """Hotel cancellation policy."""
    free_cancellation: bool
    cancellation_deadline: Optional[str] = None  # e.g., "48 hours before"
    refund_percentage: Optional[float] = None
    policy_description: Optional[str] = None

class RoomInfo(BaseModel):
    """Room type information."""
    room_type: str  # e.g., "Double Room", "Suite", "Villa"
    capacity: int  # Number of guests
    beds: int
    amenities: List[str]
    price_per_night: Optional[float] = None

class HotelOption(BaseModel):
    """Comprehensive hotel option for travel planning chatbot."""
    
    # ===== BASIC INFO =====
    option_id: str
    name: str
    type: Literal["hotel", "resort", "vacation_rental", "villa", "hostel"]
    description: Optional[str] = None
    
    # ===== LOCATION & GEOGRAPHY =====
    location: str
    address: Optional[str] = None
    latitude: float
    longitude: float
    nearby_places: List[NearbyPlace] = Field(default_factory=list)
    
    # ===== PRICING =====
    currency: str = "EUR"
    price_per_night: float
    total_price: float
    nights: int
    check_in_date: str  # ISO format
    check_out_date: str  # ISO format
    check_in_time: Optional[str] = None  # e.g., "2:00 PM"
    check_out_time: Optional[str] = None  # e.g., "11:00 AM"
    deal_active: bool = False
    deal_discount_percentage: Optional[float] = None
    
    # ===== RATINGS & REVIEWS =====
    overall_rating: Optional[float] = Field(None, ge=0, le=5)
    total_reviews: int = 0
    rating_distribution: List[RatingDistribution] = Field(default_factory=list)
    location_rating: Optional[float] = Field(None, ge=0, le=5)
    review_breakdown: List[ReviewBreakdown] = Field(default_factory=list)
    
    # ===== AMENITIES =====
    amenities: List[Amenity] = Field(default_factory=list)
    excluded_amenities: List[str] = Field(default_factory=list)
    
    # ===== POLICIES =====
    cancellation_policy: Optional[CancellationPolicy] = None
    pet_friendly: bool = False
    smoking_allowed: bool = False
    wheelchair_accessible: bool = False
    kid_friendly: bool = True
    
    # ===== ROOM INFORMATION =====
    room_types: List[RoomInfo] = Field(default_factory=list)
    
    class Config:
        json_schema_extra = {
            "example": {
                "option_id": "HT001",
                "name": "ORA Resort Avia Blue Lagoon",
                "type": "resort",
                "location": "Bali",
                "latitude": -8.707362,
                "longitude": 115.43953,
                "currency": "EUR",
                "price_per_night": 68,
                "total_price": 542,
                "nights": 8,
                "check_in_date": "2026-03-23",
                "check_out_date": "2026-03-31",
                "overall_rating": 4.5,
                "total_reviews": 454,
                "amenities": [
                    {"name": "Free Wi-Fi", "included": True},
                    {"name": "Breakfast", "included": True, "cost": 0}
                ],
                "cancellation_policy": {
                    "free_cancellation": True,
                    "cancellation_deadline": "7 days before"
                }
            }
        }

# ============================================================================
# ATTRACTION MODEL
# ============================================================================

class Attraction(BaseModel):
    """
    A single tourist attraction or point of interest.

    Sourced from SerpAPI Google top_sights. Distance fields are
    optionally populated when hotel nearby_places are provided
    and a name match is found.
    """

    # ── Identity ──────────────────────────────────────────────────────────
    name: str = Field(description="Attraction name.")
    category: Optional[str] = Field(
        None,
        description=(
            "Inferred category. One of: 'Nature', 'Culture', "
            "'Adventure', 'Viewpoint', 'Food & Drink', 'Beach'."
        )
    )
    description: Optional[str] = Field(
        None,
        description="Short description of the attraction."
    )
    open_status: Optional[str] = Field(
        None,
        description="'Open' or 'Closed' if present in SerpAPI description, else null."
    )

    # ── Ratings ───────────────────────────────────────────────────────────
    rating: Optional[float] = Field(
        None,
        ge=0,
        le=5,
        description="Overall rating out of 5."
    )
    reviews_count: Optional[int] = Field(
        None,
        description="Total number of reviews."
    )

    # ── Pricing ───────────────────────────────────────────────────────────
    price: Optional[float] = Field(
        None,
        description="Entry price. Null means free or price unknown."
    )
    is_free: bool = Field(
        False,
        description="True if no price key was present in the SerpAPI result."
    )

    # ── Media ─────────────────────────────────────────────────────────────
    # thumbnail: Optional[str] = Field(
    #     None,
    #     description="Thumbnail image URL."
    # )

    # ── Location context (from hotel nearby_places) ───────────────────────
    distance_minutes: Optional[int] = Field(
        None,
        description="Travel time from hotel in minutes. Null if hotel context not provided."
    )
    distance_source: Optional[str] = Field(
        None,
        description="Transport type for distance e.g. 'Walking', 'Taxi'. Null if not matched."
    )


# ============================================================================
# ITINERARY MODELS
# ============================================================================

class ActivitySlot(BaseModel):
    """Minimal activity slot — LLM outputs only scheduling decisions."""
    type: Literal[
        "outbound_flight", "return_flight",
        "hotel_checkin", "hotel_checkout",
        "attraction", "free_time", "travel"
    ]
    attraction_index: Optional[int] = Field(
        None,
        description="0-based index into the attractions list. Only set when type='attraction'."
    )
    time: str = Field(description="Start time e.g. '09:00'.")
    duration_minutes: int
    notes: Optional[str] = Field(None, description="Practical tip or context for this slot.")


class DaySchedule(BaseModel):
    """One day's schedule — LLM outputs only structure and theme."""
    day_number: int
    date: Optional[str] = None
    theme: str
    activities: List[ActivitySlot]
    notes: Optional[str] = None


class ItinerarySchedule(BaseModel):
    """
    Minimal LLM output for the itinerary agent.
    Python hydrates this into a full ItineraryAgentResponse.
    """
    status: Literal["success", "no_results"]
    trip_title: str
    days: List[DaySchedule] = Field(default_factory=list)
    highlights: List[str] = Field(default_factory=list)
    practical_tips: List[str] = Field(default_factory=list)
    reasoning: str = ""
    error_message: Optional[str] = None

    @field_validator("days", "highlights", "practical_tips", mode="before")
    @classmethod
    def none_to_empty_list(cls, v):
        return v if v is not None else []
    

class ItineraryActivity(BaseModel):
    """
    A single activity or event within a day.
    Can be a flight, hotel check-in/out, or attraction.
    """
    time: Optional[str] = Field(
        None,
        description="Suggested time e.g. '09:00 AM'. Null if not time-specific."
    )
    type: Literal["flight", "hotel", "attraction", "travel", "free_time"] = Field(
        description="Category of activity."
    )
    title: str = Field(description="Short title e.g. 'Visit Pico do Arieiro'")
    description: Optional[str] = Field(
        None,
        description="Additional detail or tips for this activity."
    )
    duration_minutes: Optional[int] = Field(
        None,
        description="Estimated duration in minutes."
    )
    cost: Optional[float] = Field(
        None,
        description="Estimated cost. Null if free or unknown."
    )
    is_free: bool = False
    notes: Optional[str] = Field(
        None,
        description="Practical tips e.g. 'Book in advance', 'Bring cash'."
    )


class ItineraryDay(BaseModel):
    """A single day in the itinerary."""
    day: int = Field(description="Day number starting from 1.")
    date: Optional[str] = Field(
        None,
        description="ISO date string e.g. '2026-03-23'."
    )
    theme: Optional[str] = Field(
        None,
        description="Theme for the day e.g. 'Arrival & Explore Funchal'."
    )
    activities: List[ItineraryActivity] = Field(
        default_factory=list,
        description="Ordered list of activities for the day."
    )
    estimated_daily_cost: Optional[float] = Field(
        None,
        description="Sum of known activity costs for the day."
    )
    notes: Optional[str] = Field(
        None,
        description="Day-level notes e.g. 'Arrival day, take it easy'."
    )


# ============================================================================
# REALLOCATION MODELS
# ============================================================================

class ReallocationOption(BaseModel):
    """A single budget reallocation strategy."""
    strategy_id: str = Field(..., description="Unique identifier for this strategy")
    category: str = Field(..., description="Category affected (flights, hotels, activities, itinerary)")
    action: str = Field(..., description="Description of the reallocation action")
    potential_savings: float = Field(..., ge=0, description="Amount that can be saved")
    impact_score: int = Field(..., ge=1, le=10, description="Impact on trip quality (1=low, 10=high)")
    affected_agent: Optional[str] = Field(None, description="Which agent needs to be re-called")
    feasibility: Literal["high", "medium", "low"] = "medium"
    details: Optional[str] = Field(None, description="Additional details about this option")
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "strategy_id": "flight_connecting",
                "category": "flights",
                "action": "Switch to connecting flight",
                "potential_savings": 350,
                "impact_score": 2,
                "affected_agent": "flight",
                "feasibility": "high",
            }
        }


class BudgetReport(BaseModel):
    """Complete budget status report."""
    total_budget: float
    total_spent: float
    difference: float
    is_over_budget: bool
    category_spending: Dict[str, float] = Field(default_factory=dict)
    reallocation_options: List[ReallocationOption] = Field(default_factory=list)
    recommended_strategy: Optional[str] = None
    summary: str
    
    class Config:
        """Pydantic config."""
        json_schema_extra = {
            "example": {
                "total_budget": 3000,
                "total_spent": 3400,
                "difference": -400,
                "is_over_budget": True,
                "summary": "Over budget by $400. Recommend flight downgrade.",
            }
        }
        

class BudgetOverageAnalysis(BaseModel):
    """Detailed analysis of budget overage."""
    is_over_budget: bool
    overage_amount: float
    total_spent: float
    total_budget: float
    category_breakdown: Dict[str, float]
    deficit_percentage: float = Field(..., description="Percentage over budget")
    
    @property
    def remaining_budget(self) -> float:
        """Calculate remaining budget (negative if over)."""
        return self.total_budget - self.total_spent
    

# ============================================================================
# AGENT RESPONSE MODELS
# ============================================================================
class BudgetDecision(BaseModel):
    """LLM outputs only the recommendation decision."""
    primary_recommendation: str = Field(
        description="Primary cost-cutting recommendation in plain English."
    )
    affected_agent: Optional[str] = Field(
        None,
        description="Which agent to re-call: 'flight', 'hotel', 'activities', or None."
    )
    constraint_to_apply: Optional[str] = Field(
        None,
        description=(
            "Actionable constraint to pass to the affected agent. "
            "E.g. 'Search for connecting flights only' or 'Max hotel budget €80/night'."
        )
    )
    alternative_recommendations: List[str] = Field(
        default_factory=list,
        description="Up to 2 fallback strategies if primary doesn't work."
    )
    explanation: str = Field(
        description="2-3 sentences covering what was found and what is recommended."
    )


class BudgetAgentResponse(BaseModel):
    """
    Structured response from Budget Manager Agent.
    Optimized for structured output with response_format.
    
    This schema is flatter than the old version to work better
    with LLM structured output generation.
    """
    # Status indicator
    status: str = Field(
        description="Budget status: 'allocated', 'within_budget', 'over_budget'"
    )
    
    # Budget allocation (when doing initial allocation)
    allocation: Optional[BudgetAllocation] = Field(
        None,
        description="Budget allocation breakdown when allocating budget"
    )
    
    # Budget info (when checking budget)
    total_budget: Optional[float] = Field(None, description="Total budget amount")
    total_spent: Optional[float] = Field(None, description="Total amount spent")
    difference: Optional[float] = Field(None, description="Budget remaining (negative if over)")
    is_over_budget: Optional[bool] = Field(None, description="Whether over budget")
    
    # Reallocation recommendations (when over budget)
    primary_recommendation: Optional[str] = Field(
        None,
        description="Primary cost-cutting recommendation"
    )
    savings_amount: Optional[float] = Field(
        None,
        description="Amount saved by primary recommendation"
    )
    affected_agent: Optional[str] = Field(
        None,
        description="Which agent to re-call (flight, hotel, activities, itinerary_optimizer)"
    )
    constraint_to_apply: Optional[str] = Field(
        None,
        description="Constraint to pass to the affected agent"
    )
    
    # Alternative recommendations
    alternative_recommendations: List[str] = Field(
        default_factory=list,
        description="Alternative cost-cutting strategies if primary doesn't work"
    )
    
    # Explanation
    explanation: str = Field(
        description="Human-readable explanation of the budget analysis and recommendations"
    )

 
class FlightRankingResponse(BaseModel):
    """
    Minimal structured output from the LLM ranking step in FlightAgent.
    LLM outputs ONLY ranking decisions — all field mapping is done in Python.

    Outbound and return flights are ranked in a single LLM call to allow
    the LLM to reason about combined price against the budget ceiling.
    """
    status: Literal["success", "over_budget", "no_results"] = Field(
        description=(
            "'success' if at least one outbound+return combo is within budget (or no budget set), "
            "'over_budget' if every combo exceeds the budget, "
            "'no_results' if the search returned nothing."
        )
    )

    # ── Outbound ─────────────────────────────────────────────────────────────
    recommended_index: Optional[int] = Field(
        None,
        description=(
            "0-based index into the outbound list of the best outbound flight. "
            "None only when status='no_results'."
        ),
    )
    alternative_indices: List[int] = Field(
        default_factory=list,
        description="Up to 4 runner-up outbound indices, ordered best-first.",
    )

    # ── Return (None / [] for one-way trips) ─────────────────────────────────
    recommended_return_index: Optional[int] = Field(
        None,
        description=(
            "0-based index into the return list of the best return flight. "
            "None for one-way trips or when status='no_results'."
        ),
    )
    alternative_return_indices: List[int] = Field(
        default_factory=list,
        description="Up to 4 runner-up return indices, ordered best-first.",
    )

    # ── Reasoning ────────────────────────────────────────────────────────────
    reasoning: str = Field(
        default="",
        description=(
            "2-4 sentences covering BOTH legs — why this outbound was chosen, "
            "why this return was chosen, and why alternatives ranked lower. "
            "Reference specific airlines, prices, durations, and stop counts."
        ),
    )
    error_message: Optional[str] = Field(
        None,
        description="Set only when status='no_results'.",
    )
    

class FlightAgentResponse(BaseModel):
    """
    Structured response from Flight Agent.
    Returned by FlightAgent.search_and_recommend() after searching
    and scoring available flights.

    For round-trip searches the agent performs a two-step flow:
      1. Search outbound flights → pick best → note its departure_token
      2. Call search_return_flights with that token → pick best return

    Covers three outcomes:
      - 'success'      : flights found and a recommendation was made
      - 'over_budget'  : flights found but all exceed the given budget;
                         cheapest option is surfaced as the recommendation
      - 'no_results'   : no flights could be retrieved (API error, missing
                         key, or zero results from the search provider)
    """

    status: str = Field(
        description=(
            "Outcome of the search. One of: "
            "'success' (recommendation made within budget), "
            "'over_budget' (all flights exceed budget, cheapest shown), "
            "'no_results' (search returned nothing or failed)."
        )
    )

    # ── Outbound ─────────────────────────────────────────────────────
    recommended_flight: Optional[FlightOption] = Field(
        None,
        description=(
            "The single best OUTBOUND flight. "
            "None only when status is 'no_results'."
        )
    )
 
    alternative_options: List[FlightOption] = Field(
        default_factory=list,
        description="Up to 4 runner-up outbound flights."
    )

    all_flights: List[FlightOption] = Field(
        default_factory=list,
        description="ALL outbound flights from the search tool."
    )

    # ── Return ───────────────────────────────────────────────────────
    recommended_return_flight: Optional[FlightOption] = Field(
        None,
        description=(
            "REQUIRED for round-trips. The single best RETURN flight, "
            "searched using the departure_token of the recommended "
            "outbound flight. None ONLY for one-way trips."
        )
    )

    alternative_return_options: List[FlightOption] = Field(
        default_factory=list,
        description=(
            "REQUIRED for round-trips. Up to 4 runner-up return flights. "
            "If only 1 return flight exists, this can be empty, but "
            "recommended_return_flight MUST still be set."
        )
    )

    all_return_flights: List[FlightOption] = Field(
        default_factory=list,
        description="ALL return flights from the return search."
    )
    

    # ── Reasoning ────────────────────────────────────────────────────
    reasoning: str = Field(
        default="",
        description=(
            "2-4 sentence explanation covering both outbound and return "
            "flight choices, including why alternatives were ranked lower."
        )
    )
    error_message: Optional[str] = Field(
        None,
        description="Set only when status='no_results'.",
    )
    
    @field_validator(
        "alternative_options",
        "all_flights",
        "alternative_return_options",
        "all_return_flights",
        mode="before",
    )
    @classmethod
    def none_to_empty_list(cls, v):
        """LLM may return null for list fields on one-way trips."""
        return v if v is not None else []


class HotelRankingResponse(BaseModel):
    """
    Minimal structured output from the LLM ranking step.
    LLM outputs ONLY ranking decisions — all field mapping is done in Python.
    """
    status: Literal["success", "over_budget", "no_results"] = Field(
        description=(
            "'success' if at least one hotel is within budget, "
            "'over_budget' if all hotels exceed budget, "
            "'no_results' if the search failed."
        )
    )
    recommended_index: Optional[int] = Field(
        None,
        description="0-based index of the single best hotel. None only when status='no_results'.",
    )
    alternative_indices: List[int] = Field(
        default_factory=list,
        description="Up to 4 runner-up indices, ordered best-first.",
    )
    reasoning: str = Field(
        default="",
        description=(
            "2-4 sentences explaining the recommendation. Must reference specific "
            "hotel names and concrete data points (rating, price, amenity flags)."
        ),
    )
    error_message: Optional[str] = Field(
        None,
        description="Set only when status='no_results'.",
    )


class HotelsAgentResponse(BaseModel):
    """
    Structured response from Hotels Agent.
    Returned by HotelsAgent.search_and_recommend() after searching
    and scoring available hotels.

    Covers three outcomes:
      - 'success'      : hotels found and recommendation made within budget
      - 'over_budget'  : hotels found but all exceed budget;
                         cheapest option is surfaced as recommendation
      - 'no_results'   : no hotels could be retrieved (API error, missing
                         key, or zero results from search provider)
    """

    status: str = Field(
        description=(
            "Outcome of the search. One of: "
            "'success' (recommendation made within budget), "
            "'over_budget' (all hotels exceed budget, cheapest shown), "
            "'no_results' (search returned nothing or failed)."
        )
    )

    # ── Primary Recommendation ────────────────────────────────────────
    recommended_hotel: Optional[HotelOption] = Field(
        None,
        description=(
            "The single best hotel recommendation. "
            "None only when status is 'no_results'."
        )
    )

    # ── Alternatives ──────────────────────────────────────────────────
    alternative_options: List[HotelOption] = Field(
        default_factory=list,
        description="Up to 4 runner-up hotels with different profiles."
    )

    all_hotels: List[HotelOption] = Field(
        default_factory=list,
        description="ALL hotels from the search results."
    )

    # ── Context ───────────────────────────────────────────────────────
    search_criteria: dict = Field(
        default_factory=dict,
        description=(
            "Summary of search parameters. "
            "E.g., {'location': 'Bali', 'check_in': '2026-03-23', "
            "'check_out': '2026-03-31', 'nights': 8, 'guests': 2}"
        )
    )

    detected_user_preferences: Optional[str] = Field(
        None,
        description=(
            "Inferred preferences from conversation. "
            "E.g., 'Budget max: 500, Amenities: [pool, spa], Min rating: 4.0'"
        )
    )

    # ── Market Insights ───────────────────────────────────────────────
    price_range: Optional[dict] = Field(
        None,
        description=(
            "Price distribution of search results. "
            "E.g., {'min': 25, 'max': 300, 'average': 95}"
        )
    )

    rating_range: Optional[dict] = Field(
        None,
        description=(
            "Rating distribution of search results. "
            "E.g., {'min': 3.2, 'max': 4.8, 'average': 4.3}"
        )
    )

    # ── Reasoning ─────────────────────────────────────────────────────
    reasoning: str = Field(
        default="",
        description=(
            "2-4 sentence explanation of the recommendation decision, "
            "including key factors (price, rating, amenities, location) "
            "and why alternatives were ranked lower."
        )
    )

    # ── Error Handling ────────────────────────────────────────────────
    error_message: Optional[str] = Field(
        None,
        description="Error details if status is 'no_results'."
    )

    # ── Metadata ──────────────────────────────────────────────────────
    total_results: int = Field(
        default=0,
        description="Total number of hotels found in search."
    )

    @field_validator(
        "alternative_options",
        "all_hotels",
        mode="before",
    )
    @classmethod
    def none_to_empty_list(cls, v):
        """Handle null list fields gracefully."""
        return v if v is not None else []
    

class AttractionRankingResponse(BaseModel):
    """
    Minimal structured output from the LLM ranking step.
    The LLM outputs ONLY ranking decisions — all field mapping
    and enrichment happens in Python.
    """
    status: Literal["success", "no_results"] = Field(
        description="'success' or 'no_results'."
    )
    recommended_indices: List[int] = Field(
        default_factory=list,
        description=(
            "Ordered indices into the raw sights list, best-first. "
            "~3-4 per day."
        ),
    )
    alternative_indices: List[int] = Field(
        default_factory=list,
        description="All remaining indices not in recommended_indices.",
    )
    reasoning: str = Field(
        default="",
        description="2-4 sentences on how days + preferences shaped the picks.",
    )
    error_message: Optional[str] = None
    

class AttractionAgentResponse(BaseModel):
    """
    Structured response from the Attractions Agent.
    Returned by AttractionsAgent.find_attractions() after searching
    and ranking available attractions.

    Covers two outcomes:
      - 'success'    : attractions found and recommendations made
      - 'no_results' : search returned nothing or API error
    """

    status: Literal["success", "no_results"] = Field(
        description="Outcome of the search. 'success' or 'no_results'."
    )

    # ── Recommendations ───────────────────────────────────────────────────
    recommended_attractions: List[Attraction] = Field(
        default_factory=list,
        description=(
            "Top attractions ranked by rating and user preference match. "
            "Capped at roughly 3-4 per day based on days provided — "
            "e.g. 2 days → up to 8 recommendations."
        )
    )

    alternative_attractions: List[Attraction] = Field(
        default_factory=list,
        description=(
            "Other worthwhile attractions that didn't make the top picks. "
            "Good fallbacks if recommended ones are closed or not of interest."
        )
    )

    all_attractions: List[Attraction] = Field(
        default_factory=list,
        description="Every attraction returned from the search, unfiltered."
    )

    # ── Summary ───────────────────────────────────────────────────────────
    total_results: int = Field(
        default=0,
        description="Total number of attractions found."
    )

    days: int = Field(
        default=1,
        description="Number of days the user is visiting, as provided in the request."
    )

    estimated_total_cost: Optional[float] = Field(
        None,
        description=(
            "Sum of known entry prices across recommended_attractions only. "
            "Null if all attractions are free or prices are unknown."
        )
    )

    # ── Context ───────────────────────────────────────────────────────────
    detected_user_preferences: Optional[dict] = Field(
        None,
        description=(
            "Only populate from what the user explicitly stated. "
            "Never infer. E.g. {'interests': ['hiking', 'culture'], 'budget': 'low'}"
        )
    )

    hotel_context_used: bool = Field(
        default=False,
        description="True if hotel nearby_places were provided and used to enrich distance fields."
    )

    # ── Reasoning & Errors ────────────────────────────────────────────────
    reasoning: str = Field(
        default="",
        description=(
            "2-4 sentence explanation of how attractions were selected and ranked, "
            "referencing days, preferences, and ratings."
        )
    )

    error_message: Optional[str] = Field(
        None,
        description="Error details if status is 'no_results'."
    )

    @field_validator(
        "recommended_attractions",
        "alternative_attractions",
        "all_attractions",
        mode="before",
    )
    @classmethod
    def none_to_empty_list(cls, v):
        """Handle null list fields gracefully."""
        return v if v is not None else []
    

class ItineraryAgentResponse(BaseModel):
    """
    Structured response from the Itinerary Agent.
    Combines flight, hotel, and attractions data into a
    coherent day-by-day travel plan.

    Covers two outcomes:
      - 'success'    : itinerary successfully built
      - 'no_results' : insufficient input data to build itinerary
    """

    status: Literal["success", "no_results"] = Field(
        description="'success' if itinerary was built, 'no_results' if input was insufficient."
    )

    # ── Trip Overview ─────────────────────────────────────────────────────
    destination: str = Field(description="Primary destination e.g. 'Madeira, Portugal'.")
    trip_title: str = Field(
        description="Catchy trip title e.g. '8-Day Madeira Adventure'."
    )
    total_days: int = Field(default=0)
    check_in_date: Optional[str] = None
    check_out_date: Optional[str] = None

    # ── Day-by-Day Plan ───────────────────────────────────────────────────
    days: List[ItineraryDay] = Field(
        default_factory=list,
        description="Ordered list of days. Day 1 = arrival, last day = departure."
    )

    # ── Cost Summary ──────────────────────────────────────────────────────
    total_estimated_cost: Optional[float] = Field(
        None,
        description=(
            "Total estimated trip cost across flights, hotel, and attractions. "
            "Sum of: flight total + hotel total_price + attraction entry fees."
        )
    )
    cost_breakdown: Optional[dict] = Field(
        None,
        description=(
            "Cost split by category. "
            "E.g. {'flights': 388, 'hotel': 492, 'attractions': 55, 'total': 935}"
        )
    )

    # ── Trip Highlights ───────────────────────────────────────────────────
    highlights: List[str] = Field(
        default_factory=list,
        description="3-5 headline highlights of the trip e.g. 'Sunrise at Pico do Arieiro'."
    )

    # ── Practical Info ────────────────────────────────────────────────────
    practical_tips: List[str] = Field(
        default_factory=list,
        description=(
            "General trip tips e.g. 'Rent a car for flexibility', "
            "'Book cable car tickets in advance'."
        )
    )

    # ── Reasoning & Errors ────────────────────────────────────────────────
    reasoning: str = Field(
        default="",
        description=(
            "2-4 sentence explanation of how the itinerary was structured, "
            "referencing flight times, hotel location, and user preferences."
        )
    )
    error_message: Optional[str] = None

    @field_validator("days", "highlights", "practical_tips", mode="before")
    @classmethod
    def none_to_empty_list(cls, v):
        return v if v is not None else []


# ============================================================================
# MAIN STATE MODEL
# ============================================================================

class TravelPlanState(BaseModel):
    """
    Complete state of the travel planning system.
    Shared across all agents. Populated progressively as each agent runs.
    """

    # ── User Inputs ───────────────────────────────────────────────────────
    user_request: str
    destination: str
    origin: str = ""
    origin_code: str = ""         # IATA code e.g. "FRA"
    destination_code: str = ""    # IATA code e.g. "LIS"
    check_in_date: str
    check_out_date: str
    adults: int = 1
    currency: str = "EUR"
    total_budget: float
    duration_days: int
    user_preferences: str = ""

    # ── Budget ────────────────────────────────────────────────────────────
    budget_allocation: BudgetAllocation = Field(default_factory=BudgetAllocation)

    # ── Agent Responses (populated as each agent runs) ────────────────────
    flight_response: Optional[FlightAgentResponse] = None
    hotel_response: Optional[HotelsAgentResponse] = None
    attractions_response: Optional[AttractionAgentResponse] = None
    itinerary_response: Optional[ItineraryAgentResponse] = None
    budget_response: Optional[BudgetAgentResponse] = None

    # ── Status ────────────────────────────────────────────────────────────
    status: Literal[
        "planning",
        "searching",
        "over_budget",
        "within_budget",
        "complete"
    ] = "planning"
    iteration_count: int = 0
    decisions_made: List[str] = Field(default_factory=list)

    # ── Computed Cost Properties ──────────────────────────────────────────
    @property
    def flight_cost(self) -> float:
        """
        Total flight cost for the trip.

        Round-trip: use recommended_return_flight.price — this is the
        confirmed round-trip total from SerpAPI after selecting both legs.
        It's more accurate than outbound.price which is just the initial estimate.

        One-way: use recommended_flight.price — outbound only.
        """
        if not self.flight_response:
            return 0.0
        fr = self.flight_response

        # Round-trip — return price is the confirmed total for this pair
        if fr.recommended_return_flight and fr.recommended_return_flight.price:
            return fr.recommended_return_flight.price

        # One-way — outbound price only
        if fr.recommended_flight:
            return fr.recommended_flight.price

        return 0.0

    @property
    def hotel_cost(self) -> float:
        """Total hotel cost for the full stay."""
        if self.hotel_response and self.hotel_response.recommended_hotel:
            return self.hotel_response.recommended_hotel.total_price
        return 0.0

    @property
    def attractions_cost(self) -> float:
        """Sum of known attraction entry prices."""
        if not self.attractions_response:
            return 0.0
        return sum(
            a.price or 0.0
            for a in self.attractions_response.recommended_attractions
        )

    @property
    def total_spent(self) -> float:
        """Total cost across flights, hotel, and attractions."""
        return self.flight_cost + self.hotel_cost + self.attractions_cost

    @property
    def remaining_budget(self) -> float:
        """How much budget is left (negative means over budget)."""
        return self.total_budget - self.total_spent

    @property
    def is_over_budget(self) -> bool:
        return self.total_spent > self.total_budget

    def as_expenses(self) -> List[Expense]:
        """
        Convert current agent selections into an Expense list
        for use with budget tools.
        """
        expenses = []

        if self.flight_cost > 0:
            fr = self.flight_response
            outbound_airline = (
                fr.recommended_flight.airline
                if fr and fr.recommended_flight else "unknown"
            )
            return_airline = (
                fr.recommended_return_flight.airline
                if fr and fr.recommended_return_flight else None
            )

            if return_airline:
                # Round-trip — confirmed total via return_flight.price
                if return_airline == outbound_airline:
                    description = f"Flights round-trip ({outbound_airline})"
                else:
                    description = f"Flights round-trip ({outbound_airline} / {return_airline})"
            else:
                description = f"Flights one-way ({outbound_airline})"

            expenses.append(Expense(
                category="flights",
                amount=self.flight_cost,
                description=description,
            ))

        if self.hotel_cost > 0:
            hotel_name = (
                self.hotel_response.recommended_hotel.name
                if self.hotel_response and self.hotel_response.recommended_hotel
                else "unknown"
            )
            expenses.append(Expense(
                category="hotels",
                amount=self.hotel_cost,
                description=f"Hotel: {hotel_name}"
            ))

        if self.attractions_cost > 0:
            count = (
                len(self.attractions_response.recommended_attractions)
                if self.attractions_response else 0
            )
            expenses.append(Expense(
                category="activities",
                amount=self.attractions_cost,
                description=f"Attractions ({count} items)"
            ))

        return expenses

    def update_status(self):
        """Sync status field with current spending."""
        if self.total_spent > self.total_budget:
            self.status = "over_budget"
        elif self.total_spent > 0:
            self.status = "within_budget"