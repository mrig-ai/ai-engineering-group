"""
orchestrator.py
---------------
GRAPH FLOW:
  extract_params
      ↓
  [missing_fields?] → ask_user → END (waits for next message)
  [replan?]         → skip to check_budget scope
  [else]            → allocate_budget
      ↓
  allocate_budget
      ↓ (fan-out — parallel)
  search_flights │ search_hotels │ search_attractions
      ↓ (fan-in — all must complete)
  merge_results
      ↓
  check_budget
      ↓
  [within_budget] → build_itinerary → END
  [over_budget]   → hitl_interrupt  (graph pauses, user decides)
      ↓
  [approved] → rerun_agent → build_itinerary → END
  [rejected] → build_itinerary → END (with cost warning)

REPLAN FLOW (follow-up turns):
  extract_params detects replan intent + scope
  → skips allocate_budget
  → only re-runs agents in replan_scope
  → merge_results → check_budget → build_itinerary
"""

from __future__ import annotations

import operator
from concurrent.futures import ThreadPoolExecutor
from typing import Annotated, Any, Literal, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.types import interrupt
from pydantic import BaseModel, Field

from src.agents.attractions_agent import AttractionsAgent
from src.agents.budget_agent import BudgetManagerAgent
from src.agents.flight_agent import FlightAgent
from src.agents.hotel_agent import HotelAgent
from src.agents.itinerary_agent import ItineraryAgent
from src.logger import get_logger, log_user_message
from src.models import (
    BudgetAgentResponse,
    TravelPlanState,
)

logger = get_logger(__name__)


# ============================================================================
# ROUTER OUTPUT MODEL
# ============================================================================

class RouterOutput(BaseModel):
    """Structured output from the RouterAgent LLM call."""

    intent: Literal[
        "full_itinerary",       # plan a complete trip
        "flights_only",         # user wants only flight info
        "hotels_only",          # user wants only hotel info
        "attractions_only",     # user wants only attraction info
        "replan_duration",      # duration changed (e.g. 5 days instead of 8)
        "replan_budget",        # budget changed
        "replan_preferences",   # preferences changed (e.g. driving instead of flying)
        "unknown",              # cannot classify — ask for clarification
    ]

    # ── Extracted trip params ────────────────────────────────────────────
    destination: Optional[str] = None
    origin: Optional[str] = None
    origin_code: str = Field(
        default="",
        description="IATA airport code for origin e.g. 'FRA'. Always extract this."
    )
    destination_code: str = Field(
        default="",
        description="IATA airport code for destination e.g. 'LIS'. Always extract this."
    )
    budget: Optional[float] = None
    departure_date: Optional[str] = None   # YYYY-MM-DD
    return_date: Optional[str] = None      # YYYY-MM-DD
    duration_days: Optional[int] = None
    currency: str = "EUR"
    adults: int = 1
    user_preferences: str = ""

    # ── Skip flags (detected from preferences) ───────────────────────────
    skip_flights: bool = Field(
        False,
        description="True if user mentions driving, own car, train, bus etc."
    )
    skip_hotels: bool = Field(
        False,
        description="True if user mentions staying with friends, own accommodation etc."
    )

    # ── Orchestration fields ──────────────────────────────────────────────
    missing_fields: list[str] = Field(
        default_factory=list,
        description=(
            "Fields required for this intent that cannot be extracted. "
            "e.g. ['departure_date', 'budget']. Empty if all required fields are present."
        ),
    )
    replan_scope: list[str] = Field(
        default_factory=list,
        description=(
            "For replan intents: which agents must re-run. "
            "Possible values: 'flights', 'hotels', 'attractions', 'budget', 'itinerary'. "
            "Empty for non-replan intents."
        ),
    )


# ============================================================================
# GRAPH STATE
# ============================================================================

def _merge_travel_plan(existing: Optional[dict], update: Optional[dict]) -> Optional[dict]:
    """
    Reducer for travel_plan.
    Merges update dict into existing dict — parallel nodes each write their
    own slice (flight_response, hotel_response, etc.) without overwriting others.
    """
    if existing is None:
        return update
    if update is None:
        return existing
    merged = {**existing}
    for k, v in update.items():
        if v is not None:
            merged[k] = v
    return merged


class GraphState(dict):
    """
    LangGraph state. Uses Annotated reducers for fields written by parallel nodes.

    We store TravelPlanState as a plain dict (serialised via model_dump) so that
    LangGraph's checkpointer can pickle it cleanly. It is reconstructed into
    TravelPlanState objects inside each node as needed.
    """

    # ── Core travel plan — merged across parallel writes ────────────────
    travel_plan: Annotated[Optional[dict], _merge_travel_plan]

    # ── Orchestration metadata ────────────────────────────────────────────
    intent: str
    missing_fields: list[str]
    replan_scope: list[str]
    active_constraints: dict          # e.g. {"hotel": "max €80/night"}
    skip_flights: bool
    skip_hotels: bool

    # ── Budget check result (passed from check_budget → hitl → rerun) ───
    budget_result: Optional[dict]     # BudgetAgentResponse.model_dump()

    # ── HITL decision ─────────────────────────────────────────────────────
    hitl_decision: Optional[str]      # "approve" | "reject"

    # ── Conversation messages (append-only) ──────────────────────────────
    messages: Annotated[list, operator.add]

    # ── Progress events streamed to the UI ───────────────────────────────
    progress: Annotated[list[str], operator.add]

    # ── Cost warning flag (set when user rejects reallocation) ───────────
    budget_warning: Optional[str]

    # ── Clarification tracking ────────────────────────────────────────────
    # When ask_user fires, store the original incomplete request so the
    # router can combine it with the user's follow-up answer.
    pending_clarification: Optional[str]   # original user request being clarified
    pending_intent: Optional[str]          # original intent before clarification

    # ── Last single-agent intent ──────────────────────────────────────────
    # Tracks the most recent flights_only/hotels_only/attractions_only intent
    # so follow-up replans ("change date to 26th") re-run the same agent.
    last_single_agent_intent: Optional[str]


# ============================================================================
# ROUTER AGENT
# ============================================================================

_ROUTER_SYSTEM_PROMPT = """You are a Travel Planning Router. Classify the user message and extract trip parameters.

## INTENT (pick exactly one)

| Intent | When to use |
|---|---|
| full_itinerary | User wants a complete trip plan — flights + hotel + activities + itinerary |
| flights_only | User wants flight search only |
| hotels_only | User wants hotel search only |
| attractions_only | User wants places to visit / things to do only |
| replan_duration | Existing COMPLETE itinerary exists AND user changes trip length |
| replan_budget | Existing COMPLETE itinerary exists AND user changes budget |
| replan_preferences | Existing COMPLETE itinerary exists AND user changes a preference |
| unknown | Cannot classify |

**Priority rules:**
- `flights_only` / `hotels_only` / `attractions_only` always win over replan intents.
  A new search request is never a replan — even if an itinerary exists.
- Replan intents require an existing COMPLETE itinerary AND explicit modification language
  ("change", "update", "instead", "make it", "extend", "reduce", "prefer").
- A follow-up question without modification language → `unknown` (not replan).

---

## EXTRACTION RULES

- **Dates**: YYYY-MM-DD. "8 days from April 1" → departure=2026-04-01, return=2026-04-09.
  "8 days" without dates → duration_days=8, leave dates empty.
- **Currency**: default EUR.
- **Adults**: default 1. Extract from "2 people", "couple", "family of 4", "two of us", etc.
- **origin / destination**: human-readable city name (e.g. "Frankfurt", "Lisbon").
- **origin_code / destination_code**: 3-letter IATA code. Derive from city name.
  Common codes: FRA=Frankfurt, LIS=Lisbon, LHR=London, CDG=Paris, BER=Berlin,
  MAD=Madrid, AMS=Amsterdam, FCO=Rome, BCN=Barcelona, MUC=Munich, VIE=Vienna,
  ZRH=Zurich, DUB=Dublin, BRU=Brussels, PRG=Prague, CPH=Copenhagen, ARN=Stockholm,
  OSL=Oslo, HEL=Helsinki, ATH=Athens, IST=Istanbul, DXB=Dubai, SIN=Singapore,
  NRT=Tokyo, BKK=Bangkok, JFK=New York, LAX=Los Angeles, SYD=Sydney, YYZ=Toronto,
  FNC=Madeira, DPS=Bali, TFS=Tenerife.
  Leave empty if unknown — the system will resolve it.
- **skip_flights**: true if user mentions driving / own car / train / bus.
- **skip_hotels**: true if user mentions staying with friends / own accommodation.

---

## MISSING FIELDS

Only flag fields as missing when genuinely required AND not inferable.
Do NOT flag origin_code / destination_code — the system resolves these automatically.
Use only field names that exist in the output schema: departure_date, return_date, duration_days.
NEVER flag check_in_date or check_out_date — use departure_date and return_date instead.

| Intent | Required fields |
|---|---|
| full_itinerary | destination, origin, departure_date OR duration_days |
| flights_only | destination, origin, departure_date |
| hotels_only | destination, departure_date, return_date OR duration_days |
| attractions_only | destination |
| replan_* | only NEW fields that changed |

Budget is OPTIONAL for all intents — never flag it as missing. The system handles no-budget trips gracefully.

---

## REPLAN SCOPE

**Critical distinction — budget vs preference:**
- `replan_budget` = user changes the **total trip budget** (e.g. "increase budget to €3000", "I have €500 more")
  → scope: ["budget", "itinerary"] — re-allocate all categories from new total
- `replan_preferences` = user changes a **specific category preference or per-category limit**
  (e.g. "get a cheaper hotel", "I want a hotel with pool", "increase hotel budget to €100/night",
  "find a better flight", "no more hikes", "add more culture days")
  → scope depends on what changed:
    - **Adults / number of people** → ["flights", "hotels", "itinerary"] — prices depend on adults, ALWAYS re-search both
    - Hotel preference/budget → ["hotels", "itinerary"]
    - Flight preference → ["flights", "itinerary"]
    - Activities preference → ["attractions", "itinerary"]
    - skip_flights → ["hotels", "itinerary"]
- `replan_duration` = user changes the **trip length**
  → scope: ["flights", "hotels", "attractions", "itinerary"]

**Examples:**
- "The overall budget is €2000" → replan_budget, scope=["budget", "itinerary"]
- "Change it to 2 people" → replan_preferences, adults=2, scope=["flights", "hotels", "itinerary"]
- "Make it for a family of 4" → replan_preferences, adults=4, scope=["flights", "hotels", "itinerary"]
- "Increase hotel budget to €100/night" → replan_preferences, scope=["hotels", "itinerary"]
- "Get a cheaper hotel" → replan_preferences, scope=["hotels", "itinerary"]
- "I want a pool and no hikes" → replan_preferences, scope=["hotels", "attractions", "itinerary"]
- "I'll drive instead of fly" → replan_preferences + skip_flights, scope=["hotels", "itinerary"]
- "Make it 5 days instead" → replan_duration, scope=["flights", "hotels", "attractions", "itinerary"]

---

## EXISTING STATE

Use existing trip data to fill missing fields.
Only overwrite fields the user explicitly changed.
Never invent values not stated by the user.
"""



class RouterAgent:
    """Single LLM call — classifies intent, extracts params, detects missing fields."""

    def __init__(self, model_name: str = "gpt-4o-mini", temperature: float = 0.0):
        llm = ChatOpenAI(model=model_name, temperature=temperature)
        self._structured_llm = llm.with_structured_output(RouterOutput)

    def route(
        self,
        user_message: str,
        existing_plan: Optional[TravelPlanState] = None,
    ) -> RouterOutput:
        existing_ctx = ""
        if existing_plan:
            existing_ctx = (
                f"\n\nEXISTING TRIP STATE:\n"
                f"  Destination: {existing_plan.destination}\n"
                f"  Origin: {existing_plan.origin}\n"
                f"  Dates: {existing_plan.check_in_date} → {existing_plan.check_out_date}\n"
                f"  Budget: {existing_plan.total_budget} {existing_plan.currency}\n"
                f"  Duration: {existing_plan.duration_days} days\n"
                f"  Adults: {existing_plan.adults}\n"
                f"  Preferences: {existing_plan.user_preferences or 'none'}\n"
                f"  Has flights: {existing_plan.flight_response is not None}\n"
                f"  Has hotels: {existing_plan.hotel_response is not None}\n"
                f"  Has attractions: {existing_plan.attractions_response is not None}\n"
                f"  Has itinerary: {existing_plan.itinerary_response is not None}\n"
            )

        messages = [
            {"role": "system", "content": _ROUTER_SYSTEM_PROMPT},
            {"role": "user",   "content": f"User message: {user_message}{existing_ctx}"},
        ]
        return self._structured_llm.invoke(messages)


# ============================================================================
# HELPER — reconstruct TravelPlanState from graph state dict
# ============================================================================

def _get_plan(state: GraphState) -> Optional[TravelPlanState]:
    """Reconstruct TravelPlanState from the graph state dict."""
    if not state.get("travel_plan"):
        return None
    try:
        return TravelPlanState(**state["travel_plan"])
    except Exception as e:
        logger.warning("_get_plan failed to reconstruct TravelPlanState: %s", e)
        return None


def _plan_to_dict(plan: TravelPlanState) -> dict:
    """Serialise TravelPlanState for storage in graph state."""
    return plan.model_dump()


# ============================================================================
# NODES
# ============================================================================

_router = RouterAgent()

# ── Node 1: Extract params & classify intent ──────────────────────────────

def extract_params_node(state: GraphState) -> dict:
    logger.info("[NODE] extract_params")
    last_message  = state["messages"][-1].content if state["messages"] else ""
    existing_plan = _get_plan(state)
    pending = state.get("pending_clarification")
    pending_intent = state.get("pending_intent")
    if pending and pending != last_message:
        combined_message = (
            f"{pending}\n"
            f"Additional info provided by user: {last_message}"
        )
        logger.debug("Clarification: combining '%s' + '%s'", pending, last_message)
    else:
        combined_message = last_message

    result: RouterOutput = _router.route(
        user_message=combined_message,
        existing_plan=existing_plan,
    )

    # ── Python safety net: extract adults if LLM missed it ───────────────
    if result.adults <= 1:
        import re
        msg_lower = combined_message.lower()
        # "for N people/passengers/persons/adults/travelers"
        m = re.search(r'for\s+(\d+)\s+(?:people|persons?|passengers?|adults?|travelers?|travell?ers?)', msg_lower)
        if not m:
            # "N people/passengers" anywhere in message
            m = re.search(r'(\d+)\s+(?:people|persons?|passengers?|adults?|travelers?|travell?ers?)', msg_lower)
        if not m:
            # "couple" / "two of us" / "me and my partner/friend/wife/husband"
            if any(p in msg_lower for p in ["couple", "two of us", "me and my", "my partner", "my wife", "my husband", "my friend"]):
                result.adults = 2
        if m:
            result.adults = int(m.group(1))
        if result.adults != 1:
            print(f"  [adults override] extracted adults={result.adults} from message")

    # ── If still missing fields but we had a pending intent, preserve it ──
    # Prevents the router from switching intent when user just answers a question
    if pending_intent and result.missing_fields and result.intent in ("unknown", "replan_duration", "replan_preferences", "full_itinerary"):
        result.intent = pending_intent

    # ── Build / update TravelPlanState from router output ────────────────
    if existing_plan:
        # Replan: merge only changed fields
        plan_dict = _plan_to_dict(existing_plan)
        if result.destination:      plan_dict["destination"]      = result.destination
        if result.origin:           plan_dict["origin"]           = result.origin
        if result.origin_code:      plan_dict["origin_code"]      = result.origin_code
        if result.destination_code: plan_dict["destination_code"] = result.destination_code
        # Only overwrite total_budget for replan_budget — for replan_preferences the
        # extracted number is a per-category cap (e.g. "€200/night"), not a new total budget.
        if result.budget and result.intent == "replan_budget":
            plan_dict["total_budget"] = result.budget
        if result.departure_date:   plan_dict["check_in_date"]    = result.departure_date
        if result.return_date:      plan_dict["check_out_date"]   = result.return_date
        if result.duration_days:    plan_dict["duration_days"]    = result.duration_days
        if result.user_preferences: plan_dict["user_preferences"] = result.user_preferences
        plan_dict["adults"] = result.adults
        if result.currency != "EUR": plan_dict["currency"]        = result.currency
        plan_dict["user_request"] = last_message

        if result.intent in ("flights_only", "hotels_only", "attractions_only"):
            plan_dict["check_in_date"]        = result.departure_date or ""
            plan_dict["check_out_date"]        = result.return_date   or ""
            plan_dict["duration_days"]         = result.duration_days or 0
            plan_dict["destination"]           = result.destination   or plan_dict["destination"]
            plan_dict["origin"]                = result.origin        or ""
            plan_dict["origin_code"]           = result.origin_code   or ""
            plan_dict["destination_code"]      = result.destination_code or ""
            plan_dict["adults"]                = result.adults  # explicit in reset block too
            plan_dict["flight_response"]       = None
            plan_dict["hotel_response"]        = None
            plan_dict["attractions_response"]  = None
            plan_dict["itinerary_response"]    = None
            plan_dict["budget_response"]       = None
    else:
        # Fresh plan
        plan_dict = {
            "user_request":      last_message,
            "destination":       result.destination or "",
            "origin":            result.origin or "",
            "origin_code":       result.origin_code or "",
            "destination_code":  result.destination_code or "",
            "total_budget":      result.budget or 0.0,
            "check_in_date":     result.departure_date or "",
            "check_out_date":    result.return_date or "",
            "duration_days":     result.duration_days or 0,
            "currency":          result.currency,
            "adults":            result.adults,
            "user_preferences":  result.user_preferences,
        }

    # ── Replan downgrade — must happen here, not in routing function ─────
    # routing functions receive a state snapshot — mutations don't persist.
    # If there's no complete itinerary, replan intents must be downgraded
    # to single-agent or full_itinerary before the node returns.
    final_intent = result.intent
    if final_intent.startswith("replan_"):
        existing_plan_check = _get_plan({"travel_plan": plan_dict})
        has_itinerary = existing_plan_check and existing_plan_check.itinerary_response is not None
        if not has_itinerary:
            last_single = state.get("last_single_agent_intent")
            if last_single in ("flights_only", "hotels_only", "attractions_only"):
                logger.debug("extract_params: replan→%s (last_single=%s)", last_single, last_single)
                final_intent = last_single
            else:
                logger.debug("extract_params: replan→full_itinerary (no itinerary)")
                final_intent = "full_itinerary"

    return {
        "travel_plan":                plan_dict,
        "intent":                     final_intent,
        "missing_fields":             result.missing_fields,
        "replan_scope":               result.replan_scope,
        "skip_flights":               result.skip_flights,
        "skip_hotels":                result.skip_hotels,
        "progress":                   ["🧠 Understanding your request..."],
        "pending_clarification":      None if not result.missing_fields else state.get("pending_clarification"),
        "pending_intent":             None if not result.missing_fields else state.get("pending_intent"),
        "last_single_agent_intent":   final_intent if final_intent in ("flights_only", "hotels_only", "attractions_only") else state.get("last_single_agent_intent"),
    }


# ── Node 2: Ask user for missing info ─────────────────────────────────────

def ask_user_node(state: GraphState) -> dict:
    logger.info("[NODE] ask_user")
    """Emits a clarifying question. Graph ends here — next user message re-enters."""
    fields  = state.get("missing_fields") or []
    intent  = state.get("intent", "unknown")

    field_labels = {
        "destination":                  "where you'd like to go",
        "origin":                       "where you're departing from",
        "origin city":                  "which city you're departing from",
        "origin_code":                  "your departure city or airport",
        "destination_code":             "your destination city or airport",
        "budget":                       "your total budget",
        "departure_date":               "your departure date",
        "return_date":                  "your return date",
        "duration_days":                "how many days you'd like to travel",
        "return date or trip duration": "your return date or how many days you're travelling",
    }

    # ── Unknown intent with no missing fields = greeting / off-topic ──────
    if intent == "unknown" or not fields:
        question = (
            "Hi there! I'm your AI travel planner. "
            "I can help you plan a full trip, find flights, hotels, or attractions. "
            "Try something like: *\"Plan a trip to Tokyo from Berlin, €2000, 5 days starting April 1st\"* "
            "or *\"Find flights from London to Barcelona next week\"*."
        )
        return {
            "messages": [AIMessage(content=question)],
            "progress": ["💬 Greeting sent"],
        }

    # ── Missing fields — ask for what's needed ────────────────────────────
    readable = [field_labels.get(f, f) for f in fields]

    if len(readable) == 1:
        question = f"I just need one more detail — {readable[0]}. Could you let me know?"
    elif len(readable) == 2:
        question = f"I need two more details: {readable[0]} and {readable[1]}."
    else:
        listed = ", ".join(readable[:-1]) + f", and {readable[-1]}"
        question = f"I need a few more details: {listed}."

    # Save context so the next user message can be combined with this request
    last_message = state["messages"][-1].content if state["messages"] else ""
    existing_pending = state.get("pending_clarification")
    # Keep the oldest pending (the original request), not the follow-up answers
    save_pending = existing_pending if existing_pending else last_message

    return {
        "messages":             [AIMessage(content=question)],
        "progress":             ["❓ Asking for missing details..."],
        "pending_clarification": save_pending,
        "pending_intent":        intent if intent != "unknown" else state.get("pending_intent"),
    }


# ── Node 3: Allocate budget ───────────────────────────────────────────────

def allocate_budget_node(state: GraphState) -> dict:
    logger.info("[NODE] allocate_budget")
    plan = _get_plan(state)
    if not plan:
        return {"progress": ["⚠️ No plan state found — skipping budget allocation."]}

    if not plan.total_budget:
        return {"progress": ["ℹ️ No budget provided — skipping allocation."]}

    result = BudgetManagerAgent().allocate_initial_budget(
        total_budget=plan.total_budget,
        duration_days=plan.duration_days,
        currency=plan.currency,
        user_preferences=plan.user_preferences,
    )

    plan_dict = _plan_to_dict(plan)
    plan_dict["budget_allocation"] = result.allocation.model_dump()

    return {
        "travel_plan": plan_dict,
        "progress":    [f"💰 Budget allocated — flights: €{result.allocation.flights:.0f} | "
                        f"hotels: €{result.allocation.hotels:.0f} | "
                        f"activities: €{result.allocation.activities:.0f} | "
                        f"buffer: €{result.allocation.buffer:.0f}"],
    }


# ── IATA lookup table ─────────────────────────────────────────────────────────
# Used as a hard fallback when the router passes a city name instead of a code.

_CITY_TO_IATA: dict[str, str] = {
    # Europe
    "frankfurt": "FRA", "frankfurt am main": "FRA",
    "lisbon": "LIS", "lisboa": "LIS",
    "london": "LHR", "london heathrow": "LHR", "london gatwick": "LGW",
    "paris": "CDG", "paris charles de gaulle": "CDG", "paris orly": "ORY",
    "berlin": "BER", "berlin brandenburg": "BER",
    "madrid": "MAD",
    "amsterdam": "AMS", "amsterdam schiphol": "AMS",
    "rome": "FCO", "roma": "FCO", "rome fiumicino": "FCO",
    "barcelona": "BCN",
    "munich": "MUC", "münchen": "MUC",
    "vienna": "VIE", "wien": "VIE",
    "zurich": "ZRH", "zürich": "ZRH",
    "dublin": "DUB",
    "brussels": "BRU", "bruxelles": "BRU",
    "prague": "PRG", "praha": "PRG",
    "warsaw": "WAW", "warszawa": "WAW",
    "budapest": "BUD",
    "bucharest": "OTP",
    "copenhagen": "CPH", "københavn": "CPH",
    "stockholm": "ARN",
    "oslo": "OSL",
    "helsinki": "HEL",
    "athens": "ATH", "athina": "ATH",
    "istanbul": "IST",
    "milan": "MXP", "milano": "MXP", "milan malpensa": "MXP",
    "porto": "OPO",
    "seville": "SVQ", "sevilla": "SVQ",
    "valencia": "VLC",
    "bilbao": "BIO",
    "zürich": "ZRH",
    "geneva": "GVA", "genève": "GVA",
    "nice": "NCE",
    "lyon": "LYS",
    "marseille": "MRS",
    "hamburg": "HAM",
    "dusseldorf": "DUS", "düsseldorf": "DUS",
    "cologne": "CGN", "köln": "CGN",
    "stuttgart": "STR",
    "nuremberg": "NUE", "nürnberg": "NUE",
    "leipzig": "LEJ",
    "palma": "PMI", "palma de mallorca": "PMI", "mallorca": "PMI",
    "tenerife": "TFS",
    "gran canaria": "LPA",
    "fuerteventura": "FUE",
    "lanzarote": "ACE",
    "ibiza": "IBZ",
    "madeira": "FNC", "funchal": "FNC",
    "azores": "PDL", "ponta delgada": "PDL",
    "reykjavik": "KEF",
    "edinburgh": "EDI",
    "manchester": "MAN",
    "birmingham": "BHX",
    "bristol": "BRS",
    "belfast": "BFS",
    "glasgow": "GLA",
    "sofia": "SOF",
    "zagreb": "ZAG",
    "sarajevo": "SJJ",
    "belgrade": "BEG",
    "skopje": "SKP",
    "tirana": "TIA",
    "chisinau": "KIV",
    "kyiv": "KBP", "kiev": "KBP",
    "minsk": "MSQ",
    "riga": "RIX",
    "tallinn": "TLL",
    "vilnius": "VNO",
    "luxembourg": "LUX",
    # North America
    "new york": "JFK", "new york city": "JFK", "nyc": "JFK",
    "los angeles": "LAX", "la": "LAX",
    "chicago": "ORD",
    "miami": "MIA",
    "toronto": "YYZ",
    "montreal": "YUL",
    "vancouver": "YVR",
    "boston": "BOS",
    "washington": "IAD", "washington dc": "IAD",
    "san francisco": "SFO",
    "seattle": "SEA",
    "dallas": "DFW",
    "houston": "IAH",
    "atlanta": "ATL",
    "cancun": "CUN",
    "mexico city": "MEX",
    # Middle East & Africa
    "dubai": "DXB",
    "abu dhabi": "AUH",
    "doha": "DOH",
    "cairo": "CAI",
    "casablanca": "CMN",
    "marrakech": "RAK",
    "nairobi": "NBO",
    "johannesburg": "JNB",
    "cape town": "CPT",
    # Asia & Pacific
    "singapore": "SIN",
    "tokyo": "NRT", "tokyo narita": "NRT", "tokyo haneda": "HND",
    "osaka": "KIX",
    "beijing": "PEK",
    "shanghai": "PVG",
    "hong kong": "HKG",
    "seoul": "ICN",
    "bangkok": "BKK", "bangkok suvarnabhumi": "BKK",
    "kuala lumpur": "KUL",
    "jakarta": "CGK",
    "bali": "DPS", "denpasar": "DPS",
    "delhi": "DEL", "new delhi": "DEL",
    "mumbai": "BOM",
    "sydney": "SYD",
    "melbourne": "MEL",
    "auckland": "AKL",
}


def _resolve_iata(value: str) -> str:
    """
    Returns the IATA code for a given city name or existing code.

    - If value is already a 3-letter IATA code → return as-is (uppercased).
    - If value is a city name in the lookup table → return the mapped code.
    - Otherwise → return the original value (FlightAgent will handle/fail gracefully).
    """
    if not value:
        return value

    stripped = value.strip()

    # Already looks like an IATA code (3 uppercase letters)
    if len(stripped) == 3 and stripped.isalpha():
        return stripped.upper()

    # Try lookup (case-insensitive)
    code = _CITY_TO_IATA.get(stripped.lower())
    if code:
        return code

    # Try partial match for longer strings e.g. "Frankfurt (FRA)"
    for city, iata in _CITY_TO_IATA.items():
        if city in stripped.lower():
            return iata

    # Return original — may still work for some APIs
    return stripped


# ── Node 4a: Search flights ───────────────────────────────────────────────

def search_flights_node(state: GraphState) -> dict:
    logger.info("[NODE] search_flights")
    """Runs flight search. No-op if skip_flights=True."""
    if state.get("skip_flights"):
        return {"progress": ["✈️ Flights skipped (driving/own transport)"]}

    plan = _get_plan(state)
    if not plan:
        return {}

    # ── Resolve IATA codes ────────────────────────────────────────────────
    # Priority: dedicated code fields → _resolve_iata fallback
    departure_code = plan.origin_code or _resolve_iata(plan.origin)
    arrival_code   = plan.destination_code or _resolve_iata(plan.destination)

    if not plan.origin_code and departure_code != plan.origin:
        print(f"   IATA fallback: '{plan.origin}' → '{departure_code}'")
    if not plan.destination_code and arrival_code != plan.destination:
        print(f"   IATA fallback: '{plan.destination}' → '{arrival_code}'")

    # ── Guard: if we still don't have valid codes, skip gracefully ────────
    if not departure_code or not arrival_code:
        return {"progress": ["⚠️ Cannot search flights — missing origin or destination."]}

    # ── Resolve return date ───────────────────────────────────────────────
    check_in  = plan.check_in_date
    check_out = plan.check_out_date
    intent    = state.get("intent", "")

    # Only auto-compute return date for full itinerary flows.
    # For flights_only, no return date = one-way search. Don't infer one
    # from duration_days — that would turn a one-way into a round-trip.
    if not check_out and check_in and plan.duration_days and intent != "flights_only":
        from datetime import date, timedelta
        try:
            check_out = (
                date.fromisoformat(check_in) + timedelta(days=plan.duration_days)
            ).isoformat()
        except ValueError:
            pass

    if not check_in:
        return {"progress": ["⚠️ Cannot search flights — missing departure date."]}

    constraints = state.get("active_constraints", {})
    flight_constraint = constraints.get("flight", "")

    budget = plan.budget_allocation.flights if plan.budget_allocation.flights > 0 else plan.total_budget * 0.35
    user_query = plan.user_preferences
    if flight_constraint:
        user_query = f"{user_query}. {flight_constraint}".strip(". ")

    def _do_search():
        result = FlightAgent().search_and_recommend(
            departure_airport=departure_code,
            arrival_airport=arrival_code,
            departure_date=check_in,
            return_date=check_out or "",
            budget=budget,
            adults=plan.adults,
            user_query=user_query,
        )
        plan_dict = {"flight_response": result.model_dump()}
        msg = "✈️ No flights found."
        if result.recommended_flight:
            msg = (
                f"✈️ Flight found — {result.recommended_flight.airline} "
                f"({result.recommended_flight.type}) "
                f"{departure_code} → {arrival_code} "
                f"€{result.recommended_flight.price:.0f} "
                f"User preferences: {plan.user_preferences or 'none'}"
            )
        return {"travel_plan": plan_dict, "progress": [msg]}

    return _run_with_timeout(
        _do_search,
        timeout_seconds=180,
        fallback_msg="✈️ Flight search timed out — continuing without flight results.",
    )


def _run_with_timeout(fn, timeout_seconds: int, fallback_msg: str):
    """
    Run fn() in a thread with a timeout.
    Returns fn()'s result or a graceful fallback dict if it times out.
    """
    from concurrent.futures import ThreadPoolExecutor, TimeoutError as FutureTimeout
    with ThreadPoolExecutor(max_workers=1) as ex:
        future = ex.submit(fn)
        try:
            return future.result(timeout=timeout_seconds)
        except FutureTimeout:
            future.cancel()
            return {"progress": [fallback_msg]}
        except Exception as e:
            return {"progress": [f"⚠️ {fallback_msg.split('—')[0].strip()}: {e}"]}


# ── Node 4b: Search hotels ────────────────────────────────────────────────

def search_hotels_node(state: GraphState) -> dict:
    logger.info("[NODE] search_hotels")
    """Runs hotel search. No-op if skip_hotels=True."""
    if state.get("skip_hotels"):
        return {"progress": ["🏨 Hotels skipped (own accommodation)"]}

    plan = _get_plan(state)
    if not plan:
        return {}

    # ── Resolve dates ─────────────────────────────────────────────────────
    check_in  = plan.check_in_date
    check_out = plan.check_out_date

    if not check_out and check_in and plan.duration_days:
        from datetime import date, timedelta
        try:
            check_out = (
                date.fromisoformat(check_in) + timedelta(days=plan.duration_days)
            ).isoformat()
        except ValueError:
            pass

    if not check_in or not check_out:
        return {"progress": ["⚠️ Hotel search skipped — missing check-in or check-out date."]}

    constraints      = state.get("active_constraints", {})
    hotel_constraint = constraints.get("hotel", "")

    # ── Compute nights from actual dates (more reliable than duration_days) ──
    nights = 0
    if check_in and check_out:
        from datetime import date as _date
        try:
            nights = (_date.fromisoformat(check_out) - _date.fromisoformat(check_in)).days
        except ValueError:
            pass
    if nights <= 0:
        nights = plan.duration_days  # fallback

    # ── Compute per-night budget ──────────────────────────────────────────
    # Priority order:
    # 1. Active constraint (from HITL approval e.g. "max €80/night")
    # 2. Budget allocation (from allocate_budget_node, 40% of total)
    # 3. Fallback: total_budget × 40% ÷ nights (for replans that bypass allocate_budget)
    # 4. 0 → hotel agent treats as unlimited
    if hotel_constraint:
        import re
        match = re.search(r"(\d+(?:\.\d+)?)", hotel_constraint)
        budget_per_night = float(match.group(1)) if match else 0.0
    elif plan.budget_allocation.hotels > 0 and nights > 0:
        budget_per_night = plan.budget_allocation.hotels / nights
    elif plan.total_budget > 0 and nights > 0:
        # replan bypassed allocate_budget — derive from total budget directly
        # Use 40% for hotels (matches default allocation split)
        budget_per_night = (plan.total_budget * 0.40) / nights
        logger.debug(
            "search_hotels: no allocation, derived budget_per_night=€%.0f "
            "(total=€%.0f × 0.40 ÷ %d nights)",
            budget_per_night, plan.total_budget, nights,
        )
    else:
        budget_per_night = 0.0

    prefs = plan.user_preferences
    if hotel_constraint:
        prefs = f"{prefs}. {hotel_constraint}".strip(". ")

    def _do_search():
        result = HotelAgent().search_and_recommend(
            query=f"Hotels in {plan.destination}",
            check_in_date=check_in,
            check_out_date=check_out,
            adults=plan.adults,
            currency=plan.currency,
            budget_per_night=budget_per_night,
            user_preferences=prefs,
        )

        plan_dict = {"hotel_response": result.model_dump(), "check_out_date": check_out}

        msg = "🏨 No hotels found."
        if result.recommended_hotel:
            msg = (
                f"🏨 Hotel found — {result.recommended_hotel.name} "
                f"€{result.recommended_hotel.price_per_night:.0f}/night"
            )
        return {"travel_plan": plan_dict, "progress": [msg]}

    return _run_with_timeout(
        _do_search,
        timeout_seconds=180,
        fallback_msg="🏨 Hotel search timed out — continuing without hotel results.",
    )


# ── Node 4c: Search attractions ───────────────────────────────────────────

def search_attractions_node(state: GraphState) -> dict:
    logger.info("[NODE] search_attractions")
    """Runs attractions search."""
    plan = _get_plan(state)
    if not plan:
        return {}

    constraints = state.get("active_constraints", {})
    prefs = plan.user_preferences
    activity_constraint = constraints.get("activities", "")
    if activity_constraint:
        prefs = f"{prefs}. {activity_constraint}".strip(". ")

    # Hotel context for distance enrichment — only available on re-runs
    # (first run: hotels execute in parallel so hotel_response is not yet set)
    hotel = None
    if plan.hotel_response and plan.hotel_response.recommended_hotel:
        hotel = plan.hotel_response.recommended_hotel

    def _do_search():
        # For attractions_only queries there are no trip days — default to 3
        # so the LLM targets ~12 recommendations instead of 0.
        days = plan.duration_days if plan.duration_days > 0 else 3

        result = AttractionsAgent().find_attractions(
            location=plan.destination,
            days=days,
            user_preferences=prefs,
            hotel=hotel,
            currency=plan.currency,
        )
        plan_dict = {"attractions_response": result.model_dump()}
        msg = "🗺️ No attractions found."
        if result.recommended_attractions:
            msg = f"🗺️ Found {len(result.recommended_attractions)} recommended attractions"
        return {"travel_plan": plan_dict, "progress": [msg]}

    return _run_with_timeout(
        _do_search,
        timeout_seconds=180,
        fallback_msg="🗺️ Attractions search timed out — continuing without attraction results.",
    )


# ── Node 5: Check budget ──────────────────────────────────────────────────

def check_budget_node(state: GraphState) -> dict:
    print("[NODE] check_budget")
    plan = _get_plan(state)
    if not plan:
        return {"progress": ["⚠️ Cannot check budget — no plan state."]}

    # No budget provided — skip check entirely
    if not plan.total_budget:
        return {"progress": ["ℹ️ No budget set — skipping budget check."]}

    # ── Skip for single-agent and hotel-only replan intents ──────────────
    # "under 200 euros per night" → replan_preferences with scope=["hotels"]
    # There's no full-trip budget to check in these cases.
    intent      = state.get("intent", "")
    replan_scope = set(state.get("replan_scope") or [])
    if intent in ("hotels_only", "flights_only", "attractions_only"):
        return {"progress": ["ℹ️ Single-agent query — skipping budget check."]}
    if intent.startswith("replan_") and replan_scope and replan_scope <= {"hotels", "attractions"}:
        # Replan touches only hotels/attractions — no full-trip budget to check
        return {"progress": ["ℹ️ Partial replan — skipping budget check."]}

    # ── Guard: only run budget check if we have a meaningful full-trip context ─
    has_flight      = (plan.flight_response is not None and
                       plan.flight_response.recommended_flight is not None)
    has_hotel       = (plan.hotel_response is not None and
                       plan.hotel_response.recommended_hotel is not None)
    has_attractions = (plan.attractions_response is not None and
                       bool(plan.attractions_response.recommended_attractions))

    # Need at least flight + hotel for a meaningful budget check
    if not (has_flight and has_hotel):
        return {"progress": ["ℹ️ Incomplete trip data — skipping budget check."]}

    logger.info(f"  [check_budget] available: "
          f"{'flights ' if has_flight else ''}"
          f"{'hotels ' if has_hotel else ''}"
          f"{'attractions' if has_attractions else ''}"
          f"| total_spent: €{plan.total_spent:.2f}")

    # If nothing billable yet (e.g. all free attractions + no flight/hotel)
    if plan.total_spent == 0:
        return {"progress": ["ℹ️ No costs recorded — skipping budget check."]}

    try:
        result: BudgetAgentResponse = BudgetManagerAgent().check_budget_status(plan)
    except Exception as e:
        return {"progress": [f"⚠️ Budget check failed ({str(e)}) — continuing to itinerary."]}

    msg = (
        f"✅ Within budget — spent €{result.total_spent:.0f} of €{result.total_budget:.0f}"
        if not result.is_over_budget
        else f"⚠️ Over budget by €{abs(result.difference):.0f}"
    )

    plan_dict = _plan_to_dict(plan)
    plan_dict["budget_response"] = result.model_dump()

    return {
        "travel_plan":   plan_dict,
        "budget_result": result.model_dump(),
        "progress":      [msg],
    }


# ── Node 6: HITL interrupt ────────────────────────────────────────────────

def hitl_node(state: GraphState) -> dict:
    logger.info("[NODE] hitl")
    """
    Pauses graph execution. Presents reallocation options to the user.
    Resumes when user sends a decision via Command(resume=...).
    """
    raw = state.get("budget_result")
    if not raw:
        # No budget result — skip HITL and go straight to itinerary
        return {"hitl_decision": "reject", "progress": ["⚠️ Budget result missing — building itinerary anyway."]}

    try:
        budget_result = BudgetAgentResponse(**raw)
    except Exception as e:
        return {"hitl_decision": "reject", "progress": [f"⚠️ Could not parse budget result ({e}) — building itinerary anyway."]}

    options_payload = {
        "over_budget_by":         abs(budget_result.difference),
        "primary_recommendation": budget_result.primary_recommendation,
        "constraint_to_apply":    budget_result.constraint_to_apply,
        "affected_agent":         budget_result.affected_agent,
        "savings_amount":         budget_result.savings_amount,
        "alternatives":           budget_result.alternative_recommendations or [],
        "explanation":            budget_result.explanation,
    }

    decision = interrupt(options_payload)

    return {
        "hitl_decision": decision,
        "progress": [f"User decision: {decision}"],
        "active_constraints": _build_active_constraints(
            decision, budget_result, state.get("active_constraints", {})
        ),
        "budget_warning": (
            None if decision == "approve"
            else f"⚠️ Note: trip is €{abs(budget_result.difference):.0f} over your budget."
        ),
    }


def _build_active_constraints(
    decision: str,
    budget_result: "BudgetAgentResponse",
    existing: dict,
) -> dict:
    """
    Build active_constraints from a HITL approval.
    Handles comma-separated affected_agent e.g. "hotel,activities" by splitting
    constraint_to_apply on " | " and mapping each part to its agent key.

    Examples:
      affected_agent="hotel"            constraint="Max hotel budget €49/night"
      → {"hotel": "Max hotel budget €49/night"}

      affected_agent="hotel,activities" constraint="Max hotel budget €49/night | Remove optional activities"
      → {"hotel": "Max hotel budget €49/night", "activities": "Remove optional activities"}
    """
    if decision != "approve" or not budget_result.affected_agent:
        return existing

    agents     = [a.strip() for a in budget_result.affected_agent.split(",")]
    constraint = budget_result.constraint_to_apply or ""
    parts      = [p.strip() for p in constraint.split("|")]

    constraints = dict(existing)
    for i, agent in enumerate(agents):
        constraints[agent] = parts[i] if i < len(parts) else constraint

    return constraints


# ── Node 7: Re-run affected agent ─────────────────────────────────────────

def rerun_agent_node(state: GraphState) -> dict:
    logger.info("[NODE] rerun_agent")
    budget_result = BudgetAgentResponse(**state["budget_result"])
    affected = budget_result.affected_agent

    # Support comma-separated agents e.g. "hotel,activities" or "hotel,flight"
    agents = [a.strip() for a in (affected or "").split(",") if a.strip()]

    if not agents:
        return {"progress": ["⚠️ Unknown affected agent — skipping re-run."]}

    # Single agent — simple path
    if len(agents) == 1:
        a = agents[0]
        if a == "flight":       return search_flights_node(state)
        elif a == "hotel":      return search_hotels_node(state)
        elif a == "activities": return search_attractions_node(state)
        return {"progress": [f"⚠️ Unknown affected agent '{a}' — skipping re-run."]}

    # Multiple agents — run in parallel
    logger.info("[rerun_agent] running %d agents in parallel: %s", len(agents), agents)

    def _run(agent_name: str) -> dict:
        if agent_name == "flight":       return search_flights_node(state)
        elif agent_name == "hotel":      return search_hotels_node(state)
        elif agent_name == "activities": return search_attractions_node(state)
        return {}

    updates: list[dict] = []
    with ThreadPoolExecutor(max_workers=len(agents)) as ex:
        futures = {ex.submit(_run, a): a for a in agents}
        for future in futures:
            try:
                updates.append(future.result())
            except Exception as e:
                updates.append({"progress": [f"⚠️ {futures[future]} rerun failed: {e}"]})

    merged_plan = state.get("travel_plan") or {}
    progress: list[str] = []
    for upd in updates:
        if upd.get("travel_plan"):
            merged_plan = _merge_travel_plan(merged_plan, upd["travel_plan"])
        progress.extend(upd.get("progress", []))

    return {"travel_plan": merged_plan, "progress": progress or ["🔄 Rerunning agents..."]}


# ── Node 8: Build itinerary ───────────────────────────────────────────────

def build_itinerary_node(state: GraphState) -> dict:
    logger.info("[NODE] build_itinerary")
    plan = _get_plan(state)
    if not plan:
        return {"progress": ["⚠️ Cannot build itinerary — no plan state."]}

    outbound      = plan.flight_response.recommended_flight        if plan.flight_response else None
    return_flight = plan.flight_response.recommended_return_flight if plan.flight_response else None
    hotel         = plan.hotel_response.recommended_hotel          if plan.hotel_response  else None
    attractions   = (
        plan.attractions_response.recommended_attractions
        if plan.attractions_response else []
    )

    try:

        result = ItineraryAgent().build(
            destination=plan.destination,
            outbound_flight=outbound,
            return_flight=return_flight,
            hotel=hotel,
            attractions=attractions,
            user_preferences=plan.user_preferences,
            currency=plan.currency,
        )

        # ── Guard: itinerary agent returned no_results ────────────────────
        # Can arrive as a dict with status='no_results' if the LLM returned
        # an error payload instead of a full ItineraryAgentResponse.
        if hasattr(result, "status") and result.status == "no_results":
            msg = getattr(result, "error_message", "insufficient data to build itinerary")
            return {"progress": [f"⚠️ Itinerary could not be built — {msg}"]}
        # round-trip total (return is just the confirmed version of outbound).
        if return_flight and return_flight.price:
            flights_cost = return_flight.price
            logger.info(f"  [build_itinerary] flights: round-trip total €{flights_cost:.0f}")
        elif outbound and outbound.price:
            flights_cost = outbound.price   # one-way
            logger.info(f"  [build_itinerary] flights: one-way €{flights_cost:.0f}")
        else:
            flights_cost = 0.0

        hotel_cost       = hotel.total_price if hotel else 0.0
        attractions_cost = sum(a.price for a in attractions if a.price)
        total_cost       = flights_cost + hotel_cost + attractions_cost

        # Patch the result's cost fields with correct Python-computed values
        result_dict = result.model_dump()
        result_dict["total_estimated_cost"] = round(total_cost, 2)
        result_dict["cost_breakdown"] = {
            "flights":     round(flights_cost, 2),
            "hotel":       round(hotel_cost, 2),
            "attractions": round(attractions_cost, 2),
            "total":       round(total_cost, 2),
        }

        plan_dict = _plan_to_dict(plan)
        plan_dict["itinerary_response"] = result_dict
        plan_dict["status"] = "complete"

        progress_msgs = [f"📅 Itinerary ready — {result.trip_title}"]
        if state.get("budget_warning"):
            progress_msgs.append(state["budget_warning"])

        import json as _json
        payload = {
            "itinerary":     result_dict,
            "flight":        outbound.model_dump()      if outbound      else None,
            "return_flight": return_flight.model_dump() if return_flight else None,
            "hotel":         hotel.model_dump()         if hotel         else None,
            "budget_warning": state.get("budget_warning"),
            "currency":      plan.currency,
        }
        msg_content = "__ITINERARY__:" + _json.dumps(payload)

        return {
            "travel_plan": plan_dict,
            "messages":    [AIMessage(content=msg_content)],
            "progress":    progress_msgs,
        }

    except Exception as e:
        return {"progress": [f"⚠️ Itinerary build failed: {str(e)}"]}


# ── Node 9: Single agent handler (flights/hotels/attractions only) ─────────

def single_agent_node(state: GraphState) -> dict:
    logger.info("[NODE] single_agent (intent=%s)", state.get("intent", ""))
    import json as _json
    intent = state.get("intent", "")
    plan_base = _get_plan(state)

    # ── Flights only ──────────────────────────────────────────────────────
    if intent == "flights_only":
        result = search_flights_node(state)
        merged = {**state}
        if result.get("travel_plan"):
            merged["travel_plan"] = _merge_travel_plan(
                state.get("travel_plan"), result.get("travel_plan")
            )
        plan = _get_plan(merged)

        if plan and plan.flight_response:
            fr = plan.flight_response
            payload = {
                "recommended":  fr.recommended_flight.model_dump() if fr.recommended_flight else None,
                "return_flight": fr.recommended_return_flight.model_dump() if fr.recommended_return_flight else None,
                "alternatives": [f.model_dump() for f in (fr.alternative_options or [])[:4]],
                "reasoning":    fr.reasoning or "",
                "currency":     plan_base.currency if plan_base else "EUR",
            }
            msg = "__FLIGHTS__:" + _json.dumps(payload)
            result["messages"] = [AIMessage(content=msg)]
        return result

    # ── Hotels only ───────────────────────────────────────────────────────
    elif intent == "hotels_only":
        result = search_hotels_node(state)
        merged = {**state}
        if result.get("travel_plan"):
            merged["travel_plan"] = _merge_travel_plan(
                state.get("travel_plan"), result.get("travel_plan")
            )
        plan = _get_plan(merged)

        if plan and plan.hotel_response:
            hr = plan.hotel_response
            payload = {
                "recommended":  hr.recommended_hotel.model_dump() if hr.recommended_hotel else None,
                "alternatives": [h.model_dump() for h in (hr.alternative_options or [])[:4]],
                "price_range":  hr.price_range,
                "rating_range": hr.rating_range,
                "reasoning":    hr.reasoning or "",
                "currency":     plan_base.currency if plan_base else "EUR",
            }
            msg = "__HOTELS__:" + _json.dumps(payload)
            result["messages"] = [AIMessage(content=msg)]
        return result

    # ── Attractions only ──────────────────────────────────────────────────
    elif intent == "attractions_only":
        result = search_attractions_node(state)
        merged = {**state}
        if result.get("travel_plan"):
            merged["travel_plan"] = _merge_travel_plan(
                state.get("travel_plan"), result.get("travel_plan")
            )
        plan = _get_plan(merged)

        if plan and plan.attractions_response:
            ar = plan.attractions_response
            payload = {
                "destination":   plan.destination,
                "recommended":   [a.model_dump() for a in (ar.recommended_attractions or [])],
                "alternatives":  [a.model_dump() for a in (ar.alternative_attractions or [])[:6]],
                "estimated_cost": ar.estimated_total_cost,
                "reasoning":     ar.reasoning or "",
                "currency":      plan_base.currency if plan_base else "EUR",
            }
            msg = "__ATTRACTIONS__:" + _json.dumps(payload)
            result["messages"] = [AIMessage(content=msg)]
        return result

    return {"progress": ["⚠️ Unknown single-agent intent."]}


# ============================================================================
# ROUTING FUNCTIONS (conditional edges)
# ============================================================================

def route_after_extract(state: GraphState) -> str:
    """Decides what to do after extract_params."""
    intent = state.get("intent", "unknown")

    # ── Filter LLM-reported missing fields by intent ──────────────────────
    # The router sometimes flags fields that aren't needed for the intent.
    # e.g. origin_code/destination_code for hotels_only (IATA codes irrelevant)
    # Strip them out before deciding whether to ask the user.
    raw_missing = state.get("missing_fields") or []
    if raw_missing:
        irrelevant_for_intent = {
            # budget is optional everywhere — system handles missing budget gracefully
            # (skips allocation and budget check, proceeds without constraint)
            "full_itinerary":   {"budget", "origin_code", "destination_code", "check_in_date", "check_out_date"},
            "hotels_only":      {"origin", "origin_code", "destination_code", "check_in_date", "check_out_date"},
            "attractions_only": {"origin", "origin_code", "destination_code", "return_date", "duration_days", "budget", "check_in_date", "check_out_date"},
            "flights_only":     {"budget", "destination_code", "origin_code", "check_in_date", "check_out_date"},
        }
        irrelevant = irrelevant_for_intent.get(intent, {"budget", "origin_code", "destination_code"})
        filtered = [f for f in raw_missing if f not in irrelevant]
        if filtered != raw_missing:
            state["missing_fields"] = filtered
            logger.debug("route_after_extract: stripped irrelevant missing fields %s → %s", raw_missing, filtered)

    if state.get("missing_fields"):
        return "ask_user"

    if intent == "unknown":
        return "ask_user"

    # ── Python-enforced validation — don't trust LLM to always catch these ─
    plan = _get_plan(state)
    if plan:
        computed_missing = []

        needs_flights = (
            intent in ("full_itinerary", "flights_only", "replan_duration", "replan_preferences")
            and not state.get("skip_flights")
        )
        needs_destination = intent in (
            "full_itinerary", "flights_only", "hotels_only",
            "attractions_only", "replan_duration", "replan_budget", "replan_preferences"
        )
        # flights_only: only needs departure date (one-way is valid — no return needed)
        # hotels_only: needs check_in, check_out or duration
        # full_itinerary / replan: needs both departure and return/duration
        needs_departure_date = intent in (
            "full_itinerary", "flights_only", "hotels_only", "replan_duration"
        )
        needs_return_or_duration = intent in (
            "full_itinerary", "hotels_only", "replan_duration"
            # flights_only excluded — one-way is fine
        )

        if needs_destination and not plan.destination:
            computed_missing.append("destination")
        if needs_flights and not plan.origin and not plan.origin_code:
            computed_missing.append("origin city")
        if needs_departure_date and not plan.check_in_date:
            computed_missing.append("departure date")
        if needs_return_or_duration and not plan.check_out_date and not plan.duration_days:
            computed_missing.append("return date or trip duration")

        if computed_missing:
            state["missing_fields"] = computed_missing
            print(f"  [route_after_extract] computed_missing={computed_missing}")
            return "ask_user"

    if intent in ("flights_only", "hotels_only", "attractions_only"):
        return "single_agent"

    if intent.startswith("replan_"):
        return "replan_search"

    # Full itinerary — start with budget allocation
    return "allocate_budget"


def route_after_budget_check(state: GraphState) -> str:
    plan = _get_plan(state)
    if not plan or not plan.total_budget:
        return "build_itinerary"
    budget_result = state.get("budget_result")
    if not budget_result:
        return "build_itinerary"
    result = BudgetAgentResponse(**budget_result)
    return "hitl" if result.is_over_budget else "build_itinerary"


def route_after_hitl(state: GraphState) -> str:
    decision = state.get("hitl_decision", "reject")
    return "rerun_agent" if decision == "approve" else "build_itinerary"


# ── Replan routing: fan out only the agents in replan_scope ──────────────

def replan_search_node(state: GraphState) -> dict:
    logger.info(f"[NODE] replan_search (scope={state.get('replan_scope',[])})")
    intent = state.get("intent", "")
    scope  = set(state.get("replan_scope", []))
    # ── For replan_preferences, extract per-category constraints from last message ──
    # Read from the raw last message — user_preferences may not contain the number.
    # "reduce hotel to €200/night" → active_constraints["hotel"] = "Max hotel budget €200/night"
    if intent == "replan_preferences":
        import re
        last_msg = (state["messages"][-1].content if state.get("messages") else "") or ""
        existing_constraints = dict(state.get("active_constraints") or {})

        # Per-night hotel cap: "200 euros per night", "€200/night", "200/night"
        m = re.search(
            r'(\d+(?:\.\d+)?)\s*(?:euros?|eur|€)?\s*(?:per\s*night|/night|a\s*night)',
            last_msg,
            re.IGNORECASE,
        )
        if m:
            cap = float(m.group(1))
            if cap > 5:  # sanity guard — ignore implausibly low values
                existing_constraints["hotel"] = f"Max hotel budget €{cap:.0f}/night"
                state = {**state, "active_constraints": existing_constraints}
                logger.info("replan_preferences: hotel constraint → €%.0f/night", cap)

    updates: list[dict] = []

    # ── Re-allocate budget only when the total trip budget actually changed ──
    # replan_budget = user changed the total (e.g. "budget is €2000") → re-run allocation
    # replan_preferences = user changed a category preference (e.g. "hotel to €100/night")
    #   → do NOT re-allocate from total; just pass the preference as a constraint
    if "budget" in scope and intent == "replan_budget":
        alloc_result = allocate_budget_node(state)
        if alloc_result.get("travel_plan"):
            updates.append(alloc_result)
            state = {**state, "travel_plan": _merge_travel_plan(
                state.get("travel_plan"), alloc_result.get("travel_plan")
            )}

    # Run affected agents in parallel using ThreadPoolExecutor
    def run_agent(agent_name: str) -> dict:
        if agent_name == "flights":
            return search_flights_node(state)
        elif agent_name == "hotels":
            return search_hotels_node(state)
        elif agent_name == "attractions":
            return search_attractions_node(state)
        return {}

    agents_to_run = [a for a in ["flights", "hotels", "attractions"] if a in scope]

    if agents_to_run:
        with ThreadPoolExecutor(max_workers=len(agents_to_run)) as executor:
            futures = {executor.submit(run_agent, a): a for a in agents_to_run}
            for future in futures:
                try:
                    updates.append(future.result())
                except Exception as e:
                    updates.append({"progress": [f"⚠️ {futures[future]} search failed: {e}"]})

    # Merge all updates
    merged_plan: dict = state.get("travel_plan") or {}
    progress: list[str] = []

    for upd in updates:
        if "travel_plan" in upd and upd["travel_plan"]:
            merged_plan = _merge_travel_plan(merged_plan, upd["travel_plan"])
        if "progress" in upd:
            progress.extend(upd["progress"])

    return {
        "travel_plan": merged_plan,
        "progress":    progress or ["🔄 Replanning..."],
    }


# ============================================================================
# GRAPH ASSEMBLY
# ============================================================================

def build_graph(checkpointer=None) -> Any:
    """
    Builds and compiles the travel planning LangGraph.

    Args:
        checkpointer: LangGraph checkpointer. Defaults to InMemorySaver.
                      In production use AsyncPostgresSaver.
    Returns:
        Compiled LangGraph graph.
    """
    builder = StateGraph(GraphState)

    # ── Register nodes ────────────────────────────────────────────────────
    builder.add_node("extract_params",    extract_params_node)
    builder.add_node("ask_user",          ask_user_node)
    builder.add_node("allocate_budget",   allocate_budget_node)
    builder.add_node("search_flights",    search_flights_node)
    builder.add_node("search_hotels",     search_hotels_node)
    builder.add_node("search_attractions",search_attractions_node)
    builder.add_node("check_budget",      check_budget_node)
    builder.add_node("hitl",              hitl_node)
    builder.add_node("rerun_agent",       rerun_agent_node)
    builder.add_node("build_itinerary",   build_itinerary_node)
    builder.add_node("single_agent",      single_agent_node)
    builder.add_node("replan_search",     replan_search_node)

    # ── Entry ─────────────────────────────────────────────────────────────
    builder.add_edge(START, "extract_params")

    # ── After extract_params: route to the right pipeline ─────────────────
    builder.add_conditional_edges(
        "extract_params",
        route_after_extract,
        {
            "ask_user":       "ask_user",
            "single_agent":   "single_agent",
            "allocate_budget":"allocate_budget",
            "replan_search":  "replan_search",
        },
    )

    # ── Dead ends ──────────────────────────────────────────────────────────
    builder.add_edge("ask_user",    END)
    builder.add_edge("single_agent",END)

    # ── Full itinerary pipeline ────────────────────────────────────────────
    # Fan-out: allocate_budget → 3 parallel search nodes
    builder.add_edge("allocate_budget", "search_flights")
    builder.add_edge("allocate_budget", "search_hotels")
    builder.add_edge("allocate_budget", "search_attractions")

    # Fan-in: all 3 parallel nodes → check_budget
    # LangGraph waits for all incoming edges to complete before running check_budget
    builder.add_edge("search_flights",     "check_budget")
    builder.add_edge("search_hotels",      "check_budget")
    builder.add_edge("search_attractions", "check_budget")

    # Replan also fans in to check_budget
    builder.add_edge("replan_search", "check_budget")

    # ── Budget routing ────────────────────────────────────────────────────
    builder.add_conditional_edges(
        "check_budget",
        route_after_budget_check,
        {
            "hitl":            "hitl",
            "build_itinerary": "build_itinerary",
        },
    )

    # ── HITL routing ──────────────────────────────────────────────────────
    builder.add_conditional_edges(
        "hitl",
        route_after_hitl,
        {
            "rerun_agent":     "rerun_agent",
            "build_itinerary": "build_itinerary",
        },
    )

    builder.add_edge("rerun_agent",     "build_itinerary")
    builder.add_edge("build_itinerary", END)

    # ── Compile ───────────────────────────────────────────────────────────
    if checkpointer is None:
        checkpointer = InMemorySaver()

    return builder.compile(checkpointer=checkpointer)


# ============================================================================
# ENTRY POINT — TravelOrchestrator
# ============================================================================

class TravelOrchestrator:
    """
    High-level interface for the travel planning chatbot.

    Usage:
        orchestrator = TravelOrchestrator()

        # Turn 1
        for event in orchestrator.chat("Plan a trip to Madeira, €3000, 8 days", thread_id="abc"):
            print(event)

        # HITL approval (if over budget)
        orchestrator.resume(decision="approve", thread_id="abc")

        # Turn 2 — follow-up
        for event in orchestrator.chat("What if I go for 5 days instead?", thread_id="abc"):
            print(event)
    """

    def __init__(self, checkpointer=None):
        self.graph = build_graph(checkpointer=checkpointer)

    def chat(self, user_message: str, thread_id: str):
        """
        Processes a user message. Yields progress events and the final response.

        Yields:
            dict with keys: type ("progress" | "response" | "hitl" | "error"), data
        """
        log_user_message(logger, user_message)
        config  = {"configurable": {"thread_id": thread_id}}
        initial = {"messages": [HumanMessage(content=user_message)]}
        hitl_yielded = False

        try:
            for chunk in self.graph.stream(initial, config=config, stream_mode="updates"):
                for node_name, node_output in chunk.items():

                    # ── HITL interrupt — two possible shapes ──────────────
                    # Shape A: node_output is a tuple of Interrupt objects
                    #          e.g. {"hitl": (Interrupt(value={...}),)}
                    # Shape B: LangGraph emits {"__interrupt__": (Interrupt(...),)}
                    if isinstance(node_output, tuple):
                        interrupt_obj   = node_output[0] if node_output else None
                        interrupt_value = getattr(interrupt_obj, "value", {}) if interrupt_obj else {}
                        yield {"type": "hitl", "data": interrupt_value}
                        hitl_yielded = True
                        continue

                    # ── Progress events ───────────────────────────────────
                    if isinstance(node_output, dict):
                        for msg in node_output.get("progress", []):
                            yield {"type": "progress", "node": node_name, "data": msg}

                        # ── Final assistant message ───────────────────────
                        for message in node_output.get("messages", []):
                            if isinstance(message, AIMessage):
                                yield {"type": "response", "data": message.content}

        except Exception as e:
            yield {"type": "error", "data": str(e)}
            return

        # ── Safety net: check if graph paused at an interrupt we missed ───
        # This catches cases where interrupt() fired but wasn't streamed
        # (can happen with nested thread pools or LangGraph version differences)
        if not hitl_yielded:
            try:
                state = self.graph.get_state(config)
                if state and state.next and state.tasks:
                    for task in state.tasks:
                        if hasattr(task, "interrupts") and task.interrupts:
                            for intr in task.interrupts:
                                value = getattr(intr, "value", {})
                                yield {"type": "hitl", "data": value}
                                hitl_yielded = True
                                break
            except Exception:
                pass

    def resume(self, decision: str, thread_id: str):
        """
        Resumes a HITL-interrupted graph with the user's decision.

        Yields:
            Same as chat() — progress events and final response.
        """
        from langgraph.types import Command

        config = {"configurable": {"thread_id": thread_id}}

        try:
            for chunk in self.graph.stream(
                Command(resume=decision),
                config=config,
                stream_mode="updates",
            ):
                for node_name, node_output in chunk.items():
                    if isinstance(node_output, tuple):
                        interrupt_obj   = node_output[0] if node_output else None
                        interrupt_value = getattr(interrupt_obj, "value", {}) if interrupt_obj else {}
                        yield {"type": "hitl", "data": interrupt_value}
                        continue
                    if isinstance(node_output, dict):
                        for msg in node_output.get("progress", []):
                            yield {"type": "progress", "node": node_name, "data": msg}
                        for message in node_output.get("messages", []):
                            if isinstance(message, AIMessage):
                                yield {"type": "response", "data": message.content}
        except Exception as e:
            yield {"type": "error", "data": str(e)}

    def get_state(self, thread_id: str) -> Optional[TravelPlanState]:
        """Returns the current TravelPlanState for a thread."""
        config = {"configurable": {"thread_id": thread_id}}
        state = self.graph.get_state(config)
        if state and state.values.get("travel_plan"):
            try:
                return TravelPlanState(**state.values["travel_plan"])
            except Exception:
                return None
        return None