"""
budget_agent.py
---------------
Budget Manager Agent — minimal LLM output approach.

WHAT PYTHON DOES:
  - All analysis (overage, per-category breakdown, reallocation options)
  - Detects when overage is so large that no reallocation can fix it
  - Computes concrete per-category budget caps (not just vague "downgrade")
  - Passes a pre-ranked, pre-filtered option list to the LLM

WHAT THE LLM DOES (only):
  - Picks the best strategy from the Python-computed options
  - Writes a human-readable explanation
"""

from typing import Optional
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

from src.logger import get_logger, log_agent_run
from src.models import (
    TravelPlanState,
    BudgetAgentResponse,
    BudgetAllocation,
    BudgetDecision
)
from src.tools.budget_tools import (
    allocate_budget,
    calculate_budget_overage,
    analyze_reallocation_options,
    analyze_tradeoffs,
    generate_budget_status_report,
)

logger = get_logger(__name__)


class BudgetManagerAgent:
    """
    Budget Manager Agent — handles all budget-related decisions.

    Usage:
        agent = BudgetManagerAgent()
        response = agent.allocate_initial_budget(total_budget=3000, duration_days=8)
        response = agent.check_budget_status(state)
    """

    _SYSTEM_PROMPT = """You are a Budget Manager Agent for a travel planning system.

You receive a pre-computed budget analysis and ranked reallocation options.
Your ONLY job is to pick the best strategy and explain it.

OUTPUT — these fields only:
  primary_recommendation:      best action in plain English (be specific: name the category and amount)
  affected_agent:              "flight" | "hotel" | "activities" | "hotel,flight" | null
                               Use comma-separated values when two agents must rerun together
                               e.g. "hotel,flight" when neither alone covers the deficit
  constraint_to_apply:         exact instruction to pass to that agent (include specific € amounts)
                               For multi-agent: separate constraints with " | "
                               e.g. "Max hotel budget €50/night | Max flight budget €300 total"
  alternative_recommendations: up to 2 fallback strategies (short, specific)
  explanation:                 2-3 sentences: what is over budget, by how much, and what will help

IMPORTANT:
- The options are already ranked best-first. Prefer option 1 unless there is a strong reason not to.
- If a single option covers the deficit → use one agent only.
- If no single option covers the deficit but two combined would → use "agent1,agent2".
- constraint_to_apply MUST include a specific euro amount. Never vague instructions.
  Good: "Max hotel budget €57/night"
  Bad:  "Find a cheaper hotel"
- If the best option is "no_fix_possible", set affected_agent=null and tell the user honestly
  that the budget is too low and they should either increase it or choose a closer destination.
- DO NOT re-compute numbers. Use only what is given.
"""

    def __init__(
        self,
        model_name: str = "gpt-5-nano",
        temperature: float = 0.0,
        verbose: bool = True,
    ):
        self.verbose = verbose
        self.agent = create_agent(
            model=ChatOpenAI(model=model_name, temperature=temperature),
            tools=[],
            system_prompt=self._SYSTEM_PROMPT,
            response_format=BudgetDecision,
            name="budget_decision_agent",
        )

    # ── Allocation (pure Python — no LLM needed) ──────────────────────────────

    def allocate_initial_budget(
        self,
        total_budget: float,
        duration_days: int,
        currency: str = "EUR",
        user_preferences: str = "",
    ) -> BudgetAgentResponse:
        logger.info("Allocating budget: %.2f %s / %d days", total_budget, currency, duration_days)
        if user_preferences:
            logger.debug("Budget preferences: %s", user_preferences)

        result = allocate_budget.invoke({
            "total_budget":     total_budget,
            "duration_days":    duration_days,
            "user_preferences": user_preferences,
        })

        allocation = BudgetAllocation(**result["allocation"])

        logger.info(
            "Budget allocation — flights=%.1f hotels=%.1f activities=%.1f buffer=%.1f",
            allocation.flights, allocation.hotels, allocation.activities, allocation.buffer,
        )
        if result.get("skipped"):
            logger.debug("Skipped categories: %s", ", ".join(result["skipped"]))

        return BudgetAgentResponse(
            status="allocated",
            allocation=allocation,
            total_budget=total_budget,
            total_spent=0.0,
            difference=total_budget,
            is_over_budget=False,
            explanation=result.get("strategy", "Budget allocated successfully."),
        )

    # ── Budget check ──────────────────────────────────────────────────────────

    def check_budget_status(self, state: TravelPlanState) -> BudgetAgentResponse:
        logger.info(
            "Checking budget: %.2f / %.2f %s",
            state.total_spent, state.total_budget, state.currency,
        )

        expenses = [e.model_dump() for e in state.as_expenses()]

        # ── Detect skipped categories ─────────────────────────────────────────
        prefs = (state.user_preferences or "").lower()
        skip_flights = any(w in prefs for w in [
            "no flight", "no flights", "driving", "own car", "my car", "car trip", "car"
            "road trip", "by car", "taking the car", "train", "bus", "bike", "cycling", "bicycle",
        ])
        skip_hotels = any(w in prefs for w in [
            "no hotel", "friend's place", "friends place", "staying with",
            "family house", "own place", "airbnb already", "accommodation sorted",
            "have accommodation", "free accommodation",
        ])
        skip_activities = any(w in prefs for w in [
            "no activities", "no sightseeing", "just relaxing",
        ])

        # ── Step 1: Overage analysis ──────────────────────────────────────────
        overage = calculate_budget_overage.invoke({
            "total_budget":     state.total_budget - state.budget_allocation.buffer,
            "current_expenses": expenses,
        })

        # ── Step 2: Within budget ─────────────────────────────────────────────
        if not overage["is_over_budget"]:
            report = generate_budget_status_report.invoke({
                "total_budget":        state.total_budget - state.budget_allocation.buffer,
                "expenses":            expenses,
                "reallocation_options": [],
            })
            logger.info("Within budget")
            return BudgetAgentResponse(
                status="within_budget",
                total_budget=state.total_budget,
                total_spent=overage["total_spent"],
                difference=state.total_budget - overage["total_spent"],
                is_over_budget=False,
                explanation=report["summary"],
            )

        # ── Step 3: Over budget ───────────────────────────────────────────────
        deficit = overage["overage_amount"]
        logger.warning("Over budget by %.2f", deficit)

        # ── Compute nights for per-night budget ───────────────────────────────
        nights = state.duration_days or 1
        if state.check_in_date and state.check_out_date:
            from datetime import date as _date
            try:
                nights = (_date.fromisoformat(state.check_out_date) -
                          _date.fromisoformat(state.check_in_date)).days or nights
            except ValueError:
                pass

        # ── Build smarter reallocation options in Python ──────────────────────
        # Rather than relying solely on analyze_reallocation_options, compute
        # concrete alternatives with specific budget caps here.
        smart_options = []

        flight_cost = state.flight_cost
        hotel_cost  = state.hotel_cost
        total_spent = overage["total_spent"]

        # Option A: Flight budget cap
        # Added when: (a) flights exceeded their allocation, OR
        # (b) flights are ≥25% of total spent AND a cheaper alternative exists.
        # This ensures flights are always considered when they're a meaningful cost.
        if not skip_flights and flight_cost > 0:
            flight_budget = state.budget_allocation.flights or state.total_budget * 0.40

            # Case 1: flight exceeded its allocation
            if flight_cost > flight_budget:
                smart_options.append({
                    "strategy_id":       "flight_budget_cap",
                    "category":          "flights",
                    "action":            f"Search flights with max budget €{flight_budget:.0f}",
                    "potential_savings": round(flight_cost - flight_budget, 2),
                    "impact_score":      3,
                    "affected_agent":    "flight",
                    "feasibility":       "medium",
                    "constraint":        f"Max flight budget €{flight_budget:.0f} total",
                })
            # Case 2: flight within allocation but is a big chunk — try cheaper alternatives
            elif flight_cost > 0 and total_spent > 0 and flight_cost / total_spent >= 0.25:
                # Suggest searching with a tighter cap (20% less than current)
                tighter_cap = round(flight_cost * 0.80)
                if tighter_cap > 50:  # sanity check
                    smart_options.append({
                        "strategy_id":       "flight_tighter_cap",
                        "category":          "flights",
                        "action":            f"Search for cheaper flights under €{tighter_cap:.0f}",
                        "potential_savings": round(flight_cost - tighter_cap, 2),
                        "impact_score":      3,
                        "affected_agent":    "flight",
                        "feasibility":       "medium",
                        "constraint":        f"Max flight budget €{tighter_cap:.0f} total",
                    })

        # Option B: Max hotel budget cap (per night from allocation)
        if not skip_hotels and hotel_cost > 0:
            hotel_budget   = state.budget_allocation.hotels or state.total_budget * 0.35
            max_per_night  = hotel_budget / nights if nights > 0 else hotel_budget
            if hotel_cost > hotel_budget:
                smart_options.append({
                    "strategy_id":       "hotel_budget_cap",
                    "category":          "hotels",
                    "action":            f"Search hotels with max €{max_per_night:.0f}/night",
                    "potential_savings": round(hotel_cost - hotel_budget, 2),
                    "impact_score":      4,
                    "affected_agent":    "hotel",
                    "feasibility":       "high",
                    "constraint":        f"Max hotel budget €{max_per_night:.0f}/night",
                })

        # Option C: Existing alternatives from flight agent (cheapest alternative)
        current_flight_type    = "unknown"
        cheapest_alt_flight    = 0.0
        if not skip_flights and state.flight_response:
            if state.flight_response.recommended_flight:
                current_flight_type = state.flight_response.recommended_flight.type
            if state.flight_response.alternative_options:
                sorted_alts = sorted(state.flight_response.alternative_options, key=lambda f: f.price)
                cheapest_alt_flight = sorted_alts[0].price if sorted_alts else 0.0

        # Option D: Cheapest hotel alternative
        cheaper_hotel_total = 0.0
        cheaper_hotel_ppn   = 0.0
        cheaper_hotel_tier  = "unknown"
        if not skip_hotels and state.hotel_response and state.hotel_response.alternative_options:
            sorted_hotel_alts = sorted(
                state.hotel_response.alternative_options,
                key=lambda h: h.total_price,
            )
            alt = sorted_hotel_alts[0]
            cheaper_hotel_total = alt.total_price
            cheaper_hotel_ppn   = alt.price_per_night
            cheaper_hotel_tier  = alt.type

        if cheaper_hotel_ppn > 0 and cheaper_hotel_total < hotel_cost:
            smart_options.append({
                "strategy_id":       "hotel_alt",
                "category":          "hotels",
                "action":            f"Switch to alternative hotel at €{cheaper_hotel_ppn:.0f}/night",
                "potential_savings": round(hotel_cost - cheaper_hotel_total, 2),
                "impact_score":      4,
                "affected_agent":    "hotel",
                "feasibility":       "high",
                "constraint":        f"Max hotel budget €{cheaper_hotel_ppn:.0f}/night",
            })

        # Option E: Buffer
        buffer = state.budget_allocation.buffer
        if buffer > 0:
            smart_options.append({
                "strategy_id":       "use_buffer",
                "category":          "buffer",
                "action":            f"Use buffer fund (€{buffer:.0f})",
                "potential_savings": min(deficit, buffer),
                "impact_score":      1,
                "affected_agent":    "none",
                "feasibility":       "high",
                "constraint":        None,
            })

        # Also pull standard options from the tool for activities
        tool_options = analyze_reallocation_options.invoke({
            "deficit":                          deficit,
            "current_flight_price":             0.0 if skip_flights else flight_cost,
            "current_flight_type":              "unknown" if skip_flights else current_flight_type,
            "cheapest_connecting_flight_price": 0.0 if skip_flights else cheapest_alt_flight,
            "current_hotel_price":              0.0 if skip_hotels else hotel_cost,
            "cheaper_hotel_price":              0.0 if skip_hotels else cheaper_hotel_total,
            "cheaper_hotel_tier":               "unknown" if skip_hotels else cheaper_hotel_tier,
            "optional_activities_cost":         0.0 if skip_activities else state.attractions_cost,
            "optional_activities_count":        0 if skip_activities else len(
                state.attractions_response.recommended_attractions
                if state.attractions_response else []
            ),
            "buffer_available":                 buffer,
            "trip_duration_days":               state.duration_days,
        })

        # Merge and deduplicate by affected_agent — prefer smart options
        seen_agents = {o["strategy_id"] for o in smart_options}
        for opt in tool_options:
            if opt.get("strategy_id") not in seen_agents and opt.get("potential_savings", 0) > 0:
                smart_options.append(opt)
                seen_agents.add(opt.get("strategy_id", ""))

        # Sort by savings descending
        smart_options.sort(key=lambda x: x.get("potential_savings", 0), reverse=True)

        # ── Check if any option can cover the deficit ─────────────────────────
        max_single_savings  = max((o.get("potential_savings", 0) for o in smart_options), default=0)
        total_possible      = sum(o.get("potential_savings", 0) for o in smart_options)
        impossible          = total_possible < deficit * 0.5  # Can't cover even 50%

        if impossible:
            # Over budget by too much — be honest
            smart_options.insert(0, {
                "strategy_id":       "no_fix_possible",
                "category":          "none",
                "action":            f"Budget too low — deficit €{deficit:.0f} cannot be covered by reallocation",
                "potential_savings": 0,
                "impact_score":      0,
                "affected_agent":    None,
                "feasibility":       "none",
                "constraint":        None,
                "note":              (
                    f"The cheapest available options total €{total_spent:.0f}, "
                    f"which is €{deficit:.0f} over the €{state.total_budget:.0f} budget. "
                    f"Reallocation can save at most €{total_possible:.0f}. "
                    f"Consider increasing your budget or choosing a closer/cheaper destination."
                ),
            })

        tradeoffs = analyze_tradeoffs.invoke({
            "reallocation_options": [o for o in smart_options if o.get("strategy_id") != "no_fix_possible"],
            "deficit": deficit,
        })

        # ── Step 4: LLM picks strategy ────────────────────────────────────────
        skipped = []
        if skip_flights:    skipped.append("flights")
        if skip_hotels:     skipped.append("hotels")
        if skip_activities: skipped.append("activities")

        breakdown = overage.get("category_breakdown", {})
        breakdown_str = " | ".join(
            f"{cat}: €{amt:.0f}" for cat, amt in sorted(breakdown.items(), key=lambda x: -x[1])
        )

        input_msg = (
            f"Budget: €{state.total_budget:.0f} | Spent: €{total_spent:.0f} | "
            f"Over by: €{deficit:.0f}\n"
            f"Breakdown: {breakdown_str}\n"
            f"Nights: {nights}\n"
            f"User preferences: {state.user_preferences or 'None stated'}\n"
            + (f"Skipped (do NOT suggest): {', '.join(skipped)}\n" if skipped else "")
            + f"\nOptions (ranked by savings):\n"
            + "\n".join(
                f"  {i+1}. [{o.get('strategy_id')}] {o.get('action')} "
                f"→ saves €{o.get('potential_savings', 0):.0f} "
                f"(impact={o.get('impact_score', 0)}, feasibility={o.get('feasibility', '?')})"
                + (f"\n     constraint: {o.get('constraint')}" if o.get("constraint") else "")
                + (f"\n     note: {o.get('note')}" if o.get("note") else "")
                for i, o in enumerate(smart_options[:6])
            )
            + f"\n\nTrade-off analysis:\n{tradeoffs}"
        )

        result   = self.agent.invoke({"messages": [{"role": "user", "content": input_msg}]})
        decision: BudgetDecision = result["structured_response"]

        # Find savings amount from matching option
        savings_amount = next(
            (
                o.get("potential_savings")
                for o in smart_options
                if decision.affected_agent and
                o.get("affected_agent", "") == decision.affected_agent
            ),
            None,
        )

        response = BudgetAgentResponse(
            status="over_budget",
            total_budget=state.total_budget,
            total_spent=total_spent,
            difference=state.total_budget - total_spent,
            is_over_budget=True,
            primary_recommendation=decision.primary_recommendation,
            savings_amount=savings_amount,
            affected_agent=decision.affected_agent,
            constraint_to_apply=decision.constraint_to_apply,
            alternative_recommendations=decision.alternative_recommendations,
            explanation=decision.explanation,
        )

        logger.info(
            "Budget decision — rerun=%s | constraint=%s",
            response.affected_agent,
            response.constraint_to_apply,
        )
        logger.debug("Budget recommendation: %s", response.primary_recommendation)

        return response

    @log_agent_run("BudgetAgent")
    def manage_budget(self, state: TravelPlanState) -> BudgetAgentResponse:
        if state.status == "planning" and not state.budget_allocation.total:
            return self.allocate_initial_budget(
                total_budget=state.total_budget,
                duration_days=state.duration_days,
                currency=state.currency,
            )
        return self.check_budget_status(state)