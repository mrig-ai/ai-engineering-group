"""
Tools for Budget Manager Agent.
Provides budget allocation, analysis, and reallocation capabilities.
"""
from typing import List, Dict, Optional
from langchain.tools import tool

from src.models import (
    Expense,
    BudgetAllocation,
    ReallocationOption,
    BudgetReport,
    BudgetOverageAnalysis
)


# ============================================================================
# BUDGET ALLOCATION TOOL
# ============================================================================

@tool
def allocate_budget(
    total_budget: float,
    duration_days: int,
    user_preferences: str = "",
) -> Dict:
    """
    Allocates the total budget into categories based on trip duration and user preferences.

    Detects skipped categories from user_preferences (e.g. "driving my car" zeroes
    out flights, "staying at a friend's place" zeroes out hotels) and redistributes
    that budget proportionally across the remaining active categories.

    Base allocation strategy (adjusted by duration and preferences):
      - Flights:    40% — reduced/zeroed if user has own transport
      - Hotels:     35% — reduced/zeroed if user has own accommodation
      - Activities: 20% — reduced if "no activities" stated
      - Buffer:      5% — always reserved, never redistributed

    Preference adjustments applied on top of base:
      - "luxury" / "resort" / "beachfront" → hotels +5%, activities -5%
      - "budget" / "cheap"                 → activities +5%, hotels -5%
      - "hiking" / "adventure" / "culture" → activities +5%, hotels -5%
      - "direct" / "direct only"           → flights bumped to 50%

    Args:
        total_budget:     Total available budget for the trip.
        duration_days:    Number of trip days. Short trips (≤3) weight flights
                          higher; long trips (≥10) weight hotels and activities higher.
        user_preferences: Free-text preferences from the user. Used to detect
                          skipped categories and apply weighting adjustments.
                          E.g. "driving my car, luxury resort, hiking".

    Returns:
        Dict with keys:
          - allocation:         BudgetAllocation dict (flights, hotels, activities, buffer)
          - active:             {category: bool} — which categories are active
          - skipped:            List of zeroed-out category names
          - strategy:           Human-readable description of the allocation logic applied
          - flexibility_notes:  Per-category notes on how adjustable each line item is
    """
    prefs = user_preferences.lower()

    # ── Detect active categories ──────────────────────────────────────────
    needs_flight = not any(w in prefs for w in [
        "no flight", "no flights", "driving", "own car", "my car", "car trip", "car"
        "road trip", "by car", "taking the car", "train", "bus", "bike", "cycling", "bicycle",
    ])
    needs_hotel = not any(w in prefs for w in [
        "no hotel", "friend's place", "friends place", "staying with",
        "family house", "own place", "airbnb already", "accommodation sorted",
        "have accommodation", "free accommodation",
    ])
    needs_activities = not any(w in prefs for w in [
        "no activities", "no sightseeing", "just relaxing",
    ])

    active = {
        "flights":    needs_flight,
        "hotels":     needs_hotel,
        "activities": needs_activities,
    }

    # ── Base percentages (only for active categories) ─────────────────────
    BASE = {
        "flights":    0.40,
        "hotels":     0.35,
        "activities": 0.20,
        "buffer":     0.05,
    }

    # Zero out inactive categories
    for category, is_active in active.items():
        if not is_active:
            BASE[category] = 0.0

    # Redistribute zeroed budget proportionally across remaining active ones
    # (excluding buffer which stays fixed at 5%)
    active_categories = [c for c, is_active in active.items() if is_active]
    freed = sum(
        0.40 if c == "flights" else 0.35 if c == "hotels" else 0.20
        for c in active.keys() if not active[c]
    )

    if freed > 0 and active_categories:
        share = freed / len(active_categories)
        for c in active_categories:
            BASE[c] += share

    # ── Duration adjustments (only applied to active categories) ──────────
    if needs_flight:
        if duration_days <= 3:
            BASE["flights"] = min(BASE["flights"] + 0.05, 0.60)
        elif duration_days >= 10:
            BASE["flights"] = max(BASE["flights"] - 0.05, 0.20)

    # ── Preference adjustments ────────────────────────────────────────────
    if needs_hotel and any(w in prefs for w in ["luxury", "resort", "beachfront", "5-star"]):
        BASE["hotels"]     = min(BASE["hotels"] + 0.05, 0.60)
        if needs_activities:
            BASE["activities"] = max(BASE["activities"] - 0.05, 0.10)

    if any(w in prefs for w in ["budget", "cheap", "cheapest"]):
        if needs_activities:
            BASE["activities"] += 0.05
        if needs_hotel:
            BASE["hotels"]      = max(BASE["hotels"] - 0.05, 0.20)

    if any(w in prefs for w in ["hiking", "adventure", "culture", "museum"]):
        if needs_activities:
            BASE["activities"] += 0.05
        if needs_hotel:
            BASE["hotels"]      = max(BASE["hotels"] - 0.05, 0.20)

    # ── Normalise to ensure sum = 1.0 ─────────────────────────────────────
    total_pct = sum(BASE.values())
    if total_pct > 0:
        BASE = {k: round(v / total_pct, 4) for k, v in BASE.items()}

    allocation = BudgetAllocation(
        flights    = round(total_budget * BASE["flights"],    2) if needs_flight    else 0.0,
        hotels     = round(total_budget * BASE["hotels"],     2) if needs_hotel     else 0.0,
        activities = round(total_budget * BASE["activities"], 2) if needs_activities else 0.0,
        buffer     = round(total_budget * BASE["buffer"],     2),
    )

    # Build human-readable notes on what was skipped
    skipped = [c for c, is_active in active.items() if not is_active]
    notes = (
        f"Skipped categories (user-provided): {', '.join(skipped)}. "
        f"Budget redistributed across: {', '.join(active_categories)}."
        if skipped else f"Standard allocation for {duration_days}-day trip."
    )

    return {
        "allocation":   allocation.model_dump(),
        "active":       active,
        "skipped":      skipped,
        "strategy":     notes,
        "flexibility_notes": {
            "flights":    "Skipped — user has own transport" if not needs_flight    else "Low flexibility",
            "hotels":     "Skipped — user has own accommodation" if not needs_hotel else "High flexibility",
            "activities": "Skipped — no activities requested"  if not needs_activities else "High flexibility",
            "buffer":     "Emergency reserve — avoid using",
        },
    }


# ============================================================================
# COST CALCULATION TOOLS
# ============================================================================

@tool
def calculate_total_costs(expenses: List[Dict]) -> Dict:
    """
    Calculates the total of all current expenses.
    
    Args:
        expenses: List of expense dictionaries with 'category' and 'amount'
    
    Returns:
        Dictionary with total and per-category breakdown
    """
    # Convert dicts to Expense objects
    expense_objs = [Expense(**exp) for exp in expenses]
    
    total = sum(exp.amount for exp in expense_objs)
    
    # Calculate per-category totals
    by_category = {}
    for exp in expense_objs:
        by_category[exp.category] = by_category.get(exp.category, 0) + exp.amount
    
    return {
        "total_spent": round(total, 2),
        "by_category": {k: round(v, 2) for k, v in by_category.items()},
        "expense_count": len(expense_objs)
    }


@tool
def calculate_budget_overage(
    total_budget: float, 
    current_expenses: List[Dict] = []
) -> Dict:
    """
    Analyzes budget status and calculates overage with detailed breakdown.
    
    IMPORTANT: current_expenses must be a list of expense dictionaries.
    Each expense dict must have: category (str), amount (float), description (str)
    Valid categories: "flights", "hotels", "activities", "other"
    
    Example:
    current_expenses = [
        {"category": "flights", "amount": 850, "description": "Flight to Paris"},
        {"category": "hotels", "amount": 900, "description": "5 nights hotel"},
        {"category": "activities", "amount": 57, "description": "Museum tickets"}
    ]
    
    Args:
        total_budget: Total budget allocated
        current_expenses: List of expense dicts with 'category', 'amount', 'description'
    
    Returns:
        Detailed budget overage analysis
    """
    # Handle empty expenses
    if not current_expenses:
        return {
            "is_over_budget": False,
            "overage_amount": 0.0,
            "total_spent": 0.0,
            "total_budget": total_budget,
            "category_breakdown": {},
            "deficit_percentage": 0.0
        }
    
    # Convert to Expense objects with validation
    expense_objs = []
    for exp in current_expenses:
        # Normalize category to lowercase
        category = exp.get("category", "other").lower()
        # Map common variations to correct categories
        if category in ["flight", "flights"]:
            category = "flights"
        elif category in ["hotel", "hotels"]:
            category = "hotels"
        elif category in ["activity", "activities"]:
            category = "activities"
        
        # Ensure category is valid
        if category not in ["flights", "hotels", "activities", "other"]:
            category = "other"
        
        expense_objs.append(Expense(
            category=category,
            amount=exp.get("amount", 0),
            description=exp.get("description", "")
        ))
    
    total_spent = sum(exp.amount for exp in expense_objs)
    overage = total_spent - total_budget
    
    # Calculate per-category spending
    category_totals = {}
    for exp in expense_objs:
        category_totals[exp.category] = category_totals.get(exp.category, 0) + exp.amount
    
    analysis = BudgetOverageAnalysis(
        is_over_budget=overage > 0,
        overage_amount=max(0, overage),
        total_spent=total_spent,
        total_budget=total_budget,
        category_breakdown=category_totals,
        deficit_percentage=round((overage / total_budget * 100), 2) if overage > 0 else 0
    )
    
    return analysis.model_dump()


# ============================================================================
# REALLOCATION OPTIMIZER TOOL (CRITICAL) - SIMPLIFIED VERSION
# ============================================================================

@tool
def analyze_reallocation_options(
    deficit: float,
    current_flight_price: float = 0,
    current_flight_type: str = "unknown",
    cheapest_connecting_flight_price: float = 0,
    current_hotel_price: float = 0,
    current_hotel_tier: str = "unknown",
    cheaper_hotel_price: float = 0,
    cheaper_hotel_tier: str = "unknown",
    optional_activities_cost: float = 0,
    optional_activities_count: int = 0,
    recommended_activities_cost: float = 0,
    buffer_available: float = 0,
    trip_duration_days: int = 5
) -> List[Dict]:
    """
    Analyzes budget situation and generates concrete reallocation strategies.
    
    This is a SIMPLIFIED version that works with OpenAI's function calling.
    The agent should extract relevant numbers from the conversation and pass them here.
    
    Args:
        deficit: Amount over budget that needs to be saved
        current_flight_price: Price of currently selected flight
        current_flight_type: Type of current flight (direct, 1_stop, 2_stop)
        cheapest_connecting_flight_price: Price of cheapest connecting alternative
        current_hotel_price: Total price of current hotel
        current_hotel_tier: Tier of current hotel (e.g., "4-star")
        cheaper_hotel_price: Price of next tier down hotel
        cheaper_hotel_tier: Tier of cheaper hotel (e.g., "3-star")
        optional_activities_cost: Total cost of optional activities
        optional_activities_count: Number of optional activities
        recommended_activities_cost: Total cost of recommended activities
        buffer_available: Amount in buffer fund
        trip_duration_days: Number of days for the trip
    
    Returns:
        List of reallocation options ranked by savings/impact ratio
    """
    options = []
    
    # ========================================================================
    # STRATEGY 1: Downgrade Flight (Direct → Connecting)
    # ========================================================================
    if current_flight_type == "direct" and cheapest_connecting_flight_price > 0:
        savings = current_flight_price - cheapest_connecting_flight_price
        if savings > 0:
            options.append(ReallocationOption(
                strategy_id="flight_connecting",
                category="flights",
                action=f"Switch from direct to connecting flight",
                potential_savings=round(savings, 2),
                impact_score=2,  # Low impact - just adds travel time
                affected_agent="flight",
                feasibility="high",
                details=f"Saves ${savings:.0f}. Adds travel time but maintains same destination."
            ))
    
    # ========================================================================
    # STRATEGY 2: Downgrade Hotel Tier
    # ========================================================================
    if cheaper_hotel_price > 0 and cheaper_hotel_price < current_hotel_price:
        savings = current_hotel_price - cheaper_hotel_price
        
        # Determine impact based on tier drop
        impact = 4  # Default medium impact
        if "luxury" in current_hotel_tier.lower() or "5-star" in current_hotel_tier.lower():
            impact = 3  # Lower impact when downgrading from luxury
        
        options.append(ReallocationOption(
            strategy_id="hotel_downgrade",
            category="hotels",
            action=f"Downgrade from {current_hotel_tier} to {cheaper_hotel_tier} hotel",
            potential_savings=round(savings, 2),
            impact_score=impact,
            affected_agent="hotel",
            feasibility="high",
            details=f"Saves ${savings:.0f}. Still good accommodation, just less luxurious."
        ))
    
    # ========================================================================
    # STRATEGY 3: Reduce Optional Activities
    # ========================================================================
    if optional_activities_cost > 0 and optional_activities_count > 0:
        # Calculate how much we need vs what's available
        potential_savings = min(deficit, optional_activities_cost)
        
        options.append(ReallocationOption(
            strategy_id="reduce_optional_activities",
            category="activities",
            action=f"Remove optional activities",
            potential_savings=round(potential_savings, 2),
            impact_score=5,  # Medium impact
            affected_agent="activities",
            feasibility="high",
            details=f"Remove some/all optional activities. Saves up to ${optional_activities_cost:.0f}. Focus on must-sees."
        ))
    
    # ========================================================================
    # STRATEGY 4: Use Buffer (If Available)
    # ========================================================================
    if buffer_available > 0:
        savings = min(deficit, buffer_available)
        options.append(ReallocationOption(
            strategy_id="use_buffer",
            category="buffer",
            action=f"Use emergency buffer fund",
            potential_savings=round(savings, 2),
            impact_score=1,  # No impact on trip quality
            affected_agent="none",
            feasibility="high",
            details=f"Covers ${savings:.0f} from buffer. No changes to trip needed."
        ))
    
    # ========================================================================
    # STRATEGY 5: Reduce Recommended Activities (More Aggressive)
    # ========================================================================
    if recommended_activities_cost > 0:
        potential_savings = min(deficit, recommended_activities_cost)
        
        options.append(ReallocationOption(
            strategy_id="reduce_recommended_activities",
            category="activities",
            action=f"Also reduce recommended activities",
            potential_savings=round(potential_savings, 2),
            impact_score=7,  # Higher impact
            affected_agent="activities",
            feasibility="medium",
            details=f"More aggressive cuts. Saves up to ${recommended_activities_cost:.0f}."
        ))
    
    # ========================================================================
    # STRATEGY 6: Shorten Trip (Last Resort)
    # ========================================================================
    # if trip_duration_days > 3:
    #     # Rough estimate: hotel + activities per day
    #     daily_cost = (current_hotel_price / trip_duration_days) + 50  # Assume $50/day activities
    #     estimated_savings = daily_cost
        
    #     options.append(ReallocationOption(
    #         strategy_id="shorten_trip",
    #         category="itinerary",
    #         action=f"Reduce trip from {trip_duration_days} to {trip_duration_days - 1} days",
    #         potential_savings=round(estimated_savings, 2),
    #         impact_score=9,  # Very high impact
    #         affected_agent="itinerary_optimizer",
    #         feasibility="low",
    #         details=f"Last resort. Lose entire day. Est. saves ${estimated_savings:.0f}/day."
    #     ))
    
    # ========================================================================
    # Sort by savings-to-impact ratio (best value first)
    # ========================================================================
    options.sort(
        key=lambda x: x.potential_savings / max(x.impact_score, 1), 
        reverse=True
    )
    
    # Convert to dicts for JSON serialization
    return [opt.model_dump() for opt in options]


# ============================================================================
# BUDGET REPORT GENERATION
# ============================================================================

@tool
def generate_budget_status_report(
    total_budget: float,
    expenses: List[Dict] = [],
    reallocation_options: List[Dict] = []
) -> Dict:
    """
    Generates a comprehensive budget status report.
    
    IMPORTANT: expenses must be a list of expense dictionaries.
    Each expense dict must have: category (str), amount (float), description (str)
    Valid categories: "flights", "hotels", "activities", "other"
    
    Example call:
    expenses = [
        {"category": "flights", "amount": 850, "description": "Flight"},
        {"category": "hotels", "amount": 900, "description": "Hotel"},
        {"category": "activities", "amount": 57, "description": "Activities"}
    ]
    reallocation_options = []  # Empty if within budget, or list from analyze_reallocation_options
    
    Args:
        total_budget: Total budget allocated
        expenses: List of expense dicts with 'category', 'amount', 'description'
        reallocation_options: Optional list of reallocation strategies (from analyze_reallocation_options)
    
    Returns:
        Structured budget report
    """
    # Handle empty expenses
    if not expenses:
        return {
            "total_budget": total_budget,
            "total_spent": 0.0,
            "difference": total_budget,
            "is_over_budget": False,
            "category_spending": {},
            "reallocation_options": [],
            "recommended_strategy": None,
            "summary": f"No expenses recorded yet. Full budget of ${total_budget:.2f} available."
        }
    
    # Convert to objects with category normalization
    expense_objs = []
    for exp in expenses:
        # Normalize category to lowercase
        category = exp.get("category", "other").lower()
        # Map common variations
        if category in ["flight", "flights"]:
            category = "flights"
        elif category in ["hotel", "hotels"]:
            category = "hotels"
        elif category in ["activity", "activities"]:
            category = "activities"
        
        # Ensure valid category
        if category not in ["flights", "hotels", "activities", "other"]:
            category = "other"
        
        expense_objs.append(Expense(
            category=category,
            amount=exp.get("amount", 0),
            description=exp.get("description", "")
        ))
    
    realloc_objs = [
        ReallocationOption(**opt) for opt in (reallocation_options or [])
    ]
    
    total_spent = sum(exp.amount for exp in expense_objs)
    difference = total_budget - total_spent
    is_over = total_spent > total_budget
    
    # Category breakdown
    category_spending = {}
    for exp in expense_objs:
        category_spending[exp.category] = category_spending.get(exp.category, 0) + exp.amount
    
    # Determine recommendation
    recommended_strategy = None
    if realloc_objs:
        # Recommend the best option (highest savings/impact ratio)
        best_option = realloc_objs[0] if realloc_objs else None
        if best_option:
            recommended_strategy = best_option.strategy_id
    
    # Generate summary
    if is_over:
        summary = f"Over budget by ${abs(difference):.2f}. "
        if realloc_objs:
            summary += f"Recommended: {realloc_objs[0].action}. "
        summary += f"Total spent: ${total_spent:.2f} of ${total_budget:.2f} budget."
    else:
        summary = f"Within budget. ${difference:.2f} remaining. "
        summary += f"Total spent: ${total_spent:.2f} of ${total_budget:.2f} budget."
    
    report = BudgetReport(
        total_budget=total_budget,
        total_spent=round(total_spent, 2),
        difference=round(difference, 2),
        is_over_budget=is_over,
        category_spending={k: round(v, 2) for k, v in category_spending.items()},
        reallocation_options=realloc_objs,
        recommended_strategy=recommended_strategy,
        summary=summary
    )
    
    return report.model_dump()


# ============================================================================
# TRADE-OFF ANALYZER (BONUS TOOL)
# ============================================================================

@tool
def analyze_tradeoffs(
    reallocation_options: List[Dict],
    deficit: float
) -> Dict:
    """
    Analyzes trade-offs between different reallocation strategies.
    Helps choose the best combination of strategies.
    
    Args:
        reallocation_options: List of available reallocation options
        deficit: Amount that needs to be saved
    
    Returns:
        Analysis of single and combined strategies
    """
    options = [ReallocationOption(**opt) for opt in reallocation_options]
    
    if not options:
        return {
            "analysis": "No reallocation options available",
            "recommendations": []
        }
    
    # Single strategy analysis
    single_strategies = []
    for opt in options:
        sufficient = opt.potential_savings >= deficit
        single_strategies.append({
            "strategy": opt.strategy_id,
            "action": opt.action,
            "savings": opt.potential_savings,
            "impact": opt.impact_score,
            "sufficient": sufficient,
            "efficiency": round(opt.potential_savings / opt.impact_score, 2)
        })
    
    # Find combinations (try 2-strategy combinations)
    combinations = []
    for i, opt1 in enumerate(options):
        for opt2 in options[i+1:]:
            combined_savings = opt1.potential_savings + opt2.potential_savings
            combined_impact = (opt1.impact_score + opt2.impact_score) / 2
            
            if combined_savings >= deficit:
                combinations.append({
                    "strategies": [opt1.strategy_id, opt2.strategy_id],
                    "actions": [opt1.action, opt2.action],
                    "combined_savings": round(combined_savings, 2),
                    "combined_impact": round(combined_impact, 2),
                    "efficiency": round(combined_savings / combined_impact, 2)
                })
    
    # Sort by efficiency
    single_strategies.sort(key=lambda x: x["efficiency"], reverse=True)
    combinations.sort(key=lambda x: x["efficiency"], reverse=True)
    
    return {
        "deficit_to_cover": deficit,
        "single_strategy_options": single_strategies[:3],  # Top 3
        "combination_options": combinations[:3],  # Top 3
        "recommendation": single_strategies[0] if single_strategies and single_strategies[0]["sufficient"] else combinations[0] if combinations else None
    }