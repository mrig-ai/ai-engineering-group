"""
app.py
------
Streamlit UI for the Travel Planning Chatbot.
"""

import json
import uuid
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="AI Travel Planner",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Sans:ital,wght@0,300;0,400;0,500;1,400&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }

.main .block-container { max-width: 860px; padding: 1rem 2rem 5rem; }

/* Header */
.app-header { text-align: center; padding: 1.8rem 0 1.2rem; }
.app-header h1 {
    font-family: 'Syne', sans-serif; font-weight: 800; font-size: 2.6rem;
    letter-spacing: -1px; margin: 0 0 0.3rem; line-height: 1.1;
    background: linear-gradient(135deg, #0ea5e9 0%, #6366f1 55%, #ec4899 100%);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.app-header p { color: #94a3b8; font-size: 0.95rem; font-weight: 300; margin: 0; }

/* User bubble */
.msg-user {
    background: linear-gradient(135deg, #0ea5e9, #6366f1); color: #fff !important;
    border-radius: 18px 18px 4px 18px; padding: 11px 16px;
    margin: 8px 0 8px auto; max-width: 78%;
    font-size: 0.93rem; line-height: 1.5; word-break: break-word;
}

/* Assistant bubble */
.msg-assistant {
    background: #f8fafc; color: #1e293b !important;
    border-radius: 4px 18px 18px 18px; padding: 14px 18px;
    margin: 8px auto 8px 0; max-width: 88%;
    font-size: 0.93rem; line-height: 1.65; border: 1px solid #e2e8f0;
}
.msg-assistant strong { color: #0f172a; }
.msg-assistant em     { color: #475569; }
.msg-assistant h1, .msg-assistant h2, .msg-assistant h3 {
    font-family: 'Syne', sans-serif; color: #0f172a;
    margin-top: 0.8rem; margin-bottom: 0.3rem;
}

/* Progress block */
.prog-block {
    background: #f8fafc; border: 1px solid #e2e8f0; border-radius: 12px;
    padding: 12px 16px; margin: 8px auto 8px 0; max-width: 88%;
}
.prog-step { display: flex; align-items: flex-start; gap: 10px; padding: 4px 0; font-size: 0.87rem; color: #475569; }
.prog-dot  { width: 7px; height: 7px; min-width: 7px; background: #0ea5e9; border-radius: 50%; margin-top: 5px; }

/* HITL card */
.hitl-card {
    background: linear-gradient(135deg, #fffbeb, #fef9c3);
    border: 1.5px solid #f59e0b; border-radius: 14px;
    padding: 18px 20px; margin: 8px auto 8px 0; max-width: 88%;
}
.hitl-title { font-family: 'Syne', sans-serif; font-weight: 700; font-size: 1rem; color: #92400e; margin-bottom: 4px; }
.hitl-amount { font-size: 1.7rem; font-weight: 800; color: #b45309; font-family: 'Syne', sans-serif; margin-bottom: 12px; }
.hitl-rec {
    background: #fff; border-left: 3px solid #f59e0b; border-radius: 6px;
    padding: 9px 13px; font-size: 0.88rem; color: #374151; margin-bottom: 8px;
}
.hitl-alts  { font-size: 0.81rem; color: #78716c; margin-top: 6px; line-height: 1.6; }
.hitl-expl  { font-size: 0.82rem; color: #6b7280; margin-top: 8px; font-style: italic; }

/* Welcome */
.welcome-wrap { text-align: center; padding: 2.5rem 0 1rem; }
.welcome-icon { font-size: 3.5rem; margin-bottom: 0.8rem; }
.welcome-h2   { font-family: 'Syne', sans-serif; font-size: 1.15rem; font-weight: 600; color: #334155; margin-bottom: 0.3rem; }
.welcome-sub  { font-size: 0.9rem; color: #94a3b8; }

/* Sidebar — scoped, NO wildcard * override */
[data-testid="stSidebar"] { background-color: #0f172a; }

.sb-title {
    font-family: 'Syne', sans-serif; font-size: 1.2rem; font-weight: 800;
    background: linear-gradient(135deg, #0ea5e9, #6366f1);
    -webkit-background-clip: text; -webkit-text-fill-color: transparent; background-clip: text;
}
.sb-sub   { font-size: 0.73rem; color: #64748b; margin-top: 2px; margin-bottom: 0.8rem; }
.sb-label { font-size: 0.68rem; text-transform: uppercase; letter-spacing: 0.1em; color: #64748b; font-weight: 600; margin: 0.8rem 0 0.4rem; }
.sb-tips  { font-size: 0.78rem; color: #64748b; line-height: 1.65; margin-top: 1rem; }
.sb-tips strong { color: #94a3b8; }

/* Trip card — explicit colors, not inherited */
.trip-card { background: #1e293b; border: 1px solid #334155; border-radius: 10px; padding: 12px 14px; margin: 4px 0 8px; }
.trip-row  { display: flex; justify-content: space-between; padding: 3px 0; border-bottom: 1px solid #273548; }
.trip-row:last-child { border-bottom: none; }
.trip-key  { font-size: 0.72rem; color: #64748b !important; text-transform: uppercase; letter-spacing: 0.06em; font-weight: 600; }
.trip-val  { font-size: 0.82rem; color: #e2e8f0 !important; font-weight: 500; text-align: right; max-width: 60%; word-break: break-word; }

.badge { display: inline-block; padding: 2px 9px; border-radius: 20px; font-size: 0.72rem; font-weight: 600; margin-bottom: 8px; }
.badge-complete { background: #dcfce7; color: #15803d; }
.badge-planning { background: #eff6ff; color: #1d4ed8; }
.badge-over     { background: #fef3c7; color: #b45309; }

.hist-item { background: #1e293b; border: 1px solid #334155; border-radius: 8px; padding: 8px 12px; margin-bottom: 5px; }
.hist-dest { font-size: 0.87rem; font-weight: 600; color: #e2e8f0 !important; }
.hist-meta { font-size: 0.74rem; color: #64748b !important; margin-top: 1px; }
</style>
""", unsafe_allow_html=True)


# ── Orchestrator ──────────────────────────────────────────────────────────────
@st.cache_resource
def get_orchestrator():
    from src.orchestrator import TravelOrchestrator
    return TravelOrchestrator()


# ── Session state ─────────────────────────────────────────────────────────────
_DEFAULTS = {
    "thread_id":     None,
    "history":       [],
    "awaiting_hitl": False,
    "past_trips":    [],
    "pending_input": None,   # str — process on next rerun
    "pending_hitl":  None,   # "approve" | "reject"
}
for _k, _v in _DEFAULTS.items():
    if _k not in st.session_state:
        st.session_state[_k] = _v

if not st.session_state.thread_id:
    st.session_state.thread_id = str(uuid.uuid4())


# ── Utilities ─────────────────────────────────────────────────────────────────

def _fmt(price):
    if not price:
        return "N/A"
    return f"€{price:.0f}"


def _add(role, content, msg_type="text"):
    st.session_state.history.append({"role": role, "content": content, "type": msg_type})


def _save_trip(orc):
    tid   = st.session_state.thread_id
    state = orc.get_state(tid)
    if not state:
        return
    entry = {
        "thread_id":   tid,
        "destination": state.destination or "Unknown",
        "budget":      _fmt(state.total_budget),
        "duration":    f"{state.duration_days}d" if state.duration_days else "",
        "status":      state.status,
    }
    existing = [t for t in st.session_state.past_trips if t["thread_id"] == tid]
    if existing:
        idx = st.session_state.past_trips.index(existing[0])
        st.session_state.past_trips[idx] = entry
    else:
        st.session_state.past_trips.insert(0, entry)


def _render_hitl(data: dict):
    over  = data.get("over_budget_by", 0)
    prim  = data.get("primary_recommendation", "")
    const = data.get("constraint_to_apply", "")
    agent = data.get("affected_agent", "")
    alts  = data.get("alternatives", [])
    expl  = data.get("explanation", "")

    alts_html = (
        '<div class="hitl-alts">Alternatives:<br>' +
        "<br>".join(f"• {a}" for a in alts[:2]) + "</div>"
    ) if alts else ""
    expl_html = f'<div class="hitl-expl">{expl}</div>' if expl else ""

    st.markdown(f"""
    <div class="hitl-card">
        <div class="hitl-title">⚠️ Over Budget</div>
        <div class="hitl-amount">{_fmt(over)} over limit</div>
        <div class="hitl-rec">
            <strong>💡 {prim}</strong><br>
            <span style="font-size:0.81rem;color:#6b7280">
                Affects: {agent} &nbsp;·&nbsp; {const}
            </span>
        </div>
        {alts_html}{expl_html}
    </div>
    """, unsafe_allow_html=True)


def _render_state_card(orc):
    tid   = st.session_state.thread_id
    state = orc.get_state(tid)
    if not state or not state.destination:
        return

    bcls = {"complete": "badge-complete", "over_budget": "badge-over"}.get(state.status, "badge-planning")
    blbl = {"complete": "✓ Complete", "over_budget": "⚠ Over Budget", "planning": "● Planning",
            "searching": "● Searching", "within_budget": "● Within Budget"}.get(state.status, state.status)

    rows = []
    if state.destination: rows.append(("Destination", state.destination))
    if state.origin:      rows.append(("From", state.origin))
    if state.check_in_date and state.check_out_date:
        rows.append(("Dates", f"{state.check_in_date} → {state.check_out_date}"))
    if state.total_budget:
        spent = f" · spent {_fmt(state.total_spent)}" if state.total_spent else ""
        rows.append(("Budget", f"{_fmt(state.total_budget)}{spent}"))
    if state.flight_response and state.flight_response.recommended_flight:
        rf = state.flight_response.recommended_flight
        # Use state.flight_cost — returns confirmed round-trip total or one-way price
        cost_s = _fmt(state.flight_cost)
        is_rt  = state.flight_response.recommended_return_flight is not None
        rows.append(("Flight", f"{rf.airline} {cost_s}{' RT' if is_rt else ''}"))
    if state.hotel_response and state.hotel_response.recommended_hotel:
        h = state.hotel_response.recommended_hotel
        rows.append(("Hotel", f"{h.name[:22]}… {_fmt(h.price_per_night)}/n"))

    rows_html = "".join(
        f'<div class="trip-row">'
        f'<span class="trip-key">{k}</span>'
        f'<span class="trip-val">{v}</span>'
        f'</div>'
        for k, v in rows
    )
    st.markdown(
        f'<div class="trip-card">'
        f'<span class="badge {bcls}">{blbl}</span>'
        f'{rows_html}</div>',
        unsafe_allow_html=True,
    )


_ACT_ICON = {
    "flight":      "✈️",
    "hotel":       "🏨",
    "attraction":  "🗺️",
    "travel":      "🚌",
    "free_time":   "☕",
}

# ── Shared card CSS (injected into each component) ────────────────────────────
_CARD_BASE_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
*{box-sizing:border-box;margin:0;padding:0;font-family:'DM Sans',sans-serif;}
body{background:transparent;padding:4px 0;}
.card-wrap{border-radius:14px;overflow:hidden;box-shadow:0 2px 16px rgba(0,0,0,0.08);}
.card-header{background:linear-gradient(135deg,#0f172a,#1e293b);padding:18px 22px 14px;}
.card-title{font-family:'Syne',sans-serif;font-weight:800;font-size:1.1rem;color:#f1f5f9;margin-bottom:6px;}
.card-sub{font-size:0.82rem;color:#94a3b8;line-height:1.5;}
.card-body{border:1px solid #e2e8f0;border-top:none;background:white;}
.card-reasoning{background:#fffbeb;border-top:1px solid #fde68a;padding:14px 20px;
    font-size:0.83rem;color:#78350f;line-height:1.6;}
.reasoning-label{font-family:'Syne',sans-serif;font-weight:700;font-size:0.75rem;
    color:#b45309;text-transform:uppercase;letter-spacing:0.06em;margin-bottom:5px;}

/* ── Shared pills / tags ── */
.pill{display:inline-block;border-radius:10px;padding:2px 9px;font-size:0.75rem;font-weight:600;white-space:nowrap;}
.pill-blue{background:#e0f2fe;color:#0369a1;}
.pill-green{background:#dcfce7;color:#15803d;}
.pill-amber{background:#fef3c7;color:#b45309;}
.pill-red{background:#fee2e2;color:#dc2626;}
.pill-grey{background:#f1f5f9;color:#475569;}
.pill-purple{background:#ede9fe;color:#6d28d9;}
.tag{background:#f8fafc;border:1px solid #e2e8f0;border-radius:6px;padding:2px 8px;
    font-size:0.76rem;color:#64748b;white-space:nowrap;}

/* ── Section divider ── */
.section-label{font-family:'Syne',sans-serif;font-weight:700;font-size:0.76rem;
    color:#94a3b8;text-transform:uppercase;letter-spacing:0.07em;
    padding:10px 18px 4px;background:#f8fafc;border-top:1px solid #e2e8f0;
    border-bottom:1px solid #f1f5f9;}

/* ── FLIGHTS ── */
.flight-card{padding:16px 18px;border-bottom:1px solid #f1f5f9;}
.flight-card:last-child{border-bottom:none;}
.flight-top{display:flex;align-items:center;gap:8px;margin-bottom:10px;flex-wrap:wrap;}
.flight-airline{font-family:'Syne',sans-serif;font-weight:700;font-size:0.98rem;color:#1e293b;flex:1;min-width:120px;}
.flight-price{font-family:'Syne',sans-serif;font-weight:800;font-size:1.1rem;color:#0ea5e9;}
.flight-route{display:flex;align-items:center;gap:10px;margin-bottom:8px;}
.airport-code{font-family:'Syne',sans-serif;font-weight:800;font-size:1.1rem;color:#1e293b;}
.airport-time{font-size:0.78rem;color:#64748b;margin-top:1px;}
.route-mid{flex:1;display:flex;flex-direction:column;align-items:center;gap:2px;}
.route-line{height:1px;width:100%;background:#e2e8f0;position:relative;}
.route-arrow-icon{font-size:0.7rem;color:#94a3b8;}
.route-dur{font-size:0.72rem;color:#94a3b8;}
.flight-tags{display:flex;flex-wrap:wrap;gap:6px;margin-bottom:8px;}
.leg-block{background:#f8fafc;border-radius:8px;padding:10px 12px;margin-top:8px;border:1px solid #f1f5f9;}
.leg-header{font-size:0.77rem;font-weight:600;color:#475569;margin-bottom:4px;}
.leg-row{display:flex;align-items:center;gap:6px;font-size:0.78rem;color:#64748b;}
.layover-block{background:#fef3c7;border-radius:6px;padding:6px 10px;margin:4px 0;
    font-size:0.77rem;color:#92400e;display:flex;gap:6px;align-items:center;}
.amenities-row{display:flex;flex-wrap:wrap;gap:5px;margin-top:6px;}
.carbon{font-size:0.73rem;color:#64748b;margin-top:5px;}

/* ── HOTELS ── */
.hotel-card{padding:16px 18px;border-bottom:1px solid #f1f5f9;}
.hotel-card:last-child{border-bottom:none;}
.hotel-name-row{display:flex;align-items:flex-start;gap:10px;margin-bottom:4px;}
.hotel-name{font-family:'Syne',sans-serif;font-weight:700;font-size:0.98rem;color:#1e293b;flex:1;line-height:1.3;}
.hotel-price-block{text-align:right;flex-shrink:0;}
.hotel-ppn{font-family:'Syne',sans-serif;font-weight:800;font-size:1.05rem;color:#0ea5e9;}
.hotel-total{font-size:0.74rem;color:#94a3b8;margin-top:1px;}
.hotel-rating-row{display:flex;align-items:center;gap:8px;margin-bottom:8px;flex-wrap:wrap;}
.stars{color:#f59e0b;}
.hotel-tags{display:flex;flex-wrap:wrap;gap:5px;margin-bottom:8px;}
.hotel-desc{font-size:0.81rem;color:#64748b;line-height:1.5;margin-bottom:8px;}
.amenities-section{margin-bottom:8px;}
.amenities-title{font-size:0.73rem;font-weight:600;color:#475569;margin-bottom:5px;text-transform:uppercase;letter-spacing:0.04em;}
.nearby-block{margin-top:8px;}
.nearby-item{display:flex;align-items:center;gap:8px;font-size:0.78rem;color:#64748b;padding:3px 0;border-bottom:1px solid #f8fafc;}
.nearby-item:last-child{border-bottom:none;}
.nearby-name{flex:1;font-weight:500;color:#334155;}
.cancel-block{background:#f0fdf4;border-radius:8px;padding:8px 12px;margin-top:8px;font-size:0.79rem;color:#15803d;}
.review-breakdown{margin-top:8px;}
.rb-row{display:flex;align-items:center;gap:8px;padding:3px 0;font-size:0.78rem;}
.rb-label{width:90px;color:#475569;flex-shrink:0;}
.rb-bar-wrap{flex:1;background:#f1f5f9;border-radius:4px;height:6px;overflow:hidden;}
.rb-bar{height:6px;background:linear-gradient(90deg,#0ea5e9,#6366f1);border-radius:4px;}
.rb-pct{font-size:0.72rem;color:#94a3b8;width:32px;text-align:right;}

/* ── ATTRACTIONS ── */
.attr-grid{display:grid;grid-template-columns:1fr 1fr;gap:0;}
.attr-card{padding:13px 15px;border-bottom:1px solid #f1f5f9;border-right:1px solid #f1f5f9;}
.attr-card:nth-child(even){border-right:none;}
.attr-card:nth-last-child(-n+2){border-bottom:none;}
.attr-cat{font-size:0.71rem;color:#0ea5e9;font-weight:700;text-transform:uppercase;letter-spacing:0.05em;margin-bottom:3px;}
.attr-name{font-weight:600;font-size:0.88rem;color:#1e293b;margin-bottom:4px;line-height:1.3;}
.attr-desc{font-size:0.76rem;color:#64748b;margin-bottom:5px;line-height:1.35;}
.attr-meta{display:flex;flex-wrap:wrap;align-items:center;gap:5px;font-size:0.76rem;color:#64748b;}
.attr-stars{color:#f59e0b;font-size:0.8rem;}
</style>
"""


def _flight_card_html(f: dict, label: str = "recommended") -> str:
    """Build comprehensive HTML for one flight option — shows all available fields."""
    airline   = f.get("airline", "Unknown")
    fnum      = f.get("flight_number", "") or ""
    dep_code  = f.get("departure_airport_code", "")
    arr_code  = f.get("arrival_airport_code", "")
    dep_name  = f.get("departure_airport_name", "")
    arr_name  = f.get("arrival_airport_name", "")
    dep_time  = f.get("departure_time", "")
    arr_time  = f.get("arrival_time", "")
    price     = f.get("price", 0)
    ftype     = f.get("type", "")
    duration  = f.get("total_duration_hours", 0)
    legroom   = f.get("legroom", "") or ""
    amenities = f.get("amenities", []) or []
    airplane  = f.get("airplane", "") or ""
    tclass    = f.get("travel_class", "") or ""
    carbon    = f.get("carbon_emissions") or ""
    legs      = f.get("legs", []) or []
    layovers  = f.get("layovers", []) or []
    sold_by   = f.get("ticket_also_sold_by", []) or []

    # Badges
    rec_badge  = '<span class="pill pill-blue">★ Best pick</span>' if label == "recommended" else '<span class="pill pill-grey">Alternative</span>'
    type_badge = '<span class="pill pill-green">Direct</span>' if ftype == "direct" else f'<span class="pill pill-amber">{ftype.replace("_"," ").title()}</span>'
    price_s    = f"€{price:.0f}" if price else "N/A"
    dur_s      = f"~{duration:.1f}h total" if duration else ""

    # Route block
    dep_name_s = f'<div class="airport-time">{dep_name[:30]}</div>' if dep_name else ""
    arr_name_s = f'<div class="airport-time">{arr_name[:30]}</div>' if arr_name else ""
    dep_t      = f'<div class="airport-time">{dep_time}</div>' if dep_time else ""
    arr_t      = f'<div class="airport-time">{arr_time}</div>' if arr_time else ""

    route_html = (
        f'<div class="flight-route">'
        f'<div><div class="airport-code">{dep_code}</div>{dep_name_s}{dep_t}</div>'
        f'<div class="route-mid">'
        f'<div class="route-arrow-icon">✈</div>'
        f'<div class="route-line"></div>'
        f'<div class="route-dur">{dur_s}</div>'
        f'</div>'
        f'<div style="text-align:right"><div class="airport-code">{arr_code}</div>{arr_name_s}{arr_t}</div>'
        f'</div>'
    )

    # Tags row
    tags = []
    if airplane:     tags.append(airplane)
    if tclass:       tags.append(tclass)
    if legroom:      tags.append(f"Legroom {legroom}")
    tags_html = '<div class="flight-tags">' + "".join(f'<span class="tag">{t}</span>' for t in tags) + "</div>" if tags else ""

    # Amenities
    amenity_html = ""
    if amenities:
        pills = "".join(f'<span class="pill pill-grey">{a}</span>' for a in amenities)
        amenity_html = f'<div class="amenities-row">{pills}</div>'

    # Legs breakdown (for multi-stop)
    legs_html = ""
    if len(legs) > 1:
        legs_html = "<div style='margin-top:8px'>"
        for i, leg in enumerate(legs):
            ld = leg.get("departure_airport_code", "")
            la = leg.get("arrival_airport_code", "")
            lt = leg.get("departure_time", "")
            lat = leg.get("arrival_time", "")
            lnum = leg.get("flight_number", "")
            ldur = leg.get("duration_minutes", 0)
            ldur_s = f"{ldur // 60}h {ldur % 60}m" if ldur else ""
            legs_html += (
                f'<div class="leg-block">'
                f'<div class="leg-header">Leg {i+1} · {leg.get("airline","")} {lnum}</div>'
                f'<div class="leg-row"><strong>{ld}</strong> {lt} → <strong>{la}</strong> {lat} · {ldur_s}</div>'
                f'</div>'
            )
            # Layover after this leg (except last)
            if i < len(layovers):
                lay = layovers[i]
                laydur = lay.get("duration_minutes", 0)
                laydur_s = f"{laydur // 60}h {laydur % 60}m" if laydur else ""
                legs_html += (
                    f'<div class="layover-block">⏱ Layover at {lay.get("airport_code","")} '
                    f'{lay.get("airport_name","")} · {laydur_s}</div>'
                )
        legs_html += "</div>"

    # Carbon + sold by
    extra = ""
    if carbon:
        extra += f'<div class="carbon">🌱 Carbon emissions: {carbon}</div>'
    if sold_by:
        extra += f'<div class="carbon">Also sold by: {", ".join(sold_by)}</div>'

    return (
        f'<div class="flight-card">'
        f'<div class="flight-top">{rec_badge}{type_badge}'
        f'<span class="flight-airline">{airline} {fnum}</span>'
        f'<span class="flight-price">{price_s}</span>'
        f'</div>'
        f'{route_html}'
        f'{tags_html}'
        f'{amenity_html}'
        f'{legs_html}'
        f'{extra}'
        f'</div>'
    )


def _render_flights(payload: dict):
    import streamlit.components.v1 as components

    rec     = payload.get("recommended")
    ret_flt = payload.get("return_flight")
    alts    = payload.get("alternatives", []) or []
    reason  = payload.get("reasoning", "")

    if not rec:
        st.markdown('<div class="msg-assistant">No flights found.</div>', unsafe_allow_html=True)
        return

    dep = rec.get("departure_airport_code", "")
    arr = rec.get("arrival_airport_code", "")
    price = rec.get("price", 0)
    sub = f"{dep} → {arr}"
    if price: sub += f" · Best from €{price:.0f}"
    if ret_flt: sub += " · Round-trip"

    # Recommended + all alternatives
    cards_html = (
        '<div class="section-label">Recommended</div>'
        + _flight_card_html(rec, "recommended")
    )
    if alts:
        cards_html += '<div class="section-label">Other options</div>'
        for alt in alts:
            cards_html += _flight_card_html(alt, "alternative")

    # Return flight
    ret_html = ""
    if ret_flt:
        ret_html = (
            '<div class="section-label" style="color:#6366f1;border-top:2px solid #e0e7ff">Return Flight</div>'
            + _flight_card_html(ret_flt, "recommended")
        )

    reason_html = (
        f'<div class="card-reasoning">'
        f'<div class="reasoning-label">💡 Why this flight</div>'
        f'{reason}'
        f'</div>'
    ) if reason else ""

    html = (
        _CARD_BASE_CSS
        + '<div class="card-wrap">'
        + '<div class="card-header">'
        + '<div class="card-title">✈️ Flight Results</div>'
        + f'<div class="card-sub">{sub}</div>'
        + '</div>'
        + f'<div class="card-body">{cards_html}{ret_html}</div>'
        + reason_html
        + '</div>'
    )

    # Height: header + sections + each card (~160px rec, ~130px alt) + return + reasoning
    n_legs_rec = len(rec.get("legs", []))
    h = 180 + 160 + max(n_legs_rec - 1, 0) * 70  # extra for multi-leg breakdown
    for alt in alts:
        h += 130 + max(len(alt.get("legs", [])) - 1, 0) * 50
    if ret_flt: h += 160
    if reason:  h += 80
    components.html(html, height=max(h, 350), scrolling=True)


def _render_hotels(payload: dict):
    import streamlit.components.v1 as components

    rec    = payload.get("recommended")
    alts   = payload.get("alternatives", []) or []
    reason = payload.get("reasoning", "")
    pr     = payload.get("price_range") or {}
    rr     = payload.get("rating_range") or {}

    if not rec:
        st.markdown('<div class="msg-assistant">No hotels found.</div>', unsafe_allow_html=True)
        return

    def _hotel_card(h: dict, label: str) -> str:
        name     = h.get("name", "Unknown")
        ppn      = h.get("price_per_night", 0) or 0
        total    = h.get("total_price", 0) or 0
        rating   = h.get("overall_rating") or 0
        loc_rat  = h.get("location_rating") or 0
        reviews  = h.get("total_reviews", 0) or 0
        htype    = (h.get("type") or "hotel").replace("_", " ").title()
        location = h.get("location", "")
        nights   = h.get("nights", 0) or 0
        deal     = h.get("deal_active", False)
        deal_pct = h.get("deal_discount_percentage")
        desc     = h.get("description") or ""
        ci       = h.get("check_in_time") or ""
        co       = h.get("check_out_time") or ""
        cancel   = h.get("cancellation_policy") or {}
        nearby   = h.get("nearby_places") or []
        rev_bd   = h.get("review_breakdown") or []
        amenities_raw = h.get("amenities") or []
        excl_raw      = h.get("excluded_amenities") or []

        # Parse amenity names from dicts or strings
        def _aname(a):
            if isinstance(a, dict): return a.get("name", str(a))
            return str(a)
        amenities = [_aname(a) for a in amenities_raw]
        excluded  = [_aname(a) for a in excl_raw[:4]]

        # Badges
        rec_badge  = '<span class="pill pill-blue">★ Best pick</span>' if label == "recommended" else '<span class="pill pill-grey">Alternative</span>'
        deal_html  = f'<span class="pill pill-green">🏷 {deal_pct:.0f}% off</span>' if deal and deal_pct else ""
        stars_s    = "★" * int(rating) + "☆" * (5 - int(rating)) if rating else ""

        # Tags
        tags = []
        if location: tags.append(location)
        if htype:    tags.append(htype)
        if nights:   tags.append(f"{nights} nights")
        if ci:       tags.append(f"Check-in {ci}")
        if co:       tags.append(f"Check-out {co}")
        tags_html = '<div class="hotel-tags">' + "".join(f'<span class="tag">{t}</span>' for t in tags) + "</div>" if tags else ""

        # Description
        desc_html = f'<div class="hotel-desc">{desc}</div>' if desc else ""

        # Amenities — all of them
        am_incl = [a for a in amenities]
        amenity_html = ""
        if am_incl:
            pills = "".join(f'<span class="pill pill-blue">{a}</span>' for a in am_incl)
            amenity_html = (
                f'<div class="amenities-section">'
                f'<div class="amenities-title">Amenities</div>'
                f'<div class="amenities-row" style="display:flex;flex-wrap:wrap;gap:5px">{pills}</div>'
                f'</div>'
            )
        excl_html = ""
        if excluded:
            epills = "".join(f'<span class="pill pill-grey">✗ {a}</span>' for a in excluded)
            excl_html = f'<div class="amenities-row" style="display:flex;flex-wrap:wrap;gap:5px;margin-top:4px">{epills}</div>'

        # Nearby places
        nearby_html = ""
        if nearby:
            rows = ""
            for np in nearby[:5]:
                nm = np.get("name", "")
                dm = np.get("distance_minutes", 0)
                tp = ", ".join(np.get("transportation_types", [])[:2])
                rows += (
                    f'<div class="nearby-item">'
                    f'<span class="nearby-name">{nm}</span>'
                    f'<span class="tag">{dm} min · {tp}</span>'
                    f'</div>'
                )
            nearby_html = (
                f'<div class="nearby-block">'
                f'<div class="amenities-title">Nearby places</div>'
                f'{rows}</div>'
            )

        # Cancellation
        cancel_html = ""
        if cancel:
            free = cancel.get("free_cancellation", False)
            deadline = cancel.get("cancellation_deadline", "")
            desc_c = cancel.get("policy_description", "")
            if free:
                cancel_html = f'<div class="cancel-block">✅ Free cancellation{(" until " + deadline) if deadline else ""}</div>'
            elif desc_c:
                cancel_html = f'<div class="cancel-block" style="background:#fef2f2;color:#dc2626">{desc_c}</div>'

        # Review breakdown
        rb_html = ""
        if rev_bd:
            rows = ""
            for rb in rev_bd[:6]:
                cat  = rb.get("category") or rb.get("name", "")
                pos  = rb.get("positive", 0)
                tot  = rb.get("total_mentioned", 1) or 1
                pct  = int(pos / tot * 100) if tot else 0
                rows += (
                    f'<div class="rb-row">'
                    f'<span class="rb-label">{cat}</span>'
                    f'<div class="rb-bar-wrap"><div class="rb-bar" style="width:{pct}%"></div></div>'
                    f'<span class="rb-pct">{pct}%</span>'
                    f'</div>'
                )
            rb_html = (
                f'<div class="review-breakdown">'
                f'<div class="amenities-title">Review highlights</div>'
                f'{rows}</div>'
            )

        return (
            f'<div class="hotel-card">'
            # Name + price row
            f'<div class="hotel-name-row">'
            f'<div class="hotel-name">{name}</div>'
            f'<div class="hotel-price-block">'
            f'<div class="hotel-ppn">€{ppn:.0f}<span style="font-size:0.72rem;font-weight:400;color:#94a3b8">/night</span></div>'
            + (f'<div class="hotel-total">€{total:.0f} total</div>' if total else '')
            + f'</div></div>'
            # Badges + rating
            f'<div class="hotel-rating-row">'
            f'{rec_badge} {deal_html}'
            f'<span class="stars">{stars_s}</span>'
            + (f'<span style="font-size:0.82rem;color:#64748b">{rating} ({reviews:,} reviews)</span>' if rating else '')
            + (f'<span class="pill pill-grey">📍 Location {loc_rat}</span>' if loc_rat else '')
            + f'</div>'
            # Tags + desc + amenities + nearby + cancel + review breakdown
            + tags_html + desc_html + amenity_html + excl_html
            + nearby_html + cancel_html + rb_html
            + f'</div>'
        )

    sub_parts = []
    if pr.get("min"):  sub_parts.append(f"From €{pr['min']:.0f}/night")
    if pr.get("max"):  sub_parts.append(f"Up to €{pr['max']:.0f}/night")
    if rr.get("min"):  sub_parts.append(f"Ratings {rr['min']}–{rr.get('max','')}")
    sub = " · ".join(sub_parts)

    reason_html = (
        f'<div class="card-reasoning">'
        f'<div class="reasoning-label">💡 Why this hotel</div>'
        f'{reason}'
        f'</div>'
    ) if reason else ""

    cards_html = (
        '<div class="section-label">Recommended</div>'
        + _hotel_card(rec, "recommended")
    )
    if alts:
        cards_html += '<div class="section-label">Other options</div>'
        for alt in alts:
            cards_html += _hotel_card(alt, "alternative")

    html = (
        _CARD_BASE_CSS
        + '<div class="card-wrap">'
        + '<div class="card-header">'
        + '<div class="card-title">🏨 Hotel Results</div>'
        + (f'<div class="card-sub">{sub}</div>' if sub else '')
        + '</div>'
        + f'<div class="card-body">{cards_html}</div>'
        + reason_html
        + '</div>'
    )

    def _card_height(h: dict) -> int:
        # More generous estimates — underestimating causes cut-off
        base = 220                                          # name + price + badges + tags
        if h.get("description"):           base += 50
        amenity_count = len(h.get("amenities") or [])
        base += (amenity_count // 4 + 1) * 32              # ~4 chips per row, 32px per row
        excl_count = len(h.get("excluded_amenities") or [])
        base += (excl_count // 4 + 1) * 28 if excl_count else 0
        nearby_count = min(len(h.get("nearby_places") or []), 5)
        base += 30 + nearby_count * 30 if nearby_count else 0
        if h.get("cancellation_policy"):   base += 45
        rb_count = min(len(h.get("review_breakdown") or []), 6)
        base += 30 + rb_count * 26 if rb_count else 0
        return base

    h = 180 + _card_height(rec) + sum(_card_height(a) for a in alts) + (80 if reason else 0)
    components.html(html, height=max(h, 500), scrolling=True)


def _render_attractions(payload: dict):
    import streamlit.components.v1 as components

    destination = payload.get("destination", "")
    recs   = payload.get("recommended", []) or []
    alts   = payload.get("alternatives", []) or []
    reason = payload.get("reasoning", "")
    total  = payload.get("estimated_cost")

    if not recs:
        st.markdown('<div class="msg-assistant">No attractions found.</div>', unsafe_allow_html=True)
        return

    def _attr_card(a: dict) -> str:
        name     = a.get("name", "Unknown")
        cat      = a.get("category") or "General"
        rating   = a.get("rating") or 0
        reviews  = a.get("reviews_count") or 0
        price    = a.get("price")
        is_free  = a.get("is_free", False)
        desc     = a.get("description") or ""
        status   = a.get("open_status") or ""
        dist     = a.get("distance_minutes")
        dist_src = a.get("distance_source") or ""

        price_html = (
            '<span class="pill pill-green">Free</span>' if is_free
            else f'<span class="pill pill-blue">€{price:.0f}</span>' if price
            else ""
        )
        status_html = (
            f'<span class="pill pill-green">{status}</span>' if status == "Open"
            else f'<span class="pill pill-red">{status}</span>' if status
            else ""
        )
        dist_html  = f'<span class="tag">📍 {dist} min {dist_src}</span>' if dist else ""
        stars      = "★" * int(rating) + "☆" * (5 - int(rating)) if rating else ""
        reviews_s  = f"({reviews:,})" if reviews else ""
        desc_html  = f'<div class="attr-desc">{desc}</div>' if desc else ""

        return (
            f'<div class="attr-card">'
            f'<div class="attr-cat">{cat}</div>'
            f'<div class="attr-name">{name}</div>'
            f'{desc_html}'
            f'<div class="attr-meta">'
            f'<span class="attr-stars">{stars}</span>'
            f'<span style="color:#64748b">{rating} {reviews_s}</span>'
            f'{price_html}{status_html}{dist_html}'
            f'</div>'
            f'</div>'
        )

    rec_cards = "".join(_attr_card(a) for a in recs)
    alt_cards = ""
    if alts:
        alt_items = "".join(_attr_card(a) for a in alts)
        alt_cards = (
            '<div class="section-label">More options</div>'
            f'<div class="attr-grid">{alt_items}</div>'
        )

    sub_parts = [destination] if destination else []
    if total:            sub_parts.append(f"Est. entry cost €{total:.0f}")
    sub_parts.append(f"{len(recs)} recommended")
    if alts: sub_parts.append(f"{len(alts)} more")
    sub = " · ".join(sub_parts)

    reason_html = (
        f'<div class="card-reasoning">'
        f'<div class="reasoning-label">💡 How we ranked these</div>'
        f'{reason}'
        f'</div>'
    ) if reason else ""

    html = (
        _CARD_BASE_CSS
        + '<div class="card-wrap">'
        + '<div class="card-header">'
        + '<div class="card-title">🗺️ Top Attractions</div>'
        + (f'<div class="card-sub">{sub}</div>' if sub else '')
        + '</div>'
        + f'<div class="card-body">'
        + '<div class="section-label">Recommended</div>'
        + f'<div class="attr-grid">{rec_cards}</div>'
        + f'{alt_cards}</div>'
        + reason_html
        + '</div>'
    )

    rec_rows = (len(recs) + 1) // 2
    alt_rows = (len(alts) + 1) // 2
    h = 160 + rec_rows * 155 + (30 if alts else 0) + alt_rows * 145 + (80 if reason else 0)
    components.html(html, height=max(h, 350), scrolling=True)

# Inline CSS for the itinerary component — self-contained so it works
# inside st.components.v1.html() which runs in an iframe
_ITIN_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@700;800&family=DM+Sans:wght@300;400;500&display=swap');
* { box-sizing: border-box; margin: 0; padding: 0; font-family: 'DM Sans', sans-serif; }
html, body { background: transparent; padding: 4px 0; overflow-y: auto; }

.itin-wrap { border-radius: 14px; overflow: visible; box-shadow: 0 2px 16px rgba(0,0,0,0.08); }

.itin-header { background: linear-gradient(135deg,#0f172a,#1e293b); padding: 20px 24px 18px; border-radius: 14px 14px 0 0; }
.itin-title  { font-family:'Syne',sans-serif; font-weight:800; font-size:1.25rem; color:#f1f5f9; margin-bottom:10px; }
.itin-highlights { font-size:0.82rem; color:#94a3b8; line-height:1.6; margin-bottom:12px; }
.itin-pills  { display:flex; flex-wrap:wrap; gap:8px; }
.itin-pill   { background:rgba(255,255,255,0.09); border:1px solid rgba(255,255,255,0.14);
               border-radius:20px; padding:5px 13px; font-size:0.79rem; color:#cbd5e1; }
.itin-pill strong { color:#f1f5f9; }

.itin-cost  { background:linear-gradient(90deg,#0ea5e9,#6366f1); padding:11px 24px;
              font-size:0.88rem; color:white; font-weight:600; }
.itin-warn  { background:#fef3c7; border-left:3px solid #f59e0b; padding:9px 24px;
              font-size:0.82rem; color:#92400e; }

.itin-body  { border:1px solid #e2e8f0; border-top:none; background:white; }

.day-block  { border-bottom:1px solid #f1f5f9; }
.day-block:last-child { border-bottom:none; }

.day-hdr    { display:flex; align-items:center; gap:12px; padding:12px 20px; background:#f8fafc; }
.day-num    { background:linear-gradient(135deg,#0ea5e9,#6366f1); color:white;
              font-family:'Syne',sans-serif; font-weight:700; font-size:0.77rem;
              width:28px; height:28px; border-radius:50%;
              display:flex; align-items:center; justify-content:center; flex-shrink:0; }
.day-theme  { font-family:'Syne',sans-serif; font-weight:600; font-size:0.9rem; color:#1e293b; flex:1; }
.day-meta   { font-size:0.74rem; color:#94a3b8; white-space:nowrap; }

.acts-list  { padding:4px 20px 10px; }
.day-note   { font-size:0.77rem; color:#64748b; padding:5px 0 3px; font-style:italic; }

.act-row    { display:flex; align-items:flex-start; gap:10px; padding:8px 0;
              border-bottom:1px solid #f8fafc; }
.act-row:last-child { border-bottom:none; }
.act-icon   { font-size:0.95rem; width:26px; text-align:center; flex-shrink:0; padding-top:1px; }
.act-time   { font-size:0.73rem; color:#94a3b8; font-weight:500; width:44px;
              flex-shrink:0; padding-top:3px; font-family:monospace; }
.act-body   { flex:1; min-width:0; }
.act-title  { font-size:0.88rem; font-weight:500; color:#1e293b; line-height:1.3; }
.act-desc   { font-size:0.79rem; color:#64748b; margin-top:3px; line-height:1.4; }
.act-notes  { font-size:0.76rem; color:#0ea5e9; margin-top:3px; font-style:italic; }
.act-dur    { font-size:0.73rem; color:#94a3b8; margin-top:2px; }
.act-badge  { font-size:0.75rem; font-weight:600; border-radius:10px; padding:2px 9px;
              white-space:nowrap; flex-shrink:0; align-self:flex-start; margin-top:3px; }
.act-cost   { background:#e0f2fe; color:#0369a1; }
.act-free   { background:#dcfce7; color:#15803d; }

.itin-tips       { background:#fffbeb; border-top:1px solid #fde68a; padding:14px 20px; }
.tips-title      { font-family:'Syne',sans-serif; font-weight:700; font-size:0.79rem;
                   color:#92400e; margin-bottom:9px; text-transform:uppercase; letter-spacing:0.05em; }
.tip-row         { display:flex; align-items:flex-start; gap:8px; font-size:0.84rem;
                   color:#78350f; padding:3px 0; line-height:1.4; }
.tip-dot         { color:#f59e0b; flex-shrink:0; font-size:0.68rem; padding-top:4px; }
</style>
"""


def _render_itinerary(payload: dict):
    """
    Render a full structured itinerary.
    Uses st.components.v1.html() so Streamlit's markdown parser
    never touches the HTML — no more escaped angle brackets.
    """
    import streamlit.components.v1 as components

    itin    = payload.get("itinerary", {})
    flight  = payload.get("flight")
    ret_flt = payload.get("return_flight")
    hotel   = payload.get("hotel")
    warn    = payload.get("budget_warning")

    title      = itin.get("trip_title", "Your Itinerary")
    highlights = itin.get("highlights", [])
    days       = itin.get("days", [])
    tips       = itin.get("practical_tips", [])
    total_cost = itin.get("total_estimated_cost")
    cost_bd    = itin.get("cost_breakdown") or {}

    # ── Header ────────────────────────────────────────────────────────────
    hl_html = ""
    if highlights:
        hl_html = '<div class="itin-highlights">✨ ' + " &nbsp;·&nbsp; ".join(highlights) + "</div>"

    pills = []
    if flight:
        dep     = flight.get("departure_airport_code", "")
        arr     = flight.get("arrival_airport_code", "")
        out_p   = flight.get("price", 0) or 0
        ret_p   = (ret_flt or {}).get("price", 0) or 0

        if ret_flt and ret_p:
            if out_p > 0 and abs(ret_p - out_p) / out_p < 0.15:
                # Similar prices → confirmed round-trip total
                price_s = f"€{ret_p:.0f} round-trip"
            else:
                # Per-leg prices → sum
                price_s = f"€{out_p + ret_p:.0f} round-trip"
        elif out_p:
            price_s = f"€{out_p:.0f} one-way"
        else:
            price_s = "N/A"

        pills.append(
            f"<span class='itin-pill'>✈️ <strong>{flight.get('airline','')}</strong> "
            f"{dep} → {arr} · {price_s}</span>"
        )
    if ret_flt:
        pills.append(
            f"<span class='itin-pill'>✈️ Return <strong>{ret_flt.get('airline','')}</strong> "
            f"{ret_flt.get('departure_airport_code','')} → {ret_flt.get('arrival_airport_code','')} "
            f"· included in total</span>"
        )
    if hotel:
        ppn = hotel.get("price_per_night", 0)
        pills.append(
            f"<span class='itin-pill'>🏨 <strong>{hotel.get('name','')}</strong> "
            f"· €{ppn:.0f}/night</span>"
        )
    pills_html = f'<div class="itin-pills">{"".join(pills)}</div>' if pills else ""

    # ── Cost banner ───────────────────────────────────────────────────────
    cost_html = ""
    if total_cost:
        parts = []
        if cost_bd.get("flights"):     parts.append(f"Flights €{cost_bd['flights']:.0f}")
        if cost_bd.get("hotel"):       parts.append(f"Hotel €{cost_bd['hotel']:.0f}")
        if cost_bd.get("attractions"): parts.append(f"Attractions €{cost_bd['attractions']:.0f}")
        bd_str = " &nbsp;·&nbsp; ".join(parts)
        cost_html = (
            f'<div class="itin-cost">Total estimated cost: €{total_cost:.0f}'
            + (f" &nbsp;({bd_str})" if bd_str else "")
            + "</div>"
        )

    warn_html = f'<div class="itin-warn">⚠️ {warn}</div>' if warn else ""

    # ── Days ──────────────────────────────────────────────────────────────
    days_html = ""
    for day in days:
        day_num    = day.get("day", "")
        theme      = day.get("theme") or f"Day {day_num}"
        date_str   = day.get("date", "")
        activities = day.get("activities", [])
        day_cost   = day.get("estimated_daily_cost")
        day_note   = day.get("notes", "")

        meta_parts = []
        if date_str: meta_parts.append(date_str)
        if day_cost: meta_parts.append(f"€{day_cost:.0f}")
        meta = " &nbsp;·&nbsp; ".join(meta_parts)

        note_html = f'<div class="day-note">{day_note}</div>' if day_note else ""

        acts = ""
        for act in activities:
            atype   = act.get("type", "attraction")
            icon    = _ACT_ICON.get(atype, "📍")
            time_s  = act.get("time") or ""
            title_s = act.get("title", "")
            desc_s  = act.get("description") or ""
            notes_s = act.get("notes") or ""
            dur     = act.get("duration_minutes")
            cost_v  = act.get("cost")
            is_free = act.get("is_free", False)

            desc_h  = f'<div class="act-desc">{desc_s}</div>'  if desc_s  else ""
            notes_h = f'<div class="act-notes">💡 {notes_s}</div>' if notes_s else ""
            dur_h   = f'<div class="act-dur">~{dur} min</div>' if dur     else ""

            if cost_v:
                badge = f'<span class="act-badge act-cost">€{cost_v:.0f}</span>'
            elif is_free:
                badge = '<span class="act-badge act-free">Free</span>'
            else:
                badge = ""

            acts += (
                f'<div class="act-row">'
                f'<div class="act-icon">{icon}</div>'
                f'<div class="act-time">{time_s}</div>'
                f'<div class="act-body">'
                f'<div class="act-title">{title_s}</div>'
                f'{desc_h}{notes_h}{dur_h}'
                f'</div>'
                f'{badge}'
                f'</div>'
            )

        days_html += (
            f'<div class="day-block">'
            f'<div class="day-hdr">'
            f'<div class="day-num">{day_num}</div>'
            f'<div class="day-theme">{theme}</div>'
            f'<div class="day-meta">{meta}</div>'
            f'</div>'
            f'<div class="acts-list">{note_html}{acts}</div>'
            f'</div>'
        )

    # ── Tips ──────────────────────────────────────────────────────────────
    tips_html = ""
    if tips:
        tip_rows = "".join(
            f'<div class="tip-row"><span class="tip-dot">▶</span>{t}</div>'
            for t in tips
        )
        tips_html = (
            f'<div class="itin-tips">'
            f'<div class="tips-title">💡 Travel Tips</div>'
            f'{tip_rows}</div>'
        )

    # ── Assemble and render as single HTML component ──────────────────────
    html = (
        _ITIN_CSS
        + '<div class="itin-wrap">'
        + f'<div class="itin-header">'
        + f'<div class="itin-title">{title}</div>'
        + hl_html + pills_html
        + '</div>'
        + cost_html
        + warn_html
        + f'<div class="itin-body">{days_html}</div>'
        + tips_html
        + '</div>'
    )

    # Fixed height with scrolling — user scrolls inside the component.
    # 700px shows ~3-4 days visible at once; scroll reveals the rest.
    components.html(html, height=700, scrolling=True)


def _render_msg(msg: dict):
    role    = msg["role"]
    content = msg["content"]
    mtype   = msg.get("type", "text")

    if role == "user":
        st.markdown(f'<div class="msg-user">{content}</div>', unsafe_allow_html=True)
        return

    if mtype == "progress":
        steps = [s for s in content.split("\n") if s.strip()]
        inner = "".join(
            f'<div class="prog-step"><span class="prog-dot"></span>{s}</div>'
            for s in steps
        )
        st.markdown(f'<div class="prog-block">{inner}</div>', unsafe_allow_html=True)
    elif mtype == "hitl":
        try:
            _render_hitl(json.loads(content))
        except Exception:
            st.markdown(f'<div class="msg-assistant">{content}</div>', unsafe_allow_html=True)
    elif content.startswith("__ITINERARY__:"):
        try:
            payload = json.loads(content[len("__ITINERARY__:"):])
            itin    = payload.get("itinerary", {})
            flight  = payload.get("flight") or {}
            hotel   = payload.get("hotel") or {}
            warn    = payload.get("budget_warning", "")

            title      = itin.get("trip_title", "Your Itinerary")
            highlights = itin.get("highlights", [])
            days       = itin.get("days", [])
            total_cost = itin.get("total_estimated_cost")
            cost_bd    = itin.get("cost_breakdown") or {}

            # ── Always-visible plain text summary ────────────────────────
            lines = [f"### 📅 {title}"]
            if highlights:
                lines.append("✨ " + " · ".join(highlights[:4]))
            summary_parts = []
            if flight.get("airline"):
                price = (payload.get("return_flight") or {}).get("price") or flight.get("price", 0)
                is_rt = payload.get("return_flight") is not None
                summary_parts.append(f"✈️ {flight['airline']} · {'€'+str(int(price))+(' RT' if is_rt else '') if price else 'N/A'}")
            if hotel.get("name"):
                summary_parts.append(f"🏨 {hotel['name']} · €{hotel.get('price_per_night',0):.0f}/night")
            if total_cost:
                parts = []
                if cost_bd.get("flights"):     parts.append(f"flights €{cost_bd['flights']:.0f}")
                if cost_bd.get("hotel"):       parts.append(f"hotel €{cost_bd['hotel']:.0f}")
                if cost_bd.get("attractions"): parts.append(f"activities €{cost_bd['attractions']:.0f}")
                breakdown = " · ".join(parts)
                summary_parts.append(f"💰 Total €{total_cost:.0f}" + (f" ({breakdown})" if breakdown else ""))
            if summary_parts:
                lines.append("  ".join(summary_parts))
            if warn:
                lines.append(f"⚠️ {warn}")

            # Day previews — 2 lines per day
            lines.append("")
            for day in days:
                theme = day.get("theme", f"Day {day.get('day','')}")
                acts  = day.get("activities", [])
                act_titles = " · ".join(
                    a.get("title", "") for a in acts[:3] if a.get("title")
                )
                lines.append(f"**Day {day.get('day','')} — {theme}**  {act_titles}{'…' if len(acts) > 3 else ''}")

            tips = itin.get("practical_tips", [])
            if tips:
                lines.append(f"\n💡 **Tips:** " + " · ".join(tips[:3]))

            st.markdown("\n\n".join(lines))

            # ── Full rich card in expander ────────────────────────────────
            with st.expander("📋 View full day-by-day itinerary", expanded=False):
                _render_itinerary(payload)

        except Exception as e:
            st.markdown(f'<div class="msg-assistant">⚠️ Could not render itinerary: {e}</div>',
                        unsafe_allow_html=True)
    elif content.startswith("__FLIGHTS__:"):
        try:
            payload = json.loads(content[len("__FLIGHTS__:"):])
            rec     = payload.get("recommended") or {}
            alts    = payload.get("alternatives") or []
            reason  = payload.get("reasoning", "")
            ret_flt = payload.get("return_flight")

            if rec:
                airline  = rec.get("airline", "")
                dep      = rec.get("departure_airport_code", "")
                arr      = rec.get("arrival_airport_code", "")
                ftype    = rec.get("type", "").replace("_", " ")
                dep_time = rec.get("departure_time", "")
                arr_time = rec.get("arrival_time", "")
                dur      = rec.get("total_duration_hours", 0)
                # Price: use return price for round-trips
                price = (ret_flt or {}).get("price") or rec.get("price", 0)
                is_rt = ret_flt is not None
                price_s = f"€{price:.0f} {'round-trip' if is_rt else 'one-way'}" if price else "N/A"

                lines = [f"**✈️ Best flight: {airline}** — {ftype} · {dep} → {arr} · {price_s}"]
                if dep_time or arr_time:
                    lines.append(f"Departs **{dep_time}** → Arrives **{arr_time}**" + (f" · {dur:.1f}h" if dur else ""))
                if reason:
                    lines.append(f"\n💡 *{reason}*")
                if alts:
                    lines.append(f"\n{len(alts)} alternative{'s' if len(alts) != 1 else ''} available:")
                    for a in alts:
                        ap = a.get("price", 0)
                        at = a.get("type", "").replace("_", " ")
                        lines.append(f"  • {a.get('airline','')} — {at} · {'€'+str(int(ap)) if ap else 'N/A'} · {a.get('departure_time','')} → {a.get('arrival_time','')}")
                if ret_flt:
                    r = ret_flt
                    lines.append(f"\n**↩️ Return: {r.get('airline','')}** {r.get('departure_airport_code','')} → {r.get('arrival_airport_code','')} · Departs {r.get('departure_time','')} → Arrives {r.get('arrival_time','')}")

                st.markdown("\n\n".join(lines))

                with st.expander("📋 Full flight details"):
                    _render_flights(payload)
        except Exception as e:
            st.markdown(f"⚠️ Could not display flight results: {e}")

    elif content.startswith("__HOTELS__:"):
        try:
            payload = json.loads(content[len("__HOTELS__:"):])
            rec     = payload.get("recommended") or {}
            alts    = payload.get("alternatives") or []
            reason  = payload.get("reasoning", "")

            if rec:
                name     = rec.get("name", "")
                ppn      = rec.get("price_per_night", 0)
                total    = rec.get("total_price", 0)
                rating   = rec.get("overall_rating", "")
                reviews  = rec.get("total_reviews", 0)
                location = rec.get("location", "")
                nights   = rec.get("nights", 0)
                ci       = rec.get("check_in_time", "")
                co       = rec.get("check_out_time", "")
                deal     = rec.get("deal_active", False)
                deal_pct = rec.get("deal_discount_percentage")
                amenities_raw = rec.get("amenities") or []
                top_amenities = [
                    (a.get("name") if isinstance(a, dict) else str(a))
                    for a in amenities_raw[:6]
                ]

                lines = [f"**🏨 Recommended: {name}**"]
                price_line = f"€{ppn:.0f}/night" + (f" · €{total:.0f} total for {nights} nights" if total and nights else "")
                if deal and deal_pct:
                    price_line += f" · 🏷 {deal_pct:.0f}% off"
                lines.append(price_line)
                if rating:
                    lines.append(f"★{rating}" + (f" ({reviews:,} reviews)" if reviews else "") + (f" · {location}" if location else ""))
                if ci or co:
                    lines.append(f"Check-in: {ci} · Check-out: {co}")
                if top_amenities:
                    lines.append("Amenities: " + " · ".join(top_amenities))
                if reason:
                    lines.append(f"\n💡 *{reason}*")
                if alts:
                    lines.append(f"\n{len(alts)} alternative{'s' if len(alts) != 1 else ''} available:")
                    for a in alts:
                        ap = a.get("price_per_night", 0)
                        ar = a.get("overall_rating", "")
                        lines.append(f"  • **{a.get('name','')}** — €{ap:.0f}/night" + (f" · ★{ar}" if ar else ""))

                st.markdown("\n\n".join(lines))

                with st.expander("📋 Full hotel details & alternatives"):
                    _render_hotels(payload)
        except Exception as e:
            st.markdown(f"⚠️ Could not display hotel results: {e}")

    elif content.startswith("__ATTRACTIONS__:"):
        try:
            payload     = json.loads(content[len("__ATTRACTIONS__:"):])
            destination = payload.get("destination", "")
            recs        = payload.get("recommended") or []
            alts        = payload.get("alternatives") or []
            reason      = payload.get("reasoning", "")
            total_cost  = payload.get("estimated_cost")

            if recs:
                dest_s    = f" in {destination}" if destination else ""
                cost_s    = f" · Est. entry cost €{total_cost:.0f}" if total_cost else ""
                lines     = [f"**🗺️ {len(recs)} recommended attractions{dest_s}**{cost_s}"]
                for a in recs:
                    cat     = a.get("category") or ""
                    price   = a.get("price")
                    is_free = a.get("is_free", False)
                    rating  = a.get("rating", "")
                    price_s = "Free" if is_free else (f"€{price:.0f}" if price else "")
                    rating_s = f"★{rating}" if rating else ""
                    meta    = " · ".join(filter(None, [cat, rating_s, price_s]))
                    lines.append(f"  • **{a.get('name','')}**" + (f" — {meta}" if meta else ""))
                if reason:
                    lines.append(f"\n💡 *{reason}*")
                if alts:
                    lines.append(f"\n*{len(alts)} more options available below.*")

                st.markdown("\n\n".join(lines))

                with st.expander("📋 Full attraction details & more options"):
                    _render_attractions(payload)
        except Exception as e:
            st.markdown(f"⚠️ Could not display attraction results: {e}")
    else:
        st.markdown(f'<div class="msg-assistant">{content}</div>', unsafe_allow_html=True)


# ============================================================================
# PROCESS PENDING — runs BEFORE rendering so results appear on same rerun
# ============================================================================
orc = get_orchestrator()

if st.session_state.pending_hitl:
    decision = st.session_state.pending_hitl
    st.session_state.pending_hitl = None

    # Render chat history (includes the approve/reject user message)
    for msg in st.session_state.history:
        _render_msg(msg)

    live_placeholder = st.empty()
    live_steps: list[str] = []

    label = "✅ Applying recommendation…" if decision == "approve" else "❌ Keeping original plan…"
    _add("user", f"**{label}**")
    st.session_state.awaiting_hitl = False

    progress, response, error = [], None, None

    for event in orc.resume(decision, st.session_state.thread_id):
        t = event.get("type")
        if t == "progress":
            progress.append(event["data"])
            live_steps.append(event["data"])
            inner = "".join(
                f'<div class="prog-step"><span class="prog-dot"></span>{s}</div>'
                for s in live_steps
            )
            live_placeholder.markdown(
                f'<div class="prog-block">{inner}</div>',
                unsafe_allow_html=True,
            )
        elif t == "response": response = event["data"]
        elif t == "error":    error    = event["data"]

    live_placeholder.empty()

    if progress: _add("assistant", "\n".join(progress), "progress")
    if response:
        _add("assistant", response, "text")
        _save_trip(orc)
    if error:    _add("assistant", f"⚠️ {error}", "text")
    st.rerun()

elif st.session_state.pending_input:
    user_msg = st.session_state.pending_input
    st.session_state.pending_input = None

    # Render chat history so the user message is visible while we process
    for msg in st.session_state.history:
        _render_msg(msg)

    # Live progress placeholder — updates as each step arrives
    live_placeholder = st.empty()
    live_steps: list[str] = []

    progress, response, hitl_data, error = [], None, None, None

    for event in orc.chat(user_msg, st.session_state.thread_id):
        t = event.get("type")
        if t == "progress":
            progress.append(event["data"])
            live_steps.append(event["data"])
            inner = "".join(
                f'<div class="prog-step"><span class="prog-dot"></span>{s}</div>'
                for s in live_steps
            )
            live_placeholder.markdown(
                f'<div class="prog-block">{inner}</div>',
                unsafe_allow_html=True,
            )
        elif t == "hitl":
            hitl_data = event["data"]
            st.session_state.awaiting_hitl = True
        elif t == "response":
            response = event["data"]
        elif t == "error":
            error = event["data"]

    live_placeholder.empty()

    if progress:  _add("assistant", "\n".join(progress), "progress")
    if hitl_data: _add("assistant", json.dumps(hitl_data), "hitl")
    if response:
        _add("assistant", response, "text")
        _save_trip(orc)
    if error:     _add("assistant", f"⚠️ {error}", "text")

    # ── Safety net: stream ended without HITL or response ─────────────────
    # The graph may have paused at interrupt() but not streamed the event.
    # Check graph state directly and recover.
    if not hitl_data and not response and not error:
        try:
            config = {"configurable": {"thread_id": st.session_state.thread_id}}
            graph_state = orc.graph.get_state(config)
            # Graph is paused if it has pending next steps but no final response
            if graph_state and graph_state.next:
                # Try to find interrupt value from tasks
                recovered_payload = None
                for task in (graph_state.tasks or []):
                    for intr in (getattr(task, "interrupts", None) or []):
                        recovered_payload = getattr(intr, "value", {})
                        break
                    if recovered_payload is not None:
                        break

                if recovered_payload is not None:
                    # Interrupt was missed — add it now
                    _add("assistant", json.dumps(recovered_payload), "hitl")
                    st.session_state.awaiting_hitl = True
                else:
                    # Graph paused for unknown reason
                    _add("assistant",
                         "⚠️ Something went wrong — the plan couldn't be completed. "
                         "Try rephrasing your request or start a new trip.",
                         "text")
        except Exception:
            pass
    st.rerun()


# ============================================================================
# SIDEBAR
# ============================================================================
with st.sidebar:
    st.markdown(
        '<div class="sb-title">✈ Travel Planner</div>'
        '<div class="sb-sub">AI-powered trip planning</div>',
        unsafe_allow_html=True,
    )

    if st.button("＋  New Trip", use_container_width=True):
        st.session_state.thread_id    = str(uuid.uuid4())
        st.session_state.history      = []
        st.session_state.awaiting_hitl = False
        st.session_state.pending_input = None
        st.session_state.pending_hitl  = None
        st.rerun()

    st.markdown('<div class="sb-label">Current Trip</div>', unsafe_allow_html=True)
    try:
        _render_state_card(orc)
    except Exception:
        pass

    if st.session_state.past_trips:
        st.markdown('<div class="sb-label">History</div>', unsafe_allow_html=True)
        for trip in st.session_state.past_trips[:8]:
            icon = "✓" if trip["status"] == "complete" else "○"
            st.markdown(
                f'<div class="hist-item">'
                f'<div class="hist-dest">{icon} {trip["destination"]}</div>'
                f'<div class="hist-meta">{trip["budget"]} · {trip["duration"]}</div>'
                f'</div>',
                unsafe_allow_html=True,
            )
            if st.button("Load", key=f"ld_{trip['thread_id']}", use_container_width=True):
                st.session_state.thread_id = trip["thread_id"]
                st.session_state.history   = []
                st.rerun()

    st.markdown("""
    <div class="sb-tips">
        <strong>Try asking:</strong><br>
        "Plan a trip to Bali, €3000, 10 days"<br>
        "Find flights from Berlin to Rome"<br>
        "Hotels in Lisbon under €100/night"<br>
        "Things to do in Tokyo for 5 days"
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN CONTENT
# ============================================================================
st.markdown("""
<div class="app-header">
    <h1>AI Travel Planner</h1>
    <p>Plan your perfect trip — flights, hotels, attractions &amp; itinerary in one go.</p>
</div>
""", unsafe_allow_html=True)

# ── Render chat ───────────────────────────────────────────────────────────────
if not st.session_state.history:
    st.markdown("""
    <div class="welcome-wrap">
        <div class="welcome-icon">🌍</div>
        <div class="welcome-h2">Where would you like to go?</div>
        <div class="welcome-sub">Tell me your destination, budget and dates — I'll handle the rest.</div>
    </div>
    """, unsafe_allow_html=True)

    suggestions = [
        "Plan a trip to Madeira, €2500, 7 days from Frankfurt",
        "Flights from London to Barcelona next week",
        "Hotels in Rome, June 10th for 5 nights",
        "Things to do in Tokyo for 4 days",
        "Beach trip to Bali, €4000, 10 days",
    ]
    cols = st.columns(len(suggestions))
    for i, (col, sug) in enumerate(zip(cols, suggestions)):
        with col:
            label = sug[:28] + "…" if len(sug) > 28 else sug
            if st.button(label, key=f"chip_{i}", use_container_width=True):
                _add("user", sug)
                st.session_state.pending_input = sug
                st.rerun()
else:
    for msg in st.session_state.history:
        _render_msg(msg)

# ── HITL buttons ──────────────────────────────────────────────────────────────
if st.session_state.awaiting_hitl:
    ca, cr, ci = st.columns([1, 1, 3])
    with ca:
        if st.button("✅ Approve & Replan", type="primary", key="btn_approve"):
            st.session_state.pending_hitl = "approve"
            st.rerun()
    with cr:
        if st.button("❌ Keep Original", key="btn_reject"):
            st.session_state.pending_hitl = "reject"
            st.rerun()
    with ci:
        st.markdown(
            '<span style="font-size:0.8rem;color:#94a3b8">'
            "Approve to apply the recommendation, or keep the original itinerary."
            "</span>",
            unsafe_allow_html=True,
        )

# ── Input ─────────────────────────────────────────────────────────────────────
user_input = st.chat_input(
    "Ask me to plan a trip, find flights, hotels or attractions…",
    disabled=st.session_state.awaiting_hitl,
)

if user_input and not st.session_state.awaiting_hitl:
    # KEY FIX: add to history NOW so it renders immediately on next rerun,
    # then store as pending — actual processing happens at top of NEXT rerun
    _add("user", user_input)
    st.session_state.pending_input = user_input
    st.rerun()