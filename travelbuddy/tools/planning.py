"""Deterministic budget planning tools. LLMs (especially small local ones) are unreliable at
arithmetic, so all money math happens here and the model only supplies the inputs."""

from .registry import Tool

# Share of the total budget per category, by travel style. Each row sums to 1.0.
STYLE_SPLITS = {
    "budget": {"transport_to_destination": 0.30, "accommodation": 0.25, "food": 0.20,
               "activities": 0.12, "local_transport": 0.06, "buffer": 0.07},
    "mid-range": {"transport_to_destination": 0.25, "accommodation": 0.32, "food": 0.18,
                  "activities": 0.13, "local_transport": 0.05, "buffer": 0.07},
    "luxury": {"transport_to_destination": 0.20, "accommodation": 0.40, "food": 0.18,
               "activities": 0.12, "local_transport": 0.04, "buffer": 0.06},
}

UNITS = ("total", "per_person", "per_night", "per_day", "per_person_per_day", "per_person_per_night")


def _style(style: str) -> str:
    s = (style or "mid-range").strip().lower()
    if s in {"mid", "midrange", "moderate", "standard", "comfort"}:
        return "mid-range"
    if s in {"cheap", "backpacker", "backpacking", "economy"}:
        return "budget"
    return s if s in STYLE_SPLITS else "mid-range"


def allocate_budget(
    total_budget: float,
    days: int,
    travelers: int = 1,
    travel_style: str = "mid-range",
    currency: str = "USD",
    includes_transport_to_destination: bool = True,
) -> dict:
    if total_budget <= 0:
        raise ValueError("total_budget must be positive.")
    days, travelers = max(1, int(days)), max(1, int(travelers))
    nights = max(1, days - 1)
    style = _style(travel_style)
    split = dict(STYLE_SPLITS[style])
    if not includes_transport_to_destination:
        dropped = split.pop("transport_to_destination")
        scale = 1 / (1 - dropped)
        split = {k: v * scale for k, v in split.items()}

    allocation = {}
    for category, share in split.items():
        amount = round(total_budget * share, 2)
        entry = {"amount": amount, "share_pct": round(share * 100, 1)}
        if category == "accommodation":
            entry["per_night"] = round(amount / nights, 2)
            entry["per_night_note"] = f"for {travelers} traveler(s) together, {nights} night(s)"
        elif category in {"food", "activities", "local_transport"}:
            entry["per_person_per_day"] = round(amount / days / travelers, 2)
        elif category == "transport_to_destination":
            entry["per_person_round_trip"] = round(amount / travelers, 2)
        allocation[category] = entry

    per_person_per_day = total_budget / days / travelers
    return {
        "currency": currency.upper(),
        "travel_style": style,
        "days": days,
        "nights": nights,
        "travelers": travelers,
        "total_budget": round(total_budget, 2),
        "per_person_per_day_all_in": round(per_person_per_day, 2),
        "allocation": allocation,
        "guidance": (
            "These are target caps. Compare them with real prices from web_search; if real prices are "
            "higher, shift money between categories or suggest a cheaper style, and say so explicitly."
        ),
    }


def calculate_trip_cost(
    items: list[dict],
    days: int,
    travelers: int = 1,
    budget: float | None = None,
    currency: str = "USD",
) -> dict:
    days, travelers = max(1, int(days)), max(1, int(travelers))
    nights = max(1, days - 1)
    multipliers = {
        "total": 1,
        "per_person": travelers,
        "per_night": nights,
        "per_day": days,
        "per_person_per_day": travelers * days,
        "per_person_per_night": travelers * nights,
    }
    lines, by_category, warnings = [], {}, []
    for raw in items:
        unit = str(raw.get("unit", "total")).strip().lower().replace(" ", "_").replace("-", "_")
        if unit not in multipliers:
            warnings.append(f"Unknown unit '{unit}' for '{raw.get('item')}', treated as 'total'.")
            unit = "total"
        try:
            cost = float(raw.get("cost", 0))
        except (TypeError, ValueError):
            warnings.append(f"Invalid cost for '{raw.get('item')}', treated as 0.")
            cost = 0.0
        quantity = float(raw.get("quantity", 1) or 1)
        line_total = round(cost * quantity * multipliers[unit], 2)
        category = str(raw.get("category", "other")).strip().lower() or "other"
        by_category[category] = round(by_category.get(category, 0) + line_total, 2)
        lines.append({
            "category": category, "item": raw.get("item", ""), "unit_cost": cost,
            "unit": unit, "quantity": quantity, "line_total": line_total,
        })

    total = round(sum(line["line_total"] for line in lines), 2)
    result = {
        "currency": currency.upper(),
        "days": days,
        "nights": nights,
        "travelers": travelers,
        "line_items": lines,
        "totals_by_category": by_category,
        "grand_total": total,
        "per_person": round(total / travelers, 2),
        "per_person_per_day": round(total / travelers / days, 2),
    }
    if budget:
        remaining = round(budget - total, 2)
        result.update({
            "budget": budget,
            "remaining": remaining,
            "budget_used_pct": round(total / budget * 100, 1),
            "status": "within budget" if remaining >= 0 else f"OVER budget by {abs(remaining)} {currency.upper()}",
        })
    if warnings:
        result["warnings"] = warnings
    return result


TOOLS = [
    Tool(
        name="allocate_budget",
        description=(
            "Split the traveler's total budget into target amounts per category (transport, accommodation, "
            "food, activities, local transport, buffer) with per-night and per-person-per-day caps. "
            "Call this early to know what price range to search for."
        ),
        parameters={
            "type": "object",
            "properties": {
                "total_budget": {"type": "number", "description": "Total trip budget for all travelers."},
                "days": {"type": "integer", "description": "Trip length in days."},
                "travelers": {"type": "integer", "description": "Number of travelers (default 1)."},
                "travel_style": {"type": "string", "enum": list(STYLE_SPLITS)},
                "currency": {"type": "string", "description": "ISO currency code of the budget."},
                "includes_transport_to_destination": {
                    "type": "boolean",
                    "description": "False if the budget excludes getting to/from the destination.",
                },
            },
            "required": ["total_budget", "days"],
        },
        func=allocate_budget,
    ),
    Tool(
        name="calculate_trip_cost",
        description=(
            "Add up an itemized cost estimate exactly and compare it with the budget. Call this with the "
            "real prices you found before writing the final plan; use its numbers in the budget table."
        ),
        parameters={
            "type": "object",
            "properties": {
                "items": {
                    "type": "array",
                    "description": "Cost line items, all in the same currency.",
                    "items": {
                        "type": "object",
                        "properties": {
                            "category": {
                                "type": "string",
                                "description": "transport, accommodation, food, activities, local_transport, other",
                            },
                            "item": {"type": "string", "description": "e.g. 'Hostel dorm bed in Old Town'"},
                            "cost": {"type": "number", "description": "Unit cost."},
                            "unit": {"type": "string", "enum": list(UNITS)},
                            "quantity": {"type": "number", "description": "Multiplier, default 1."},
                        },
                        "required": ["category", "item", "cost", "unit"],
                    },
                },
                "days": {"type": "integer"},
                "travelers": {"type": "integer"},
                "budget": {"type": "number", "description": "Total budget to compare against."},
                "currency": {"type": "string"},
            },
            "required": ["items", "days"],
        },
        func=calculate_trip_cost,
    ),
]
