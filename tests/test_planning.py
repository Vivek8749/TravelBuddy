import json

import pytest

from travelbuddy.tools import default_registry
from travelbuddy.tools.planning import STYLE_SPLITS, allocate_budget, calculate_trip_cost


@pytest.mark.parametrize("style", list(STYLE_SPLITS))
def test_style_splits_sum_to_one(style):
    assert sum(STYLE_SPLITS[style].values()) == pytest.approx(1.0)


def test_allocate_budget_amounts_add_up():
    r = allocate_budget(2000, days=5, travelers=2, travel_style="budget", currency="eur")
    assert r["currency"] == "EUR"
    assert r["nights"] == 4
    assert sum(a["amount"] for a in r["allocation"].values()) == pytest.approx(2000, abs=0.05)
    assert r["allocation"]["accommodation"]["per_night"] == pytest.approx(2000 * 0.25 / 4)
    assert r["per_person_per_day_all_in"] == 200


def test_allocate_budget_without_travel_redistributes():
    r = allocate_budget(1000, days=3, includes_transport_to_destination=False)
    assert "transport_to_destination" not in r["allocation"]
    assert sum(a["amount"] for a in r["allocation"].values()) == pytest.approx(1000, abs=0.05)


def test_allocate_budget_unknown_style_falls_back():
    assert allocate_budget(500, days=2, travel_style="backpacker")["travel_style"] == "budget"
    assert allocate_budget(500, days=2, travel_style="whatever")["travel_style"] == "mid-range"


def test_calculate_trip_cost_units_and_budget():
    items = [
        {"category": "transport", "item": "Flight", "cost": 300, "unit": "per_person"},
        {"category": "accommodation", "item": "Hotel", "cost": 80, "unit": "per_night"},
        {"category": "food", "item": "Meals", "cost": 25, "unit": "per_person_per_day"},
        {"category": "activities", "item": "Museum", "cost": 15, "unit": "per person", "quantity": 2},
    ]
    r = calculate_trip_cost(items, days=4, travelers=2, budget=1500)
    # 600 + 80*3 + 25*2*4 + 15*2*2
    assert r["grand_total"] == 600 + 240 + 200 + 60
    assert r["totals_by_category"]["accommodation"] == 240
    assert r["remaining"] == 1500 - 1100
    assert r["status"] == "within budget"


def test_calculate_trip_cost_over_budget_and_bad_input():
    items = [
        {"category": "x", "item": "a", "cost": 100, "unit": "weekly"},
        {"category": "x", "item": "b", "cost": "oops", "unit": "total"},
    ]
    r = calculate_trip_cost(items, days=2, budget=50)
    assert r["grand_total"] == 100
    assert r["status"].startswith("OVER budget by 50")
    assert len(r["warnings"]) == 2


def test_registry_handles_errors_and_extra_args():
    reg = default_registry()
    assert json.loads(reg.run("nope", {}))["error"].startswith("Unknown tool")
    assert "Missing required" in json.loads(reg.run("allocate_budget", {"days": 3}))["error"]
    out = json.loads(reg.run("allocate_budget", {"total_budget": 900, "days": 3, "invented": True}))
    assert out["total_budget"] == 900
    assert "ValueError" in json.loads(reg.run("allocate_budget", {"total_budget": -1, "days": 3}))["error"]
