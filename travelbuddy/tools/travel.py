"""Travel data tools backed by free, key-less APIs (Open-Meteo, ExchangeRate-API)."""

from datetime import date, timedelta
from statistics import mean

import httpx

from .registry import Tool

GEOCODE_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
FX_URL = "https://open.er-api.com/v6/latest/{base}"
FORECAST_HORIZON_DAYS = 15

# WMO weather interpretation codes, grouped.
WEATHER_CODES = {
    0: "clear", 1: "mostly clear", 2: "partly cloudy", 3: "overcast", 45: "fog", 48: "fog",
    51: "light drizzle", 53: "drizzle", 55: "heavy drizzle", 61: "light rain", 63: "rain",
    65: "heavy rain", 71: "light snow", 73: "snow", 75: "heavy snow", 80: "rain showers",
    81: "rain showers", 82: "violent rain showers", 95: "thunderstorm", 96: "thunderstorm with hail",
    99: "thunderstorm with hail",
}


def geocode(place: str) -> dict:
    # Open-Meteo matches on the place name only, so "Kyoto, Japan" -> try "Kyoto" if needed.
    candidates = [place, place.split(",")[0].strip()]
    for name in dict.fromkeys(candidates):
        resp = httpx.get(GEOCODE_URL, params={"name": name, "count": 1}, timeout=15)
        resp.raise_for_status()
        results = resp.json().get("results")
        if results:
            r = results[0]
            return {
                "name": r["name"], "country": r.get("country", ""), "latitude": r["latitude"],
                "longitude": r["longitude"], "timezone": r.get("timezone", "auto"),
            }
    raise ValueError(f"Could not find a location named '{place}'.")


def get_weather(location: str, start_date: str | None = None, end_date: str | None = None) -> dict:
    loc = geocode(location)
    today = date.today()
    start = date.fromisoformat(start_date) if start_date else today
    end = date.fromisoformat(end_date) if end_date else start + timedelta(days=6)
    if end < start:
        start, end = end, start
    end = min(end, start + timedelta(days=30))

    daily = "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code"
    common = {
        "latitude": loc["latitude"], "longitude": loc["longitude"], "daily": daily,
        "timezone": loc["timezone"],
    }
    if start >= today and end <= today + timedelta(days=FORECAST_HORIZON_DAYS):
        kind = "forecast"
        params = {**common, "start_date": start.isoformat(), "end_date": end.isoformat()}
        resp = httpx.get(FORECAST_URL, params=params, timeout=15)
    else:
        # Too far out (or in the past) for a forecast: use the same dates last year as a climate guide.
        kind = "typical (observed on the same dates last year)"
        offset = 1
        while end.replace(year=end.year - offset) >= today - timedelta(days=5):
            offset += 1
        params = {
            **common,
            "start_date": _shift_year(start, -offset).isoformat(),
            "end_date": _shift_year(end, -offset).isoformat(),
        }
        resp = httpx.get(ARCHIVE_URL, params=params, timeout=20)
    resp.raise_for_status()
    d = resp.json()["daily"]

    highs = [v for v in d["temperature_2m_max"] if v is not None]
    lows = [v for v in d["temperature_2m_min"] if v is not None]
    rain = [v or 0 for v in d["precipitation_sum"]]
    codes = [c for c in d["weather_code"] if c is not None]
    days = [
        {
            "date": day, "high_c": hi, "low_c": lo, "rain_mm": pr,
            "conditions": WEATHER_CODES.get(code, f"code {code}"),
        }
        for day, hi, lo, pr, code in zip(d["time"], d["temperature_2m_max"], d["temperature_2m_min"],
                                         d["precipitation_sum"], d["weather_code"])
    ]
    return {
        "location": f"{loc['name']}, {loc['country']}",
        "kind": kind,
        "summary": {
            "avg_high_c": round(mean(highs), 1) if highs else None,
            "avg_low_c": round(mean(lows), 1) if lows else None,
            "rainy_days": sum(1 for r in rain if r >= 1.0),
            "total_days": len(rain),
            "most_common_conditions": WEATHER_CODES.get(mode := max(set(codes), key=codes.count), f"code {mode}")
            if codes else None,
        },
        "days": days if kind == "forecast" else days[:10],
    }


def _shift_year(d: date, years: int) -> date:
    try:
        return d.replace(year=d.year + years)
    except ValueError:  # Feb 29
        return d.replace(year=d.year + years, day=28)


def convert_currency(amount: float, from_currency: str, to_currency: str) -> dict:
    src, dst = from_currency.strip().upper(), to_currency.strip().upper()
    resp = httpx.get(FX_URL.format(base=src), timeout=15)
    resp.raise_for_status()
    data = resp.json()
    if data.get("result") != "success":
        raise ValueError(f"Unknown currency '{src}'.")
    rate = data["rates"].get(dst)
    if rate is None:
        raise ValueError(f"Unknown currency '{dst}'.")
    return {
        "amount": amount, "from": src, "to": dst, "rate": rate,
        "converted": round(float(amount) * rate, 2), "rate_date": data.get("time_last_update_utc"),
    }


TOOLS = [
    Tool(
        name="get_weather",
        description=(
            "Get weather for a destination over the travel dates. Returns a real forecast if the dates are "
            "within ~2 weeks, otherwise typical weather from the same dates last year. Use it to plan "
            "indoor/outdoor activities and the packing list."
        ),
        parameters={
            "type": "object",
            "properties": {
                "location": {"type": "string", "description": "City name, e.g. 'Kyoto' or 'Kyoto, Japan'."},
                "start_date": {"type": "string", "description": "Trip start date, YYYY-MM-DD."},
                "end_date": {"type": "string", "description": "Trip end date, YYYY-MM-DD."},
            },
            "required": ["location"],
        },
        func=get_weather,
    ),
    Tool(
        name="convert_currency",
        description=(
            "Convert an amount between currencies with live exchange rates (ISO codes like USD, EUR, INR, "
            "JPY). Use it whenever prices found online are in a different currency from the user's budget."
        ),
        parameters={
            "type": "object",
            "properties": {
                "amount": {"type": "number"},
                "from_currency": {"type": "string", "description": "ISO code, e.g. 'EUR'."},
                "to_currency": {"type": "string", "description": "ISO code, e.g. 'INR'."},
            },
            "required": ["amount", "from_currency", "to_currency"],
        },
        func=convert_currency,
    ),
]
