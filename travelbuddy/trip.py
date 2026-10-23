from datetime import date, timedelta

from pydantic import BaseModel, Field, field_validator

TRAVEL_STYLES = ("budget", "mid-range", "luxury")


class TripRequest(BaseModel):
    destination: str = Field(min_length=1)
    origin: str = ""
    start_date: date | None = None
    days: int = Field(ge=1, le=60)
    travelers: int = Field(default=1, ge=1, le=50)
    budget: float = Field(gt=0)
    currency: str = Field(default="USD", min_length=3, max_length=3)
    travel_style: str = "mid-range"
    budget_includes_travel: bool = True
    interests: str = ""
    notes: str = ""

    @field_validator("currency")
    @classmethod
    def _upper(cls, v: str) -> str:
        return v.strip().upper()

    @field_validator("travel_style")
    @classmethod
    def _style(cls, v: str) -> str:
        v = v.strip().lower()
        if v not in TRAVEL_STYLES:
            raise ValueError(f"travel_style must be one of {', '.join(TRAVEL_STYLES)}")
        return v

    @property
    def end_date(self) -> date | None:
        return self.start_date + timedelta(days=self.days - 1) if self.start_date else None

    def to_prompt(self) -> str:
        when = (
            f"{self.start_date.isoformat()} to {self.end_date.isoformat()} ({self.days} days)"
            if self.start_date
            else f"{self.days} days, dates flexible (suggest the best time to go)"
        )
        lines = [
            "Please plan this trip for me.",
            f"- Destination: {self.destination}",
            f"- Starting from: {self.origin or 'not specified (skip getting-there costs)'}",
            f"- When: {when}",
            f"- Travelers: {self.travelers}",
            f"- Total budget: {self.budget:,.0f} {self.currency} for everyone, "
            + ("including" if self.budget_includes_travel else "excluding")
            + " travel to and from the destination",
            f"- Travel style: {self.travel_style}",
        ]
        if self.interests:
            lines.append(f"- Interests: {self.interests}")
        if self.notes:
            lines.append(f"- Other notes: {self.notes}")
        return "\n".join(lines)
