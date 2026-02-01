from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class DailyTotal(BaseModel):
    """Daily totals for a specific date."""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    minutes: float = Field(..., description="Total listening minutes for this day")
    sessions: int = Field(..., description="Total number of sessions for this day")


class WeeklyCategoryDistribution(BaseModel):
    """Category distribution for the week."""
    gossip: float = Field(0.0, description="Percentage of gossip speech")
    unethical: float = Field(0.0, description="Percentage of unethical speech")
    waste: float = Field(0.0, description="Percentage of waste speech")
    productive: float = Field(0.0, description="Percentage of productive speech")


class WeeklyStatsResponse(BaseModel):
    """Response model for weekly statistics."""
    total_sessions_week: int = Field(..., description="Total sessions in the week")
    total_listening_minutes_week: float = Field(..., description="Total listening minutes in the week")
    daily_totals: List[DailyTotal] = Field(..., description="Daily totals for each day of the week")
    weekly_category_distribution: WeeklyCategoryDistribution = Field(
        ..., 
        description="Category distribution percentages for the week"
    )
    week_start: datetime = Field(..., description="Start of the week (Monday)")
    week_end: datetime = Field(..., description="End of the week (Sunday)")

