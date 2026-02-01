from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class PerDayTotal(BaseModel):
    """Daily totals for a specific date in the month."""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    minutes: float = Field(..., description="Total listening minutes for this day")
    sessions: int = Field(..., description="Total number of sessions for this day")


class MonthlyCategoryDistribution(BaseModel):
    """Category distribution for the month."""
    gossip: float = Field(0.0, description="Percentage of gossip speech")
    unethical: float = Field(0.0, description="Percentage of unethical speech")
    waste: float = Field(0.0, description="Percentage of waste speech")
    productive: float = Field(0.0, description="Percentage of productive speech")


class MonthlyStatsResponse(BaseModel):
    """Response model for monthly statistics."""
    total_sessions_month: int = Field(..., description="Total sessions in the month")
    total_listening_minutes_month: float = Field(..., description="Total listening minutes in the month")
    per_day_totals: List[PerDayTotal] = Field(..., description="Daily totals for each day of the month")
    month_category_distribution: MonthlyCategoryDistribution = Field(
        ..., 
        description="Category distribution percentages for the month"
    )
    month_start: datetime = Field(..., description="Start of the month")
    month_end: datetime = Field(..., description="End of the month")
    month_name: str = Field(..., description="Month name (e.g., 'January 2024')")


class MonthlyComparisonResponse(BaseModel):
    """Response model for comparing current month with previous month."""
    current_month: MonthlyStatsResponse
    previous_month: Optional[MonthlyStatsResponse] = Field(None, description="Previous month stats if available")
    category_trend: dict = Field(..., description="Trend comparison for each category")

