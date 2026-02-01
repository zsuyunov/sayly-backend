from typing import List, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class MonthlyAverage(BaseModel):
    """Monthly average statistics."""
    year: int = Field(..., description="Year")
    month: int = Field(..., description="Month (1-12)")
    month_name: str = Field(..., description="Month name (e.g., 'January 2024')")
    average_minutes_per_day: float = Field(..., description="Average listening minutes per day in this month")
    total_sessions: int = Field(..., description="Total sessions in this month")
    total_minutes: float = Field(..., description="Total listening minutes in this month")


class LifetimeCategoryDistribution(BaseModel):
    """Category distribution for lifetime."""
    gossip: float = Field(0.0, description="Percentage of gossip speech")
    unethical: float = Field(0.0, description="Percentage of unethical speech")
    waste: float = Field(0.0, description="Percentage of waste speech")
    productive: float = Field(0.0, description="Percentage of productive speech")


class LifetimeStatsResponse(BaseModel):
    """Response model for lifetime statistics."""
    total_sessions: int = Field(..., description="Total sessions in lifetime")
    total_listening_minutes: float = Field(..., description="Total listening minutes in lifetime")
    active_days: int = Field(..., description="Number of days with at least one session")
    missed_days: int = Field(..., description="Number of days without any sessions (since signup)")
    monthly_average_series: List[MonthlyAverage] = Field(..., description="Monthly averages since signup")
    lifetime_category_distribution: LifetimeCategoryDistribution = Field(
        ..., 
        description="Category distribution percentages for lifetime"
    )
    account_created_at: datetime = Field(..., description="Account creation date")
    first_session_date: Optional[datetime] = Field(None, description="Date of first session")

