from typing import List, Literal, Optional
from pydantic import BaseModel, Field
from datetime import datetime


class CategoryDistribution(BaseModel):
    """Category distribution percentages."""
    gossip: float = Field(..., description="Gossip percentage")
    unethical: float = Field(..., description="Unethical speech percentage")
    waste: float = Field(..., description="Wasteful talk percentage")
    productive: float = Field(..., description="Productive speech percentage")


class ProgressReportResponse(BaseModel):
    """Response model for progress reports (today, week, month, lifetime)."""
    period: Literal["today", "week", "month", "lifetime"] = Field(..., description="The time period for this report")
    dateContext: str = Field(..., description="Human-readable date context (e.g., 'Jan 29', 'This week', 'January', 'Since Jan 2026')")
    totalSessions: int = Field(..., description="Total number of stopped sessions in the period")
    totalListeningMinutes: float = Field(..., description="Total listening time in minutes (rounded)")
    totalFlagged: int = Field(0, description="Total flagged interactions in the period")
    totalPositive: int = Field(0, description="Total positive interactions in the period")
    averageMinutesPerSession: float = Field(0.0, description="Average minutes per session in the period (0 if no sessions)")
    periodStart: datetime = Field(..., description="Start of the reporting period")
    periodEnd: datetime = Field(..., description="End of the reporting period (usually now)")
    categoryDistribution: Optional[CategoryDistribution] = Field(None, description="Speech category distribution percentages")


class ChartDataPoint(BaseModel):
    """A single data point for chart visualization."""
    timestamp: datetime = Field(..., description="Timestamp for this data point")
    minutes: float = Field(..., description="Listening minutes for this time period")


class ChartDataResponse(BaseModel):
    """Response model for chart data."""
    period: Literal["today", "week", "month", "lifetime"] = Field(..., description="The time period for this chart")
    points: List[ChartDataPoint] = Field(..., description="Time series data points for the chart")

