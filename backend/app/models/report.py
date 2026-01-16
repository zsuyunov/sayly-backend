from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


class WeeklyReportResponse(BaseModel):
    """Response model for weekly report."""
    weekStart: datetime = Field(..., description="Start of the reporting period (7 days ago)")
    weekEnd: datetime = Field(..., description="End of the reporting period (now)")
    totalSessions: int = Field(..., description="Total number of stopped sessions in the period")
    totalListeningMinutes: float = Field(..., description="Total listening time in minutes")
    totalFlagged: int = Field(..., description="Total flagged interactions")
    totalPositive: int = Field(..., description="Total positive interactions")
    averageMinutesPerSession: float = Field(..., description="Average minutes per session (0 if no sessions)")
    productivityScore: int = Field(..., description="Productivity score from 0-100 based on positive vs flagged interactions")


class MonthlyReportResponse(BaseModel):
    """Response model for monthly report."""
    monthStart: datetime = Field(..., description="Start of the reporting period (30 days ago)")
    monthEnd: datetime = Field(..., description="End of the reporting period (now)")
    totalSessions: int = Field(..., description="Total number of stopped sessions in the period")
    totalListeningMinutes: float = Field(..., description="Total listening time in minutes")
    totalFlagged: int = Field(..., description="Total flagged interactions")
    totalPositive: int = Field(..., description="Total positive interactions")
    averageMinutesPerSession: float = Field(..., description="Average minutes per session (0 if no sessions)")
    bestDay: Optional[datetime] = Field(None, description="Day with highest positiveCount (null if no sessions)")

