from pydantic import BaseModel, Field
from typing import List


class RewardsStatusResponse(BaseModel):
    """Response model for rewards status."""
    currentStreak: int = Field(..., description="Current consecutive days with at least one stopped session")
    bestStreak: int = Field(..., description="Maximum historical streak of consecutive days")
    earnedBadges: List[str] = Field(..., description="List of badge names that have been earned")
    availableBadges: List[str] = Field(..., description="List of badge names that are available but not yet earned")

