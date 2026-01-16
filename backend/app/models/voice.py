from typing import Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class VoiceProfile(BaseModel):
    """Model representing a voice profile.
    
    A voice profile tracks the registration status and metadata
    for a user's voice samples.
    """
    uid: str = Field(..., description="User ID who owns this voice profile")
    status: Literal["PENDING", "READY"] = Field(..., description="Registration status")
    createdAt: datetime = Field(..., description="When the profile was created")
    completedAt: Optional[datetime] = Field(None, description="When registration was completed")
    recordingsCount: Optional[int] = Field(None, description="Number of recordings submitted")
    sampleText: Optional[str] = Field(None, description="Sample text used for recording")


class StartVoiceRegistrationRequest(BaseModel):
    """Request model for starting voice registration."""
    pass  # No additional fields needed, uid comes from auth


class StartVoiceRegistrationResponse(BaseModel):
    """Response model for starting voice registration."""
    uid: str = Field(..., description="User ID")
    status: Literal["PENDING", "READY"] = Field(..., description="Registration status")
    sampleText: str = Field(..., description="Sample text to record")


class CompleteVoiceRegistrationRequest(BaseModel):
    """Request model for completing voice registration."""
    recordingsCount: int = Field(..., ge=1, description="Number of recordings submitted")


class CompleteVoiceRegistrationResponse(BaseModel):
    """Response model for completing voice registration."""
    profile: VoiceProfile = Field(..., description="The updated voice profile")

