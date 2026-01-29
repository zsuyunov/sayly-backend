from typing import Optional, Literal
from pydantic import BaseModel, Field
from datetime import datetime


class ListeningSession(BaseModel):
    """Model representing a listening session.
    
    A listening session tracks when a user starts and stops listening,
    along with statistics about flagged and positive interactions.
    """
    id: str = Field(..., description="Unique session identifier")
    uid: str = Field(..., description="User ID who owns this session")
    startedAt: datetime = Field(..., description="When the session started")
    endedAt: Optional[datetime] = Field(None, description="When the session ended (null if active)")
    status: Literal["ACTIVE", "STOPPED"] = Field(..., description="Current status of the session")
    device: Literal["ios", "android", "unknown"] = Field(..., description="Device type")
    totals: dict = Field(
        default_factory=lambda: {
            "totalSeconds": 0,
            "flaggedCount": 0,
            "positiveCount": 0,
        },
        description="Session statistics"
    )
    note: Optional[str] = Field(None, description="User's reflection note for this session")
    updatedAt: Optional[datetime] = Field(None, description="When the note was last updated")
    audioUrl: Optional[str] = Field(None, description="URL/path to the audio file for this session")
    audioProcessed: bool = Field(False, description="Whether the audio has been processed")
    analysisStatus: Literal["PENDING", "PROCESSING", "COMPLETED", "FAILED"] = Field(
        "PENDING",
        description="Status of AI analysis for this session"
    )


class StartSessionRequest(BaseModel):
    """Request model for starting a session."""
    device: Literal["ios", "android", "unknown"] = Field(
        default="unknown",
        description="Device type"
    )


class StartSessionResponse(BaseModel):
    """Response model for starting a session."""
    session: ListeningSession = Field(..., description="The created or existing active session")


class StopSessionResponse(BaseModel):
    """Response model for stopping a session."""
    session: ListeningSession = Field(..., description="The stopped session")


class LastSessionResponse(BaseModel):
    """Response model for getting the last session."""
    session: Optional[ListeningSession] = Field(None, description="The most recent session, or null if none exists")


class SessionSummary(BaseModel):
    """Summary model for listing sessions (without full details)."""
    id: str = Field(..., description="Session identifier")
    startedAt: datetime = Field(..., description="When the session started")
    endedAt: Optional[datetime] = Field(None, description="When the session ended (null if active)")
    totalSeconds: int = Field(..., description="Total duration in seconds")
    flaggedCount: int = Field(..., description="Number of flagged interactions")
    positiveCount: int = Field(..., description="Number of positive interactions")
    status: Literal["ACTIVE", "STOPPED"] = Field(..., description="Current status of the session")


class SessionsListResponse(BaseModel):
    """Response model for listing all sessions."""
    sessions: list[SessionSummary] = Field(..., description="List of sessions, sorted by startedAt DESC")


class UpdateNoteRequest(BaseModel):
    """Request model for updating a session note."""
    note: str = Field(..., description="The note text (max 500 characters)", max_length=500)


class SessionDetailResponse(BaseModel):
    """Response model for getting a single session detail."""
    id: str = Field(..., description="Session identifier")
    startedAt: datetime = Field(..., description="When the session started")
    endedAt: Optional[datetime] = Field(None, description="When the session ended (null if active)")
    totalSeconds: int = Field(..., description="Total duration in seconds")
    flaggedCount: int = Field(..., description="Number of flagged interactions")
    positiveCount: int = Field(..., description="Number of positive interactions")
    status: Literal["ACTIVE", "STOPPED"] = Field(..., description="Current status of the session")
    device: Literal["ios", "android", "unknown"] = Field(..., description="Device type")
    note: Optional[str] = Field(None, description="User's reflection note for this session")
    updatedAt: Optional[datetime] = Field(None, description="When the note was last updated")
    audioUrl: Optional[str] = Field(None, description="URL/path to the audio file for this session")
    audioProcessed: bool = Field(False, description="Whether the audio has been processed")
    analysisStatus: Literal["PENDING", "PROCESSING", "COMPLETED", "FAILED"] = Field(
        "PENDING",
        description="Status of AI analysis for this session"
    )
    summary: Optional[str] = Field(None, description="AI-generated summary of the session")
    gossipScore: Optional[int] = Field(None, description="Gossip score (0-100), lower is better")

