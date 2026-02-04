from typing import Optional, Literal, List
from pydantic import BaseModel, Field
from datetime import datetime


class EnrollmentEmbeddingMetadata(BaseModel):
    """Metadata for a single enrollment embedding."""
    index: int = Field(..., description="Index of this enrollment sample (0, 1, or 2)")
    extractedAt: datetime = Field(..., description="When this embedding was extracted")
    similarityToOthers: Optional[List[float]] = Field(
        None,
        description="Similarity scores to other enrollment embeddings (for quality check)"
    )


class VoiceProfile(BaseModel):
    """Model representing a voice profile.
    
    A voice profile tracks the registration status and metadata
    for a user's voice samples. Now stores individual embeddings
    instead of a single averaged embedding for better verification accuracy.
    """
    uid: str = Field(..., description="User ID who owns this voice profile")
    status: Literal["PENDING", "READY"] = Field(..., description="Registration status")
    createdAt: datetime = Field(..., description="When the profile was created")
    completedAt: Optional[datetime] = Field(None, description="When registration was completed")
    recordingsCount: Optional[int] = Field(None, description="Number of recordings submitted")
    sampleText: Optional[str] = Field(None, description="Sample text used for recording")
    
    # Legacy field (for backward compatibility)
    voiceEmbedding: Optional[List[float]] = Field(
        None,
        description="Legacy: Single averaged embedding vector (deprecated, use enrollmentEmbeddings)"
    )
    
    # New fields for individual embeddings
    enrollmentEmbeddings: Optional[List[List[float]]] = Field(
        None,
        description="Array of 3 individual enrollment embeddings (not averaged)"
    )
    enrollmentMetadata: Optional[List[EnrollmentEmbeddingMetadata]] = Field(
        None,
        description="Metadata for each enrollment embedding"
    )
    
    # Model versioning
    model: Optional[str] = Field(None, description="Model ID used for embedding extraction")
    modelRevision: Optional[str] = Field(None, description="Model revision/commit hash")
    modelVersion: Optional[str] = Field(None, description="Internal model version")
    registeredAt: Optional[datetime] = Field(None, description="When voice was registered with embedding")


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


class EnrollVoiceResponse(BaseModel):
    """Response model for voice enrollment."""
    success: bool = Field(..., description="Whether enrollment was successful")
    message: str = Field(..., description="Success or error message")
    registeredAt: Optional[datetime] = Field(None, description="When voice was registered")


class VoiceVerificationRequest(BaseModel):
    """Request model for voice verification."""
    sessionAudioEmbedding: List[float] = Field(..., description="Embedding vector from session audio")


class VoiceVerificationResponse(BaseModel):
    """Response model for voice verification (v1: binary decision).
    
    V1 SIMPLIFIED POLICY:
    =====================
    User-facing result is binary: OWNER or OTHER only.
    Internal states (UNCERTAIN, SKIPPED) are kept for logging but mapped to OWNER/OTHER.
    """
    result: Literal["OWNER", "OTHER"] = Field(..., description="User-facing verification result (binary)")
    score: Optional[float] = Field(None, description="Cosine similarity score (0.0 to 1.0), None if verification failed")
    internalState: Optional[Literal["OWNER", "UNCERTAIN", "OTHER", "SKIPPED"]] = Field(
        None, 
        description="Internal state for logging/debugging (not exposed to users)"
    )


class VoiceStatusResponse(BaseModel):
    """Response model for voice registration status."""
    isRegistered: bool = Field(..., description="Whether voice is registered")
    registeredAt: Optional[datetime] = Field(None, description="When voice was registered")
    model: Optional[str] = Field(None, description="Model used for embedding")

