"""
Verification Decision Models

This module defines models for speaker verification decisions and results,
including chunk-level verification results and decision policies.
"""
from typing import List, Optional, Literal, Dict, Any
from pydantic import BaseModel, Field
from datetime import datetime


class VerificationDecision(BaseModel):
    """Verification decision with metadata.
    
    V1 SIMPLIFIED POLICY:
    =====================
    User-facing decision is binary: OWNER or OTHER only.
    Internal states (UNCERTAIN, SKIPPED) are kept for logging/debugging
    but mapped to OWNER/OTHER for user-facing results.
    
    - decision: User-facing decision (OWNER or OTHER)
    - internalState: Internal state for logging (OWNER, UNCERTAIN, OTHER, SKIPPED)
    """
    decision: Literal["OWNER", "OTHER"] = Field(
        ...,
        description="User-facing verification decision (binary)"
    )
    internalState: Literal["OWNER", "UNCERTAIN", "OTHER", "SKIPPED"] = Field(
        ...,
        description="Internal state for logging/debugging (includes UNCERTAIN, SKIPPED)"
    )
    maxSimilarity: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Maximum similarity score across all enrollment embeddings"
    )
    topKMean: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Mean of top-K (K=2) similarity scores"
    )
    allSimilarities: List[float] = Field(
        ...,
        description="All similarity scores (one per enrollment embedding)"
    )
    thresholdUsed: Dict[str, float] = Field(
        ...,
        description="Thresholds used for this decision (ownerThreshold, uncertainThreshold)"
    )


class ChunkVerification(BaseModel):
    """Verification result for a single audio chunk."""
    startTime: float = Field(..., description="Start time of chunk in seconds")
    endTime: float = Field(..., description="End time of chunk in seconds")
    decision: VerificationDecision = Field(..., description="Verification decision for this chunk")
    chunkIndex: int = Field(..., description="Index of this chunk in the session")


class VerificationResult(BaseModel):
    """Complete verification result for a session or chunk.
    
    V1 SIMPLIFIED POLICY:
    =====================
    - decision: User-facing binary decision (OWNER or OTHER)
    - internalStatus: Internal status for logging (includes SKIPPED, ERROR)
    - shouldProcess: Always True in v1 (verification filters, doesn't block)
    """
    status: Literal["SUCCESS", "SKIPPED", "ERROR"] = Field(
        ...,
        description="Internal verification status (for logging)"
    )
    decision: Literal["OWNER", "OTHER"] = Field(
        ...,
        description="User-facing binary decision"
    )
    internalStatus: Literal["SUCCESS", "SKIPPED", "ERROR"] = Field(
        ...,
        description="Internal status (same as status, kept for clarity)"
    )
    shouldProcess: bool = Field(
        True,
        description="Always True in v1 - verification filters, doesn't block"
    )
    
    # For chunk-level verification
    chunkVerifications: Optional[List[ChunkVerification]] = Field(
        None,
        description="Per-chunk verification results"
    )
    
    # For session-level verification (legacy)
    maxSimilarity: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Maximum similarity score (session-level)"
    )
    topKMean: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Top-K mean similarity (session-level)"
    )
    allSimilarities: Optional[List[float]] = Field(
        None,
        description="All similarity scores (session-level)"
    )
    
    # Error information
    error: Optional[str] = Field(None, description="Error message if verification failed")
    errorReason: Optional[str] = Field(None, description="Reason for failure (timeout, rate_limit, etc.)")
    retryable: Optional[bool] = Field(None, description="Whether the error is retryable")
    
    # Metadata
    modelMetadata: Optional[Dict[str, Any]] = Field(
        None,
        description="Model metadata used for verification"
    )
    verifiedAt: datetime = Field(..., description="When verification was performed")


class VerificationPolicy:
    """V1 Simplified Decision Policy for Speaker Verification.
    
    V1 POLICY (BINARY):
    ==================
    Simple, deterministic, explainable:
    - If maxSimilarity >= ownerThreshold → OWNER
    - Else → OTHER
    
    That's it. No intermediate states for users.
    
    INTERNAL STATES (for logging only):
    ===================================
    - OWNER: maxSimilarity >= ownerThreshold
    - UNCERTAIN: maxSimilarity >= uncertainThreshold but < ownerThreshold (logged, mapped to OTHER)
    - OTHER: maxSimilarity < uncertainThreshold
    - SKIPPED: Verification failed (logged, mapped to OWNER to avoid blocking)
    
    USER-FACING DECISION:
    =====================
    Always binary: OWNER or OTHER
    - OWNER: Speech is treated as user's own, included in analysis
    - OTHER: Speech is ignored for analysis (not transcribed, not counted)
    
    WHY THIS WORKS:
    ==============
    - Deterministic and explainable
    - Easier to test
    - User-friendly (no confusing intermediate states)
    - Sufficient for reflective self-improvement app
    - Verification filters text segments, doesn't block sessions
    """
    
    @staticmethod
    def apply_decision_v1(
        max_similarity: float,
        owner_threshold: float,
        uncertain_threshold: float
    ) -> tuple[Literal["OWNER", "OTHER"], Literal["OWNER", "UNCERTAIN", "OTHER", "SKIPPED"]]:
        """Apply v1 simplified binary decision policy.
        
        Returns:
            Tuple of (user_facing_decision, internal_state)
            - user_facing_decision: OWNER or OTHER (binary)
            - internal_state: OWNER, UNCERTAIN, OTHER, or SKIPPED (for logging)
        """
        # Determine internal state (for logging)
        if max_similarity >= owner_threshold:
            internal_state = "OWNER"
            user_decision = "OWNER"
        elif max_similarity >= uncertain_threshold:
            internal_state = "UNCERTAIN"  # Logged internally, but mapped to OTHER
            user_decision = "OTHER"
        else:
            internal_state = "OTHER"
            user_decision = "OTHER"
        
        return user_decision, internal_state
    
    @staticmethod
    def apply_decision(
        max_similarity: float,
        top_k_mean: float,
        owner_threshold: float,
        uncertain_threshold: float
    ) -> tuple[Literal["OWNER", "OTHER"], Literal["OWNER", "UNCERTAIN", "OTHER"]]:
        """Apply decision policy (v1 simplified - uses only max_similarity).
        
        Args:
            max_similarity: Maximum similarity across all enrollment embeddings
            top_k_mean: Mean of top-K (K=2) similarities (kept for logging, not used in v1 decision)
            owner_threshold: Threshold for OWNER classification
            uncertain_threshold: Threshold for UNCERTAIN classification (used for internal logging)
            
        Returns:
            Tuple of (user_facing_decision, internal_state)
        """
        return VerificationPolicy.apply_decision_v1(
            max_similarity,
            owner_threshold,
            uncertain_threshold
        )

