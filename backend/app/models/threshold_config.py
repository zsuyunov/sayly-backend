"""
Threshold Configuration Models for Speaker Verification

This module defines the configuration models for dynamic threshold management
in speaker verification. Thresholds are environment-specific and can be
calibrated based on similarity distributions.
"""
from typing import Optional, Dict, Literal, Any
from pydantic import BaseModel, Field
from datetime import datetime


class ThresholdConfig(BaseModel):
    """Configuration for speaker verification thresholds.
    
    Thresholds determine the decision boundaries for OWNER, UNCERTAIN, and OTHER
    classifications based on cosine similarity scores.
    """
    environment: Literal["dev", "test", "prod"] = Field(
        ...,
        description="Environment this configuration applies to"
    )
    ownerThreshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for OWNER classification (default: 0.75)"
    )
    uncertainThreshold: float = Field(
        ...,
        ge=0.0,
        le=1.0,
        description="Minimum similarity score for UNCERTAIN classification (default: 0.6)"
    )
    calibratedAt: datetime = Field(
        ...,
        description="When this threshold configuration was calibrated"
    )
    similarityDistribution: Optional[Dict[str, Any]] = Field(
        None,
        description="Distribution statistics for similarity scores (for calibration)"
    )
    notes: Optional[str] = Field(
        None,
        description="Notes about this threshold configuration"
    )


class ThresholdConfigResponse(BaseModel):
    """Response model for threshold configuration."""
    config: ThresholdConfig = Field(..., description="The threshold configuration")


class SimilarityDistribution(BaseModel):
    """Distribution statistics for similarity scores.
    
    Used for threshold calibration and analysis.
    """
    mean: float = Field(..., description="Mean similarity score")
    std: float = Field(..., description="Standard deviation")
    min: float = Field(..., description="Minimum observed similarity")
    max: float = Field(..., description="Maximum observed similarity")
    percentiles: Dict[str, float] = Field(
        ...,
        description="Percentiles (e.g., {'p50': 0.75, 'p95': 0.90})"
    )
    sampleCount: int = Field(..., description="Number of samples in distribution")

