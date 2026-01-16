from typing import Optional
from pydantic import BaseModel, Field
from datetime import datetime


class UserPrivacyConsent(BaseModel):
    """Model representing a user's privacy and consent preferences.
    
    This model stores user consent for various data processing activities.
    All fields are stored in Firestore collection: user_privacy
    """
    uid: str = Field(..., description="User ID who owns this consent record")
    listeningEnabled: bool = Field(..., description="Consent for audio listening/recording")
    dataAnalysisEnabled: bool = Field(..., description="Consent for data analysis and processing")
    analyticsEnabled: bool = Field(..., description="Consent for analytics and usage tracking")
    consentGivenAt: datetime = Field(..., description="When the consent was first given")
    lastUpdatedAt: datetime = Field(..., description="When the consent was last updated")


class PrivacyConsentResponse(BaseModel):
    """Response model for privacy and consent data."""
    uid: str = Field(..., description="User ID")
    listeningEnabled: bool = Field(..., description="Consent for audio listening/recording")
    dataAnalysisEnabled: bool = Field(..., description="Consent for data analysis and processing")
    analyticsEnabled: bool = Field(..., description="Consent for analytics and usage tracking")
    consentGivenAt: Optional[datetime] = Field(None, description="When the consent was first given (null if never set)")
    lastUpdatedAt: Optional[datetime] = Field(None, description="When the consent was last updated (null if never set)")


class UpdatePrivacyConsentRequest(BaseModel):
    """Request model for updating privacy and consent preferences."""
    listeningEnabled: bool = Field(..., description="Consent for audio listening/recording")
    dataAnalysisEnabled: bool = Field(..., description="Consent for data analysis and processing")
    analyticsEnabled: bool = Field(..., description="Consent for analytics and usage tracking")

