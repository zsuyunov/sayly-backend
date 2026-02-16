from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime, timezone

from app.auth.dependencies import get_current_user
from app.models.consent import (
    PrivacyConsentResponse,
    UpdatePrivacyConsentRequest,
)
from firebase_admin import firestore

router = APIRouter(
    prefix="/privacy",
    tags=["privacy"],
)


def get_firestore_db():
    """Get Firestore database instance."""
    try:
        return firestore.client()
    except Exception as e:
        raise RuntimeError(f"Firestore not available: {e}")


@router.get(
    "",
    response_model=PrivacyConsentResponse,
    summary="Get privacy and consent preferences",
    description="Returns the current user's privacy and consent preferences. Returns default values if no record exists.",
    responses={
        200: {
            "description": "Privacy and consent preferences retrieved successfully",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
    },
)
def get_privacy_consent(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PrivacyConsentResponse:
    """Get the current user's privacy and consent preferences.
    
    If no record exists for the user, returns default values:
    - listeningEnabled = false
    - dataAnalysisEnabled = false
    - analyticsEnabled = false
    
    Args:
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        PrivacyConsentResponse: Privacy and consent preferences
        
    Raises:
        HTTPException: 401 if user is not authenticated, 500 for server errors
    """
    try:
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Get privacy document
        doc_ref = db.collection('user_privacy').document(uid)
        doc = doc_ref.get()
        
        if not doc.exists:
            # Return defaults if no record exists
            print(f"[PRIVACY] No privacy record found for user {uid}, returning defaults")
            return PrivacyConsentResponse(
                uid=uid,
                listeningEnabled=False,
                dataAnalysisEnabled=False,
                analyticsEnabled=False,
                consentGivenAt=None,
                lastUpdatedAt=None,
            )
        
        privacy_data = doc.to_dict()
        
        # Convert timestamps
        consent_given_at = privacy_data.get('consentGivenAt')
        if consent_given_at:
            if hasattr(consent_given_at, 'timestamp'):
                consent_given_at = datetime.fromtimestamp(consent_given_at.timestamp(), tz=timezone.utc)
            elif isinstance(consent_given_at, datetime):
                if consent_given_at.tzinfo is None:
                    consent_given_at = consent_given_at.replace(tzinfo=timezone.utc)
        else:
            consent_given_at = None
        
        last_updated_at = privacy_data.get('lastUpdatedAt')
        if last_updated_at:
            if hasattr(last_updated_at, 'timestamp'):
                last_updated_at = datetime.fromtimestamp(last_updated_at.timestamp(), tz=timezone.utc)
            elif isinstance(last_updated_at, datetime):
                if last_updated_at.tzinfo is None:
                    last_updated_at = last_updated_at.replace(tzinfo=timezone.utc)
        else:
            last_updated_at = None
        
        print(f"[PRIVACY] Retrieved privacy preferences for user {uid}")
        
        return PrivacyConsentResponse(
            uid=uid,
            listeningEnabled=privacy_data.get('listeningEnabled', False),
            dataAnalysisEnabled=privacy_data.get('dataAnalysisEnabled', False),
            analyticsEnabled=privacy_data.get('analyticsEnabled', False),
            consentGivenAt=consent_given_at,
            lastUpdatedAt=last_updated_at,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[PRIVACY] Error getting privacy consent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve privacy preferences"
        )


@router.put(
    "",
    response_model=PrivacyConsentResponse,
    summary="Update privacy and consent preferences",
    description="Updates the current user's privacy and consent preferences. Overwrites existing record.",
    responses={
        200: {
            "description": "Privacy and consent preferences updated successfully",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
    },
)
def update_privacy_consent(
    request: UpdatePrivacyConsentRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> PrivacyConsentResponse:
    """Update the current user's privacy and consent preferences.
    
    This endpoint:
    1. Overwrites the existing privacy record for the user
    2. Sets consentGivenAt if this is the first time consent is given
    3. Updates lastUpdatedAt timestamp
    
    Args:
        request: The privacy consent update request
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        PrivacyConsentResponse: Updated privacy and consent preferences
        
    Raises:
        HTTPException: 401 if user is not authenticated, 500 for server errors
    """
    try:
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Get existing document to check if this is first time
        doc_ref = db.collection('user_privacy').document(uid)
        doc = doc_ref.get()
        
        now = datetime.now(timezone.utc)
        
        # Prepare update data
        update_data: Dict[str, Any] = {
            'uid': uid,
            'listeningEnabled': request.listeningEnabled,
            'dataAnalysisEnabled': request.dataAnalysisEnabled,
            'analyticsEnabled': request.analyticsEnabled,
            'lastUpdatedAt': firestore.SERVER_TIMESTAMP,
        }
        
        # If document doesn't exist, set consentGivenAt
        if not doc.exists:
            update_data['consentGivenAt'] = firestore.SERVER_TIMESTAMP
            print(f"[PRIVACY] Creating new privacy record for user {uid}")
        else:
            print(f"[PRIVACY] Updating existing privacy record for user {uid}")
        
        # Save to Firestore (set overwrites entire document)
        doc_ref.set(update_data, merge=False)
        
        # Refresh document to get updated timestamps
        updated_doc = doc_ref.get()
        updated_data = updated_doc.to_dict()
        
        # Convert timestamps
        consent_given_at = updated_data.get('consentGivenAt')
        if consent_given_at:
            if hasattr(consent_given_at, 'timestamp'):
                consent_given_at = datetime.fromtimestamp(consent_given_at.timestamp(), tz=timezone.utc)
            elif isinstance(consent_given_at, datetime):
                if consent_given_at.tzinfo is None:
                    consent_given_at = consent_given_at.replace(tzinfo=timezone.utc)
        else:
            consent_given_at = None
        
        last_updated_at = updated_data.get('lastUpdatedAt')
        if last_updated_at:
            if hasattr(last_updated_at, 'timestamp'):
                last_updated_at = datetime.fromtimestamp(last_updated_at.timestamp(), tz=timezone.utc)
            elif isinstance(last_updated_at, datetime):
                if last_updated_at.tzinfo is None:
                    last_updated_at = last_updated_at.replace(tzinfo=timezone.utc)
        else:
            last_updated_at = None
        
        print(f"[PRIVACY] Updated privacy preferences for user {uid}")
        
        return PrivacyConsentResponse(
            uid=uid,
            listeningEnabled=updated_data.get('listeningEnabled', False),
            dataAnalysisEnabled=updated_data.get('dataAnalysisEnabled', False),
            analyticsEnabled=updated_data.get('analyticsEnabled', False),
            consentGivenAt=consent_given_at,
            lastUpdatedAt=last_updated_at,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[PRIVACY] Error updating privacy consent: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update privacy preferences"
        )

