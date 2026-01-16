from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime, timezone
import random

from app.auth.dependencies import get_current_user
from app.models.voice import (
    VoiceProfile,
    StartVoiceRegistrationRequest,
    StartVoiceRegistrationResponse,
    CompleteVoiceRegistrationRequest,
    CompleteVoiceRegistrationResponse,
)
from firebase_admin import firestore

router = APIRouter(
    prefix="/voice",
    tags=["voice"],
)

# Sample texts for voice registration
SAMPLE_TEXTS = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?",
    "Peter Piper picked a peck of pickled peppers.",
    "The five boxing wizards jump quickly.",
]


def get_firestore_db():
    """Get Firestore database instance."""
    try:
        return firestore.client()
    except Exception as e:
        raise RuntimeError(f"Firestore not available: {e}")


def get_random_sample_text() -> str:
    """Get a random sample text for voice registration."""
    return random.choice(SAMPLE_TEXTS)


@router.post(
    "/register/start",
    response_model=StartVoiceRegistrationResponse,
    summary="Start voice registration",
    description="Creates a voice profile for the current user if it doesn't exist, or returns existing profile.",
    responses={
        200: {
            "description": "Voice registration started or existing profile returned",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
    },
)
def start_voice_registration(
    request: StartVoiceRegistrationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> StartVoiceRegistrationResponse:
    """Start voice registration by creating or retrieving a voice profile.
    
    If a voice profile already exists for the user, returns the existing one.
    Otherwise, creates a new profile with PENDING status.
    
    Args:
        request: Start voice registration request (no additional fields needed)
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        StartVoiceRegistrationResponse: The voice profile with sample text
    """
    try:
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Check if voice profile already exists
        doc_ref = db.collection('voice_profiles').document(uid)
        doc = doc_ref.get()
        
        if doc.exists:
            # Return existing profile
            profile_data = doc.to_dict()
            
            # Convert timestamps
            if 'createdAt' in profile_data:
                created_at = profile_data['createdAt']
                if hasattr(created_at, 'timestamp'):
                    profile_data['createdAt'] = datetime.fromtimestamp(created_at.timestamp(), tz=timezone.utc)
            
            if 'completedAt' in profile_data and profile_data['completedAt']:
                completed_at = profile_data['completedAt']
                if hasattr(completed_at, 'timestamp'):
                    profile_data['completedAt'] = datetime.fromtimestamp(completed_at.timestamp(), tz=timezone.utc)
            
            # Get or generate sample text
            sample_text = profile_data.get('sampleText') or get_random_sample_text()
            
            # Update sample text if not present
            if 'sampleText' not in profile_data:
                doc_ref.update({'sampleText': sample_text})
            
            print(f"[VOICE] Returning existing voice profile for user {uid}")
            return StartVoiceRegistrationResponse(
                uid=uid,
                status=profile_data.get('status', 'PENDING'),
                sampleText=sample_text,
            )
        
        # Create new voice profile
        created_at = datetime.now(timezone.utc)
        sample_text = get_random_sample_text()
        
        profile_data = {
            'uid': uid,
            'status': 'PENDING',
            'createdAt': created_at,
            'completedAt': None,
            'recordingsCount': None,
            'sampleText': sample_text,
        }
        
        # Store in Firestore
        doc_ref.set(profile_data)
        
        print(f"[VOICE] Created new voice profile for user {uid}")
        return StartVoiceRegistrationResponse(
            uid=uid,
            status='PENDING',
            sampleText=sample_text,
        )
        
    except Exception as e:
        print(f"[VOICE] Error starting voice registration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start voice registration"
        )


@router.post(
    "/register/complete",
    response_model=CompleteVoiceRegistrationResponse,
    summary="Complete voice registration",
    description="Marks voice registration as complete with the number of recordings submitted.",
    responses={
        200: {
            "description": "Voice registration completed successfully",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
        404: {
            "description": "Voice profile not found",
        },
        403: {
            "description": "Forbidden - Voice profile does not belong to current user",
        },
    },
)
def complete_voice_registration(
    request: CompleteVoiceRegistrationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> CompleteVoiceRegistrationResponse:
    """Complete voice registration by updating the voice profile status.
    
    Marks the profile as READY and stores the completedAt timestamp and recordingsCount.
    Only the profile owner can complete their own registration.
    
    Args:
        request: Complete voice registration request with recordingsCount
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        CompleteVoiceRegistrationResponse: The updated voice profile
        
    Raises:
        HTTPException: 404 if profile not found, 403 if user doesn't own the profile
    """
    try:
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Get voice profile document
        doc_ref = db.collection('voice_profiles').document(uid)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Voice profile not found. Please start registration first."
            )
        
        profile_data = doc.to_dict()
        
        # Enforce uid ownership (should always match, but check for safety)
        if profile_data.get('uid') != uid:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to complete this voice registration"
            )
        
        # Update profile to READY status
        completed_at = datetime.now(timezone.utc)
        
        update_data = {
            'status': 'READY',
            'completedAt': completed_at,
            'recordingsCount': request.recordingsCount,
        }
        
        doc_ref.update(update_data)
        
        # Get updated profile data
        updated_doc = doc_ref.get()
        updated_data = updated_doc.to_dict()
        
        # Convert timestamps
        if 'createdAt' in updated_data:
            created_at = updated_data['createdAt']
            if hasattr(created_at, 'timestamp'):
                updated_data['createdAt'] = datetime.fromtimestamp(created_at.timestamp(), tz=timezone.utc)
        
        if 'completedAt' in updated_data and updated_data['completedAt']:
            completed_at = updated_data['completedAt']
            if hasattr(completed_at, 'timestamp'):
                updated_data['completedAt'] = datetime.fromtimestamp(completed_at.timestamp(), tz=timezone.utc)
        
        profile = VoiceProfile(**updated_data)
        
        print(f"[VOICE] Completed voice registration for user {uid} with {request.recordingsCount} recordings")
        return CompleteVoiceRegistrationResponse(profile=profile)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[VOICE] Error completing voice registration: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to complete voice registration"
        )

