from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime

from app.auth.dependencies import get_current_user
from app.auth.firebase import get_firebase_auth
from firebase_admin import firestore

router = APIRouter(
    prefix="/me",
    tags=["user"],
)


def get_firestore_db():
    """Get Firestore database instance."""
    try:
        return firestore.client()
    except Exception as e:
        raise RuntimeError(f"Firestore not available: {e}")


@router.get(
    "",
    summary="Get current user profile",
    description="Returns the current authenticated user's profile including nickname and email verification status from Firestore.",
    responses={
        200: {
            "description": "Successfully retrieved user profile",
            "content": {
                "application/json": {
                    "example": {
                        "uid": "user123",
                        "email": "user@example.com",
                        "nickname": "JohnDoe",
                        "createdAt": "2026-01-16T03:00:00Z",
                        "emailVerified": True,
                    }
                }
            },
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
    },
)
def get_me(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Get the current authenticated user's profile.
    
    This endpoint:
    1. Uses the authenticated user from the auth dependency
    2. Fetches additional profile data from Firestore (nickname, createdAt, emailVerified)
    3. Returns a combined profile object
    
    Args:
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        Dict[str, Any]: User profile containing:
            - uid: User's unique identifier
            - email: User's email address
            - nickname: User's nickname from Firestore (null if not found)
            - createdAt: Account creation timestamp (ISO string)
            - emailVerified: Email verification status from Firestore (defaults to False)
    """
    try:
        uid = current_user["uid"]
        email = current_user.get("email")
        
        # Fetch user profile from Firestore
        db = get_firestore_db()
        user_doc_ref = db.collection('users').document(uid)
        user_doc = user_doc_ref.get()
        
        # Initialize response with data from auth token
        profile: Dict[str, Any] = {
            "uid": uid,
            "email": email,
            "nickname": None,
            "createdAt": None,
            "emailVerified": False,
        }
        
        # If Firestore document exists, get additional data
        if user_doc.exists:
            user_data = user_doc.to_dict()
            
            # Get nickname
            if "nickname" in user_data:
                profile["nickname"] = user_data["nickname"]
            
            # Get createdAt (handle both Timestamp and datetime)
            if "createdAt" in user_data:
                created_at = user_data["createdAt"]
                if hasattr(created_at, 'timestamp'):
                    # Firestore Timestamp
                    profile["createdAt"] = datetime.fromtimestamp(created_at.timestamp()).isoformat() + "Z"
                elif isinstance(created_at, datetime):
                    # Python datetime
                    profile["createdAt"] = created_at.isoformat() + "Z"
                else:
                    profile["createdAt"] = str(created_at)
            
            # Get emailVerified from Firestore (preferred over Firebase Auth)
            if "emailVerified" in user_data:
                profile["emailVerified"] = bool(user_data["emailVerified"])
        
        # If createdAt not found in Firestore, try to get from Firebase Auth user record
        if not profile["createdAt"]:
            try:
                auth_service = get_firebase_auth()
                firebase_user = auth_service.get_user(uid)
                if firebase_user.user_metadata and firebase_user.user_metadata.creation_timestamp:
                    # Convert milliseconds timestamp to ISO string
                    created_timestamp = firebase_user.user_metadata.creation_timestamp / 1000
                    profile["createdAt"] = datetime.fromtimestamp(created_timestamp).isoformat() + "Z"
            except Exception as e:
                # If we can't get it from Firebase Auth, leave as None
                print(f"[ME] Could not get createdAt from Firebase Auth: {e}")
        
        return profile
        
    except Exception as e:
        # Log error but return 500 instead of exposing internal errors
        print(f"[ME] Error fetching user profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve user profile"
        )

