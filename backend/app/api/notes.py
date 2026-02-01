from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime, timezone
from pydantic import BaseModel, Field

from app.auth.dependencies import get_current_user
from firebase_admin import firestore

router = APIRouter(
    prefix="/notes",
    tags=["notes"],
)


def get_firestore_db():
    """Get Firestore database instance."""
    try:
        return firestore.client()
    except Exception as e:
        raise RuntimeError(f"Firestore not available: {e}")


class DailyNoteResponse(BaseModel):
    """Response model for daily note."""
    date: str = Field(..., description="Date in YYYY-MM-DD format")
    notes: Optional[str] = Field(None, description="Note text for this date")
    createdAt: Optional[datetime] = Field(None, description="When the note was created")
    updatedAt: Optional[datetime] = Field(None, description="When the note was last updated")


class SaveDailyNoteRequest(BaseModel):
    """Request model for saving a daily note."""
    notes: str = Field(..., description="Note text (max 2000 characters)", max_length=2000)


@router.get(
    "/daily/{date}",
    response_model=DailyNoteResponse,
    summary="Get daily note",
    description="Returns the daily note for a specific date (YYYY-MM-DD format).",
    responses={
        200: {
            "description": "Daily note retrieved successfully",
        },
        400: {
            "description": "Invalid date format",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
    },
)
def get_daily_note(
    date: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> DailyNoteResponse:
    """Get daily note for a specific date.
    
    Args:
        date: Date in YYYY-MM-DD format
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        DailyNoteResponse: The daily note for the specified date (notes may be None if not set)
    """
    try:
        # Validate date format
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format. Must be YYYY-MM-DD"
            )
        
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Document ID format: {userId}_{date}
        doc_id = f"{uid}_{date}"
        doc_ref = db.collection('daily_notes').document(doc_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            # Return empty note
            return DailyNoteResponse(
                date=date,
                notes=None,
                createdAt=None,
                updatedAt=None,
            )
        
        note_data = doc.to_dict()
        
        # Convert timestamps
        created_at = note_data.get('createdAt')
        if created_at and hasattr(created_at, 'timestamp'):
            created_at = datetime.fromtimestamp(created_at.timestamp(), tz=timezone.utc)
        elif created_at and isinstance(created_at, datetime):
            if created_at.tzinfo is None:
                created_at = created_at.replace(tzinfo=timezone.utc)
        else:
            created_at = None
        
        updated_at = note_data.get('updatedAt')
        if updated_at and hasattr(updated_at, 'timestamp'):
            updated_at = datetime.fromtimestamp(updated_at.timestamp(), tz=timezone.utc)
        elif updated_at and isinstance(updated_at, datetime):
            if updated_at.tzinfo is None:
                updated_at = updated_at.replace(tzinfo=timezone.utc)
        else:
            updated_at = None
        
        return DailyNoteResponse(
            date=date,
            notes=note_data.get('notes'),
            createdAt=created_at,
            updatedAt=updated_at,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[NOTES] Error getting daily note: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve daily note"
        )


@router.put(
    "/daily/{date}",
    response_model=DailyNoteResponse,
    summary="Save daily note",
    description="Saves or updates the daily note for a specific date (YYYY-MM-DD format). Only one note per day is allowed.",
    responses={
        200: {
            "description": "Daily note saved successfully",
        },
        400: {
            "description": "Invalid date format or note too long",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
    },
)
def save_daily_note(
    date: str,
    request: SaveDailyNoteRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> DailyNoteResponse:
    """Save or update daily note for a specific date.
    
    Only one note per day is allowed. Updates overwrite previous note.
    
    Args:
        date: Date in YYYY-MM-DD format
        request: The note text to save
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        DailyNoteResponse: The saved daily note
    """
    try:
        # Validate date format
        try:
            datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid date format. Must be YYYY-MM-DD"
            )
        
        # Validate note length
        note_text = request.notes.strip() if request.notes else ""
        if len(note_text) > 2000:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Note exceeds maximum length of 2000 characters"
            )
        
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Document ID format: {userId}_{date}
        doc_id = f"{uid}_{date}"
        doc_ref = db.collection('daily_notes').document(doc_id)
        
        # Check if note already exists
        existing_doc = doc_ref.get()
        now = datetime.now(timezone.utc)
        
        if existing_doc.exists:
            # Update existing note
            update_data = {
                'notes': note_text if note_text else None,
                'updatedAt': now,
            }
            doc_ref.update(update_data)
            
            # Get created_at from existing doc
            existing_data = existing_doc.to_dict()
            created_at = existing_data.get('createdAt')
            if created_at and hasattr(created_at, 'timestamp'):
                created_at = datetime.fromtimestamp(created_at.timestamp(), tz=timezone.utc)
            elif created_at and isinstance(created_at, datetime):
                if created_at.tzinfo is None:
                    created_at = created_at.replace(tzinfo=timezone.utc)
            else:
                created_at = now
        else:
            # Create new note
            note_data = {
                'userId': uid,
                'date': date,
                'notes': note_text if note_text else None,
                'createdAt': now,
                'updatedAt': now,
            }
            doc_ref.set(note_data)
            created_at = now
        
        print(f"[NOTES] Saved daily note for user {uid}, date {date}")
        
        return DailyNoteResponse(
            date=date,
            notes=note_text if note_text else None,
            createdAt=created_at,
            updatedAt=now,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[NOTES] Error saving daily note: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to save daily note"
        )

