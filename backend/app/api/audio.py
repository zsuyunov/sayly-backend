from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from datetime import datetime, timezone
import os
import uuid
import aiofiles

from app.auth.dependencies import get_current_user
from firebase_admin import firestore

router = APIRouter(
    prefix="/audio",
    tags=["audio"],
)

# Maximum file size: 50MB
MAX_FILE_SIZE = 50 * 1024 * 1024

# Allowed audio MIME types
ALLOWED_MIME_TYPES = [
    "audio/mpeg",
    "audio/mp3",
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/aac",
    "audio/m4a",
    "audio/x-m4a",
    "audio/ogg",
    "audio/webm",
]

# Directory to store uploaded audio files
AUDIO_STORAGE_DIR = os.getenv("AUDIO_STORAGE_DIR", "./audio_storage")


def get_firestore_db():
    """Get Firestore database instance."""
    try:
        return firestore.client()
    except Exception as e:
        raise RuntimeError(f"Firestore not available: {e}")


def ensure_storage_directory():
    """Ensure the audio storage directory exists."""
    os.makedirs(AUDIO_STORAGE_DIR, exist_ok=True)


@router.post(
    "/upload",
    summary="Upload audio file for a session",
    description="Uploads an audio file for a listening session. The file is stored and associated with the session.",
    responses={
        200: {
            "description": "Audio uploaded successfully",
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "audioUrl": "audio_storage/session_123_audio.mp3",
                        "message": "Audio uploaded successfully"
                    }
                }
            },
        },
        400: {
            "description": "Bad request - Invalid file or session",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
        403: {
            "description": "Forbidden - Session does not belong to current user",
        },
        404: {
            "description": "Session not found",
        },
    },
)
async def upload_audio(
    session_id: str = Form(..., description="Session ID to associate the audio with"),
    file: UploadFile = File(..., description="Audio file to upload"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Upload an audio file for a listening session.
    
    Validates the file type and size, stores it, and associates it with the session.
    Only the session owner can upload audio for their own session.
    
    Args:
        session_id: The ID of the session to associate the audio with
        file: The audio file to upload
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        Dict with success status, audioUrl, and message
        
    Raises:
        HTTPException: 400 if file is invalid, 403 if user doesn't own the session, 404 if session not found
    """
    try:
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Verify session exists and belongs to user
        session_ref = db.collection('listening_sessions').document(session_id)
        session_doc = session_ref.get()
        
        if not session_doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        session_data = session_doc.to_dict()
        if session_data.get('uid') != uid:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to upload audio for this session"
            )
        
        # Validate file type
        if file.content_type not in ALLOWED_MIME_TYPES:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid file type. Allowed types: {', '.join(ALLOWED_MIME_TYPES)}"
            )
        
        # Read file content to check size
        file_content = await file.read()
        file_size = len(file_content)
        
        if file_size > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"File too large. Maximum size is {MAX_FILE_SIZE / (1024 * 1024):.1f}MB"
            )
        
        if file_size == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="File is empty"
            )
        
        # Ensure storage directory exists
        ensure_storage_directory()
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename or "audio")[1] or ".mp3"
        unique_filename = f"{session_id}_{uuid.uuid4().hex[:8]}{file_extension}"
        file_path = os.path.join(AUDIO_STORAGE_DIR, unique_filename)
        
        # Save file
        async with aiofiles.open(file_path, 'wb') as f:
            await f.write(file_content)
        
        # Update session with audio URL
        audio_url = file_path  # Store relative path
        session_ref.update({
            'audioUrl': audio_url,
            'audioProcessed': False,
            'analysisStatus': 'PENDING',
        })
        
        print(f"[AUDIO] Uploaded audio for session {session_id}: {file_path} ({file_size} bytes)")
        
        return {
            "success": True,
            "audioUrl": audio_url,
            "message": "Audio uploaded successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[AUDIO] Error uploading audio: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to upload audio"
        )

