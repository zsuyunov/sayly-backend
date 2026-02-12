from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from datetime import datetime, timezone
import os
import uuid
import aiofiles
import wave
import struct

from app.auth.dependencies import get_current_user
from app.services.audio_service import normalize_audio, extract_audio_metadata as extract_audio_metadata_pydub
from firebase_admin import firestore

router = APIRouter(
    prefix="/audio",
    tags=["audio"],
)

# Maximum file size: 50MB
MAX_FILE_SIZE = 50 * 1024 * 1024

# Allowed audio MIME types (WAV prioritized for AI compatibility)
ALLOWED_MIME_TYPES = [
    "audio/wav",
    "audio/wave",
    "audio/x-wav",
    "audio/mpeg",
    "audio/mp3",
    "audio/aac",
    "audio/m4a",
    "audio/x-m4a",
    "audio/ogg",
    "audio/webm",
]

# Preferred MIME type for AI processing
PREFERRED_MIME_TYPE = "audio/wav"

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


def extract_audio_metadata(file_path: str) -> Dict[str, Any]:
    """Extract audio metadata from WAV file.
    
    Args:
        file_path: Path to the audio file
        
    Returns:
        Dict with sample_rate, channels, duration_seconds, and format info
    """
    metadata = {
        'sampleRate': None,
        'channels': None,
        'durationSeconds': None,
        'format': None,
        'isWAV': False,
    }
    
    try:
        # Check if file is WAV format
        if file_path.lower().endswith('.wav'):
            metadata['isWAV'] = True
            metadata['format'] = 'WAV'
            
            # Try to read WAV file metadata
            try:
                with wave.open(file_path, 'rb') as wav_file:
                    metadata['sampleRate'] = wav_file.getframerate()
                    metadata['channels'] = wav_file.getnchannels()
                    frames = wav_file.getnframes()
                    if metadata['sampleRate'] > 0:
                        metadata['durationSeconds'] = frames / float(metadata['sampleRate'])
            except Exception as e:
                print(f"[AUDIO] Could not read WAV metadata: {e}")
                # File might be corrupted or not a valid WAV
                metadata['isWAV'] = False
        else:
            # Non-WAV format - metadata extraction would require additional libraries
            # For now, just note the format
            file_ext = os.path.splitext(file_path)[1].lower()
            metadata['format'] = file_ext[1:].upper() if file_ext else 'UNKNOWN'
            print(f"[AUDIO] Non-WAV format detected: {metadata['format']}. Metadata extraction limited.")
            
    except Exception as e:
        print(f"[AUDIO] Error extracting audio metadata: {e}")
    
    return metadata


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
        
        # Warn if non-WAV format (for AI compatibility)
        is_wav = file.content_type in ["audio/wav", "audio/wave", "audio/x-wav"]
        if not is_wav:
            print(f"[AUDIO] Warning: Non-WAV format uploaded ({file.content_type}). AI processing may require conversion.")
        
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
        
        # Normalize audio to AI-ready format (16kHz mono WAV)
        normalized_path = None
        try:
            # Create normalized file path
            normalized_filename = f"{session_id}_{uuid.uuid4().hex[:8]}_normalized.wav"
            normalized_path = os.path.join(AUDIO_STORAGE_DIR, normalized_filename)
            
            # Normalize audio
            normalize_audio(file_path, normalized_path)
            
            # Delete original file and use normalized version
            try:
                os.remove(file_path)
                file_path = normalized_path
                print(f"[AUDIO] Replaced original with normalized audio: {normalized_path}")
            except Exception as e:
                print(f"[AUDIO] Warning: Could not delete original file: {e}")
                # Keep both files if deletion fails
                
        except Exception as normalize_error:
            print(f"[AUDIO] Warning: Audio normalization failed: {normalize_error}")
            print(f"[AUDIO] Using original file without normalization")
            # Continue with original file if normalization fails
        
        # Extract audio metadata (use pydub if available, fallback to wave)
        try:
            audio_metadata = extract_audio_metadata_pydub(file_path)
        except Exception:
            # Fallback to basic metadata extraction
            audio_metadata = extract_audio_metadata(file_path)
        
        # Update session with audio URL and metadata
        audio_url = file_path  # Store relative path
        update_data = {
            'audioUrl': audio_url,
            'audioProcessed': False,
            'analysisStatus': 'PENDING',
        }
        
        # Add metadata if available
        if audio_metadata.get('sampleRate'):
            update_data['audioSampleRate'] = audio_metadata['sampleRate']
        if audio_metadata.get('channels'):
            update_data['audioChannels'] = audio_metadata['channels']
        if audio_metadata.get('durationSeconds'):
            update_data['audioDurationSeconds'] = audio_metadata['durationSeconds']
        if audio_metadata.get('format'):
            update_data['audioFormat'] = audio_metadata['format']
        
        session_ref.update(update_data)
        
        # Log metadata for debugging
        if not audio_metadata.get('isWAV'):
            print(f"[AUDIO] Warning: Non-WAV format uploaded for session {session_id}. Format: {audio_metadata.get('format')}")
        
        print(f"[AUDIO] Uploaded audio for session {session_id}: {file_path} ({file_size} bytes, {audio_metadata.get('format', 'UNKNOWN')})")
        
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

