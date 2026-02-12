from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status, UploadFile, File, Form
from datetime import datetime, timezone
import random
import os
import tempfile
import numpy as np

from app.auth.dependencies import get_current_user
from app.models.voice import (
    VoiceProfile,
    StartVoiceRegistrationRequest,
    StartVoiceRegistrationResponse,
    CompleteVoiceRegistrationRequest,
    CompleteVoiceRegistrationResponse,
    EnrollVoiceResponse,
    VoiceVerificationRequest,
    VoiceVerificationResponse,
    VoiceStatusResponse,
)
from app.services.huggingface_service import extract_speaker_embedding
from app.services.audio_quality_service import validate_audio_quality, validate_enrollment_audio
from app.services.model_versioning_service import get_current_model_metadata
from app.services.verification_service import verify_speaker, cosine_similarity
from app.models.voice import EnrollmentEmbeddingMetadata
from firebase_admin import firestore

router = APIRouter(
    prefix="/voice",
    tags=["voice"],
)

# Fixed prompts for voice registration (users must say exactly these texts)
# Prompt 1 (neutral identity)
# Prompt 2 (natural continuous speech)
# Prompt 3 (longer, varied phonetics)
SAMPLE_TEXTS = [
    "I am registering my voice so the application can recognize me during my reflection sessions.",
    "Today I am speaking clearly and naturally. This recording helps the system learn my speaking style.",
    "Reflection helps me become more mindful of my words, actions, and intentions over time.",
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


# cosine_similarity moved to app.services.verification_service to avoid circular imports


@router.post(
    "/validate",
    summary="Validate single audio file quality",
    description="Validate a single audio file for quality before enrollment. Returns user-friendly error messages.",
    responses={
        200: {
            "description": "Audio quality validation result",
        },
        400: {
            "description": "Bad request - Invalid file or quality check failed",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
    },
)
async def validate_audio_file(
    audio_file: UploadFile = File(..., description="Single WAV audio file to validate"),
    current_user: Dict[str, Any] = Depends(get_current_user)
):
    """Validate a single audio file for quality.
    
    This endpoint allows validating audio quality immediately after recording,
    before adding it to the enrollment set. This provides immediate feedback
    to users so they can retry a specific recording if needed.
    
    Args:
        audio_file: Single audio file to validate
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        Dict with validation result and user-friendly message
    """
    temp_file = None
    normalized_file = None
    try:
        # Create temporary file (preserve original extension)
        file_extension = os.path.splitext(audio_file.filename or 'audio.wav')[1] or '.wav'
        temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
        
        # Read and save file content
        content = await audio_file.read()
        if len(content) == 0:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Audio file is empty"
            )
        
        temp_file.write(content)
        temp_file.close()
        
        # Normalize audio to 16kHz WAV before validation (handles format conversion)
        # This allows Android M4A files to be converted to WAV at correct sample rate
        normalized_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
        normalized_file.close()
        
        try:
            from app.services.audio_service import normalize_audio
            normalize_audio(temp_file.name, normalized_file.name)
            validation_file = normalized_file.name
        except Exception as normalize_error:
            # If normalization fails, try validating original file
            print(f"[VOICE] Normalization failed, using original file: {normalize_error}")
            validation_file = temp_file.name
        
        # Validate audio quality (now guaranteed to be WAV at 16kHz if normalization succeeded)
        quality_result = validate_audio_quality(validation_file)
        
        if quality_result.status == "FAIL":
            # Return user-friendly error message
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=quality_result.message
            )
        
        # Return success with metrics
        return {
            "valid": True,
            "message": "Audio quality is good!",
            "metrics": quality_result.metrics
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[VOICE] Error validating audio file: {e}")
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Could not analyze audio file. Please try recording again."
        )
    finally:
        # Clean up temporary files
        if temp_file and os.path.exists(temp_file.name):
            try:
                os.unlink(temp_file.name)
            except Exception as e:
                print(f"[VOICE] Error deleting temp file: {e}")
        if normalized_file and os.path.exists(normalized_file.name):
            try:
                os.unlink(normalized_file.name)
            except Exception as e:
                print(f"[VOICE] Error deleting normalized file: {e}")


@router.post(
    "/enroll",
    response_model=EnrollVoiceResponse,
    summary="Enroll voice with 3 recordings",
    description="Upload 3 voice recordings, validate quality, extract embeddings, and store individually in Firestore.",
    responses={
        200: {
            "description": "Voice enrolled successfully",
        },
        400: {
            "description": "Bad request - Invalid files or format",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
    },
)
async def enroll_voice(
    audio_files: List[UploadFile] = File(..., description="3 WAV audio files (16kHz mono)"),
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> EnrollVoiceResponse:
    """Enroll user's voice by processing 3 audio recordings.
    
    SECURITY & PRIVACY:
    ===================
    - Requires authentication (get_current_user dependency)
    - Embeddings stored with uid as document ID (user-scoped)
    - Audio files deleted immediately after embedding extraction
    - Quality validation prevents poor enrollments
    - Model versioning tracked for compatibility
    
    PROCESSING FLOW:
    ================
    1. Validates that exactly 3 WAV files are provided
    2. For each file:
       a. Validates audio quality (duration ≥8s, silence ≤30%, RMS threshold)
       b. Extracts speaker embedding via Hugging Face API
       c. Stores embedding individually (NOT averaged)
    3. Computes inter-enrollment similarities (quality check)
    4. Stores all 3 embeddings + metadata in Firestore
    5. Deletes temporary audio files
    
    STORAGE:
    ========
    - enrollmentEmbeddings: List of 3 individual embeddings
    - enrollmentMetadata: Per-embedding metadata with similarities
    - Model versioning info (modelId, revision, version)
    - Legacy voiceEmbedding: Averaged embedding (for backward compatibility)
    
    Args:
        audio_files: List of 3 audio files (must be WAV format, 16kHz mono)
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        EnrollVoiceResponse: Success status and message
        
    Raises:
        HTTPException: 400 if invalid files or quality check fails, 401 if unauthorized
    """
    uid = current_user["uid"]
    temp_files = []
    
    try:
        # Validate exactly 3 files
        if len(audio_files) != 3:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Exactly 3 audio files are required for voice enrollment"
            )
        
        # Validate file formats
        for i, file in enumerate(audio_files):
            if not file.filename:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File {i+1} has no filename"
                )
            
            # Accept both WAV and M4A files (M4A will be normalized to WAV)
            file_ext = file.filename.lower().split('.')[-1] if '.' in file.filename else ''
            if file_ext not in ['wav', 'm4a']:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File {i+1} must be a WAV or M4A file. Got: {file.filename}"
                )
            
            # Check content type
            if file.content_type and 'audio' not in file.content_type.lower():
                print(f"[VOICE] Warning: File {i+1} has unexpected content type: {file.content_type}")
        
        print(f"[VOICE] Starting voice enrollment for user {uid}")
        
        # Get current model metadata
        model_metadata = get_current_model_metadata()
        
        # Save files temporarily, validate quality, and extract embeddings
        embeddings = []
        enrollment_metadata_list = []
        registered_at = datetime.now(timezone.utc)
        
        for i, file in enumerate(audio_files):
            # Create temporary file (preserve original extension for M4A files)
            file_extension = os.path.splitext(file.filename or 'audio.wav')[1] or '.wav'
            temp_file = tempfile.NamedTemporaryFile(delete=False, suffix=file_extension)
            temp_files.append(temp_file.name)
            
            # Read and save file content
            content = await file.read()
            if len(content) == 0:
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=f"File {i+1} is empty"
                )
            
            temp_file.write(content)
            temp_file.close()
            
            # Normalize audio to 16kHz WAV before validation (handles format conversion)
            # This allows Android M4A files to be converted to WAV at correct sample rate
            normalized_file = tempfile.NamedTemporaryFile(delete=False, suffix='.wav')
            normalized_file.close()
            temp_files.append(normalized_file.name)  # Track for cleanup
            
            validation_file = temp_file.name
            try:
                from app.services.audio_service import normalize_audio
                normalize_audio(temp_file.name, normalized_file.name)
                validation_file = normalized_file.name
                print(f"[VOICE] Normalized file {i+1} from {file_extension} to WAV at 16kHz")
            except Exception as normalize_error:
                # If normalization fails, try validating original file
                print(f"[VOICE] Normalization failed for file {i+1}, using original file: {normalize_error}")
                validation_file = temp_file.name
            
            # Validate audio quality BEFORE extracting embedding or storing
            print(f"[VOICE] Validating audio quality for file {i+1}/{len(audio_files)}")
            quality_result = validate_audio_quality(validation_file)
            
            if quality_result.status == "FAIL":
                # Build detailed error message
                error_detail = f"File {i+1} failed quality validation: {quality_result.message}"
                if quality_result.reasons:
                    error_detail += f" (Reasons: {', '.join(quality_result.reasons)})"
                
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail=error_detail
                )
            
            # Store quality metrics for passed recordings (for audit/debugging)
            print(f"[VOICE] File {i+1} passed quality validation: duration={quality_result.metrics.get('durationSeconds', 0):.1f}s, silence={quality_result.metrics.get('silenceRatio', 0)*100:.1f}%, RMS={quality_result.metrics.get('rms', 0):.0f}, clipping={quality_result.metrics.get('clippingRatio', 0)*100:.1f}%")
            
            # Extract embedding (use normalized file if available, otherwise original)
            print(f"[VOICE] Extracting embedding from file {i+1}/{len(audio_files)}")
            try:
                embedding = extract_speaker_embedding(validation_file)
                embeddings.append(embedding)
                print(f"[VOICE] Extracted embedding {i+1} (dimension: {len(embedding)})")
            except ValueError as e:
                # API key not set
                error_message = str(e)
                print(f"[VOICE] Configuration error for file {i+1}: {error_message}")
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail="Voice processing service is not configured. Please contact support."
                )
            except Exception as e:
                error_message = str(e)
                print(f"[VOICE] Error extracting embedding from file {i+1}: {error_message}")
                
                # Provide user-friendly error messages
                if (
                    "API key" in error_message
                    or "invalid" in error_message.lower()
                    or "received html error page" in error_message.lower()
                    or "html" in error_message.lower()
                    or "not configured" in error_message.lower()
                    or "hugging face api error" in error_message.lower()
                ):
                    user_message = (
                        "Voice processing service is unavailable (speaker embedding extraction failed). "
                        "If you're the developer: check Hugging Face billing/inference access and model access, "
                        "then restart the backend."
                    )
                elif "timeout" in error_message.lower():
                    user_message = "Voice processing timed out. Please try again."
                elif "network" in error_message.lower():
                    user_message = "Network error during voice processing. Please check your connection and try again."
                else:
                    user_message = f"Failed to process recording {i+1}. Please try recording again."
                
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=user_message
                )
        
        # Validate embeddings
        if len(embeddings) == 0:
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail="No embeddings extracted"
            )
        
        # Ensure all embeddings have the same dimension
        embedding_dim = len(embeddings[0])
        for i, emb in enumerate(embeddings):
            if len(emb) != embedding_dim:
                raise HTTPException(
                    status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                    detail=f"Embedding dimension mismatch: file 1 has {embedding_dim}, file {i+1} has {len(emb)}"
                )
        
        # Compute similarity between enrollment embeddings (for quality check)
        print(f"[VOICE] Computing inter-enrollment similarities for quality check")
        for i, emb1 in enumerate(embeddings):
            similarities_to_others = []
            for j, emb2 in enumerate(embeddings):
                if i != j:
                    sim = cosine_similarity(emb1, emb2)
                    similarities_to_others.append(sim)
            
            # Create enrollment metadata
            enrollment_metadata_list.append({
                'index': i,
                'extractedAt': registered_at,
                'similarityToOthers': similarities_to_others,
            })
        
        # Store individual embeddings (NOT averaged) in Firestore
        db = get_firestore_db()
        doc_ref = db.collection('voice_profiles').document(uid)
        
        # Get or create profile
        doc = doc_ref.get()
        if doc.exists:
            profile_data = doc.to_dict()
        else:
            profile_data = {
                'uid': uid,
                'status': 'PENDING',
                'createdAt': registered_at,
                'sampleText': get_random_sample_text(),
            }
        
        # Update with individual embeddings and metadata
        update_data = {
            # Store all 3 individual embeddings (not averaged)
            'enrollmentEmbeddings': [emb.tolist() if isinstance(emb, np.ndarray) else emb for emb in embeddings],
            'enrollmentMetadata': enrollment_metadata_list,
            
            # Model versioning
            'model': model_metadata.model_id,
            'modelRevision': model_metadata.model_revision,
            'modelVersion': model_metadata.internal_version,
            
            # Legacy field (for backward compatibility) - compute average
            'voiceEmbedding': np.mean(embeddings, axis=0).tolist(),
            
            'registeredAt': registered_at,
            'status': 'READY',
            'completedAt': registered_at,
            'recordingsCount': 3,
        }
        
        doc_ref.set(profile_data, merge=True)
        doc_ref.update(update_data)
        
        print(f"[VOICE] Successfully enrolled voice for user {uid} with {len(embeddings)} individual embeddings")
        
        return EnrollVoiceResponse(
            success=True,
            message="Voice enrolled successfully",
            registeredAt=registered_at
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[VOICE] Error enrolling voice: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to enroll voice: {str(e)}"
        )
    finally:
        # Clean up temporary files
        for temp_file_path in temp_files:
            try:
                if os.path.exists(temp_file_path):
                    os.remove(temp_file_path)
                    print(f"[VOICE] Deleted temporary file: {temp_file_path}")
            except Exception as e:
                print(f"[VOICE] Warning: Could not delete temporary file {temp_file_path}: {e}")


@router.post(
    "/verify",
    response_model=VoiceVerificationResponse,
    summary="Verify speaker identity",
    description="Compare session audio embedding with stored user embedding.",
    responses={
        200: {
            "description": "Verification result",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
        404: {
            "description": "Voice profile not found - user has not enrolled",
        },
    },
)
def verify_voice(
    request: VoiceVerificationRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> VoiceVerificationResponse:
    """Verify if session audio matches the registered user's voice.
    
    SECURITY:
    =========
    - Requires authentication (get_current_user dependency)
    - User-scoped: Only accesses current user's embeddings
    - Uses dynamic thresholds (no hardcoded values)
    - Logs similarity scores for audit and calibration
    
    VERIFICATION LOGIC:
    ===================
    1. Retrieves user's enrollment embeddings (all 3 individually)
    2. Compares session embedding against each enrollment embedding
    3. Computes:
       - Max similarity: Best match across all enrollments
       - Top-K mean: Mean of top 2 similarities (K=2)
    4. Applies decision policy using dynamic thresholds:
       - OWNER: Both metrics >= ownerThreshold
       - UNCERTAIN: At least one >= uncertainThreshold
       - OTHER: Both < uncertainThreshold
    5. Returns decision + similarity scores
    
    Args:
        request: Voice verification request with session audio embedding
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        VoiceVerificationResponse: Verification result and similarity score
        
    Raises:
        HTTPException: 404 if user has not enrolled voice
    """
    try:
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Get user's voice profile
        doc_ref = db.collection('voice_profiles').document(uid)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Voice profile not found. Please enroll your voice first."
            )
        
        profile_data = doc.to_dict()
        
        # Get enrollment embeddings (prefer new format, fallback to legacy)
        enrollment_embeddings = profile_data.get('enrollmentEmbeddings')
        if not enrollment_embeddings:
            # Fallback to legacy single embedding
            stored_embedding = profile_data.get('voiceEmbedding')
            if not stored_embedding:
                raise HTTPException(
                    status_code=status.HTTP_404_NOT_FOUND,
                    detail="Voice embedding not found. Please re-enroll your voice."
                )
            # Convert legacy single embedding to list format
            enrollment_embeddings = [stored_embedding]
        
        # Use verification service (handles multi-embedding comparison and dynamic thresholds)
        decision, warning = verify_speaker(
            request.sessionAudioEmbedding,
            enrollment_embeddings,
            environment=None,  # Will use default environment
            uid=uid
        )
        
        if warning:
            print(f"[VOICE] Verification warning for user {uid}: {warning}")
        
        print(f"[VOICE] Verification for user {uid}: {decision.decision} (internal={decision.internalState}, max_sim={decision.maxSimilarity:.3f}, topK_mean={decision.topKMean:.3f})")
        
        # Return verification result (v1: binary decision)
        return VoiceVerificationResponse(
            result=decision.decision,  # User-facing: OWNER or OTHER
            score=decision.maxSimilarity if decision.maxSimilarity > 0 else None,
            internalState=decision.internalState  # Internal state for logging
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[VOICE] Error verifying voice: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to verify voice: {str(e)}"
        )


@router.get(
    "/status",
    response_model=VoiceStatusResponse,
    summary="Get voice registration status",
    description="Check if the current user has registered their voice.",
    responses={
        200: {
            "description": "Voice registration status",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
    },
)
def get_voice_status(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> VoiceStatusResponse:
    """Get voice registration status for the current user.
    
    Args:
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        VoiceStatusResponse: Registration status and metadata
    """
    try:
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Get voice profile
        doc_ref = db.collection('voice_profiles').document(uid)
        doc = doc_ref.get()
        
        if not doc.exists:
            return VoiceStatusResponse(
                isRegistered=False,
                registeredAt=None,
                model=None
            )
        
        profile_data = doc.to_dict()
        voice_embedding = profile_data.get('voiceEmbedding')
        
        if not voice_embedding:
            return VoiceStatusResponse(
                isRegistered=False,
                registeredAt=None,
                model=None
            )
        
        # Convert registeredAt timestamp
        registered_at = None
        if 'registeredAt' in profile_data and profile_data['registeredAt']:
            reg_at = profile_data['registeredAt']
            if hasattr(reg_at, 'timestamp'):
                registered_at = datetime.fromtimestamp(reg_at.timestamp(), tz=timezone.utc)
        
        return VoiceStatusResponse(
            isRegistered=True,
            registeredAt=registered_at,
            model=profile_data.get('model', 'speechbrain/spkrec-ecapa-voxceleb')
        )
        
    except Exception as e:
        print(f"[VOICE] Error getting voice status: {e}")
        # Return not registered on error
        return VoiceStatusResponse(
            isRegistered=False,
            registeredAt=None,
            model=None
        )


@router.delete(
    "/delete",
    summary="Delete voice profile",
    description="Remove the user's voice profile and embedding from the system.",
    responses={
        200: {
            "description": "Voice profile deleted successfully",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
        404: {
            "description": "Voice profile not found",
        },
    },
)
def delete_voice_profile(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Delete the user's voice profile.
    
    This removes the voice embedding and profile data from Firestore.
    The user can re-enroll their voice after deletion.
    
    Args:
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        Dict with success status and message
        
    Raises:
        HTTPException: 404 if profile not found
    """
    try:
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Get voice profile
        doc_ref = db.collection('voice_profiles').document(uid)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Voice profile not found"
            )
        
        # Delete the document
        doc_ref.delete()
        
        print(f"[VOICE] Deleted voice profile for user {uid}")
        
        return {
            "success": True,
            "message": "Voice profile deleted successfully"
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[VOICE] Error deleting voice profile: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to delete voice profile: {str(e)}"
        )

