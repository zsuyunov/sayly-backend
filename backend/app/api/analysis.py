from typing import Dict, Any, Optional
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from datetime import datetime, timezone
import os
import numpy as np

from app.auth.dependencies import get_current_user
from app.services.ai_service import analyze_speech, generate_session_summary
from app.services.whisper_service import transcribe_audio
from app.services.verification_service import verify_session_audio, verify_chunk_audio
from app.services.audio_chunking_service import split_audio, cleanup_chunks, reconstruct_audio_from_chunks
from app.services.model_versioning_service import store_model_metadata_for_verification
from app.models.verification import ChunkVerification, VerificationDecision
from firebase_admin import firestore
import tempfile

router = APIRouter(
    prefix="/analysis",
    tags=["analysis"],
)


def get_firestore_db():
    """Get Firestore database instance."""
    try:
        return firestore.client()
    except Exception as e:
        raise RuntimeError(f"Firestore not available: {e}")


def get_user_privacy_preferences(uid: str, db) -> Dict[str, Any]:
    """Get user's privacy preferences from Firestore."""
    try:
        doc_ref = db.collection('user_privacy').document(uid)
        doc = doc_ref.get()
        
        if not doc.exists:
            return {
                'listeningEnabled': False,
                'dataAnalysisEnabled': False,
                'analyticsEnabled': False,
            }
        
        privacy_data = doc.to_dict()
        return {
            'listeningEnabled': privacy_data.get('listeningEnabled', False),
            'dataAnalysisEnabled': privacy_data.get('dataAnalysisEnabled', False),
            'analyticsEnabled': privacy_data.get('analyticsEnabled', False),
        }
    except Exception as e:
        print(f"[ANALYSIS] Error fetching privacy preferences: {e}")
        return {
            'listeningEnabled': False,
            'dataAnalysisEnabled': False,
            'analyticsEnabled': False,
        }


def is_owner_segment(segment_start: float, segment_end: float, owner_chunks: list) -> bool:
    """
    Check if a transcript segment overlaps with OWNER chunks by at least 50%.
    
    Args:
        segment_start: Start time of the transcript segment
        segment_end: End time of the transcript segment
        owner_chunks: List of (start, end) tuples for OWNER chunks
        
    Returns:
        True if the segment is considered OWNER speech, False otherwise
    """
    segment_duration = segment_end - segment_start
    if segment_duration <= 0:
        return False
        
    overlap_duration = 0.0
    for chunk_start, chunk_end in owner_chunks:
        # Calculate overlap
        overlap_start = max(segment_start, chunk_start)
        overlap_end = min(segment_end, chunk_end)
        
        if overlap_end > overlap_start:
            overlap_duration += (overlap_end - overlap_start)
            
    # If segment overlaps OWNER chunks by >= 50% of its duration, keep it
    return (overlap_duration / segment_duration) >= 0.5


async def process_audio_analysis(session_id: str, uid: str):
    """Background task to process audio analysis.
    
    This function:
    1. Gets the session and audio file
    2. Transcribes the audio
    3. Analyzes the speech
    4. Generates a summary
    5. Updates the session with results
    """
    db = get_firestore_db()
    session_ref = db.collection('listening_sessions').document(session_id)
    
    try:
        # Update status to PROCESSING
        session_ref.update({
            'analysisStatus': 'PROCESSING',
        })
        
        # Get session data
        session_doc = session_ref.get()
        if not session_doc.exists:
            print(f"[ANALYSIS] Session {session_id} not found")
            return
        
        session_data = session_doc.to_dict()
        audio_url = session_data.get('audioUrl')
        
        if not audio_url:
            print(f"[ANALYSIS] No audio URL for session {session_id}")
            session_ref.update({
                'analysisStatus': 'FAILED',
                'errorReason': 'No audio file found for session',
            })
            return
        
        # Check privacy preferences
        privacy_prefs = get_user_privacy_preferences(uid, db)
        if not privacy_prefs.get('dataAnalysisEnabled', False):
            print(f"[ANALYSIS] Data analysis disabled for user {uid}")
            session_ref.update({
                'analysisStatus': 'FAILED',
                'errorReason': 'Data analysis disabled in privacy settings',
            })
            return
        
        # Check if audio file exists
        if not os.path.exists(audio_url):
            print(f"[ANALYSIS] Audio file not found: {audio_url}")
            session_ref.update({
                'analysisStatus': 'FAILED',
                'errorReason': f'Audio file not found: {audio_url}',
            })
            return
        
        # Step 0: Chunk-level speaker verification (ONLY if voice is registered)
        # V1: Verification labels chunks for filtering, doesn't block processing
        # IMPORTANT: Without voice registration, verification is DISABLED and all speech is analyzed transparently
        verification_result = None
        verification_enabled = False
        chunk_verifications = []
        chunks = []
        owner_ratio = 1.0  # Default to 1.0 if verification not performed
        
        try:
            # Check if user has registered voice
            voice_profile_ref = db.collection('voice_profiles').document(uid)
            voice_profile_doc = voice_profile_ref.get()
            
            if voice_profile_doc.exists:
                voice_profile_data = voice_profile_doc.to_dict()
                
                # Get enrollment embeddings (prefer new format, fallback to legacy)
                enrollment_embeddings_raw = voice_profile_data.get('enrollmentEmbeddings')
                
                # Handle dictionary format (Firestore map) - convert to list of lists
                if isinstance(enrollment_embeddings_raw, dict):
                    # Sort by key (index) to maintain order
                    sorted_keys = sorted(enrollment_embeddings_raw.keys(), key=lambda x: int(x) if x.isdigit() else 0)
                    enrollment_embeddings = []
                    for k in sorted_keys:
                        emb_value = enrollment_embeddings_raw[k]
                        # Ensure it's a list
                        if isinstance(emb_value, list):
                            enrollment_embeddings.append(emb_value)
                        else:
                            print(f"[ANALYSIS] WARNING: Embedding at key '{k}' is not a list! Type: {type(emb_value)}")
                            # Try to convert if it's a single value or array-like
                            if hasattr(emb_value, '__iter__') and not isinstance(emb_value, str):
                                enrollment_embeddings.append(list(emb_value))
                            else:
                                print(f"[ANALYSIS] ERROR: Cannot convert embedding at key '{k}' to list")
                    print(f"[ANALYSIS] Converted enrollmentEmbeddings from dict to list: {len(enrollment_embeddings)} embeddings")
                else:
                    enrollment_embeddings = enrollment_embeddings_raw
                
                if not enrollment_embeddings:
                    # Fallback to legacy single embedding
                    stored_embedding = voice_profile_data.get('voiceEmbedding')
                    if stored_embedding:
                        enrollment_embeddings = [stored_embedding]
                
                if enrollment_embeddings:
                    # Debug: Check embedding dimensions
                    if enrollment_embeddings and len(enrollment_embeddings) > 0:
                        for i, emb in enumerate(enrollment_embeddings):
                            if isinstance(emb, list):
                                print(f"[ANALYSIS] Enrollment embedding {i} dimension: {len(emb)}")
                            else:
                                print(f"[ANALYSIS] WARNING: Enrollment embedding {i} is not a list! Type: {type(emb)}")
                                # Try to fix it
                                if hasattr(emb, '__iter__') and not isinstance(emb, str):
                                    enrollment_embeddings[i] = list(emb)
                                    print(f"[ANALYSIS] Fixed embedding {i} by converting to list: {len(enrollment_embeddings[i])} dimensions")
                                else:
                                    print(f"[ANALYSIS] ERROR: Cannot fix embedding {i}")
                    # Voice is registered - enable verification
                    verification_enabled = True
                    print(f"[ANALYSIS] Voice registration found. Starting chunk-level verification for session {session_id} (v1: filtering mode)")
                    
                    # Split audio into chunks (10-15 seconds)
                    try:
                        chunks = split_audio(audio_url, chunk_duration=12.0)  # 12 second chunks
                        print(f"[ANALYSIS] Split audio into {len(chunks)} chunks")
                    except Exception as chunk_error:
                        print(f"[ANALYSIS] Error splitting audio: {chunk_error}, continuing without chunking")
                        chunks = []
                    
                    if chunks:
                        # Verify each chunk
                        chunk_decisions = []
                        for chunk in chunks:
                            try:
                                chunk_verification = verify_chunk_audio(
                                    chunk.path,
                                    chunk.start_time,
                                    chunk.end_time,
                                    chunk.index,
                                    enrollment_embeddings,
                                    environment=None,
                                    uid=uid
                                )
                                chunk_verifications.append(chunk_verification)
                                # Use user-facing decision (OWNER or OTHER)
                                chunk_decisions.append(chunk_verification.decision.decision)
                                print(f"[ANALYSIS] Chunk {chunk.index}: {chunk_verification.decision.decision} (max_sim={chunk_verification.decision.maxSimilarity:.3f}, internal={chunk_verification.decision.internalState})")
                            except Exception as chunk_verify_error:
                                print(f"[ANALYSIS] Error verifying chunk {chunk.index}: {chunk_verify_error}")
                                # V1: Map errors to OWNER to avoid blocking
                                decision = VerificationDecision(
                                    decision="OWNER",  # User-facing: treat as OWNER
                                    internalState="SKIPPED",  # Internal: log as SKIPPED
                                    maxSimilarity=0.0,
                                    topKMean=0.0,
                                    allSimilarities=[],
                                    thresholdUsed={}
                                )
                                chunk_verification = ChunkVerification(
                                    startTime=chunk.start_time,
                                    endTime=chunk.end_time,
                                    decision=decision,
                                    chunkIndex=chunk.index
                                )
                                chunk_verifications.append(chunk_verification)
                                chunk_decisions.append("OWNER")  # V1: Map to OWNER
                        
                        # Calculate owner ratio for UX
                        if chunk_decisions:
                            owner_count = sum(1 for d in chunk_decisions if d == "OWNER")
                            owner_ratio = owner_count / len(chunk_decisions)
                            verification_result = "OWNER" if owner_ratio >= 0.5 else "OTHER"
                            print(f"[ANALYSIS] Verification complete: {owner_count}/{len(chunks)} chunks are OWNER (ratio: {owner_ratio:.2f})")
                        else:
                            verification_result = "OWNER"  # Default if no chunks verified
                    else:
                        print(f"[ANALYSIS] No chunks created, skipping chunk-level verification")
                        verification_result = "OWNER"  # Default: treat as OWNER if chunking fails
                else:
                    # Voice profile exists but no embeddings - treat as not registered
                    print(f"[ANALYSIS] Voice profile exists but no enrollment embeddings found for user {uid}. Verification DISABLED - processing all speech transparently.")
                    verification_enabled = False
                    verification_result = "NOT_APPLICABLE"
            else:
                # No voice profile - verification is disabled
                print(f"[ANALYSIS] No voice profile found for user {uid}. Verification DISABLED - processing all speech transparently.")
                verification_enabled = False
                verification_result = "NOT_APPLICABLE"
        except Exception as e:
            print(f"[ANALYSIS] Error during voice verification check (v1: logging only): {e}")
            # On error, disable verification and process all speech
            verification_enabled = False
            verification_result = "NOT_APPLICABLE"
            # Store error internally for logging
            try:
                session_ref.update({
                    'voiceVerificationError': str(e),
                    'voiceVerificationInternalStatus': 'ERROR',
                })
            except:
                pass
        
        # Store model metadata for verification (only if verification was performed)
        if verification_enabled and (chunk_verifications or verification_result):
            try:
                from app.services.model_versioning_service import get_current_model_metadata
                model_metadata = get_current_model_metadata()
                store_model_metadata_for_verification(session_id, uid, model_metadata)
            except Exception as e:
                print(f"[ANALYSIS] Error storing model metadata: {e}")
        
        # Step 1: Transcribe full audio using Whisper Service
        print(f"[ANALYSIS] Transcribing full audio for session {session_id}")
        
        stt_result = None
        try:
            stt_result = await transcribe_audio(audio_url)
        except Exception as stt_error:
            print(f"[ANALYSIS] STT failed for session {session_id}: {stt_error}")
            session_ref.update({
                'stt': {
                    'status': 'FAILED',
                    'error': str(stt_error)
                },
                'analysisStatus': 'FAILED',
                'errorReason': 'Speech-to-text service unavailable'
            })
            if chunks:
                cleanup_chunks(chunks)
            return

        raw_text = stt_result.get("text", "")
        segments = stt_result.get("segments", [])
        
        if not raw_text or not raw_text.strip():
            print(f"[ANALYSIS] Empty transcript for session {session_id}")
            session_ref.update({
                'analysisStatus': 'COMPLETED',
                'audioProcessed': True,
                'summary': 'No speech detected in this session.',
                'gossipScore': 50,
                'voiceVerificationResult': verification_result,
                'voiceVerificationOwnerRatio': owner_ratio,
                'stt': {
                    'status': 'SUCCESS',
                    'model': 'openai/whisper-small',
                    'raw_text': '',
                    'owner_text_only': '',
                    'segments_count': 0
                }
            })
            if chunks:
                cleanup_chunks(chunks)
            return

        # Step 1.5: Filter transcript based on chunk verification
        owner_text_only = raw_text
        
        if verification_enabled and chunk_verifications:
            print(f"[ANALYSIS] Filtering transcript for OWNER segments (ratio: {owner_ratio:.2f})")
            
            # Extract OWNER chunks
            owner_chunks = []
            for cv in chunk_verifications:
                if cv.decision.decision == "OWNER":
                    owner_chunks.append((cv.startTime, cv.endTime))
            
            # Filter segments
            filtered_segments = []
            for seg in segments:
                if is_owner_segment(seg['start'], seg['end'], owner_chunks):
                    filtered_segments.append(seg['text'])
            
            if filtered_segments:
                owner_text_only = " ".join(filtered_segments)
            else:
                owner_text_only = ""
                print(f"[ANALYSIS] All segments filtered out as non-OWNER")
        else:
            print(f"[ANALYSIS] Verification disabled/not applicable - using full transcript")

        # Store STT metadata
        session_ref.update({
            'stt': {
                'status': 'SUCCESS',
                'model': 'openai/whisper-small',
                'raw_text': raw_text,
                'owner_text_only': owner_text_only,
                'segments_count': len(segments)
            }
        })

        filtered_transcript = owner_text_only
        
        # Step 2: Analyze speech
        analysis = analyze_speech(filtered_transcript)
        
        # Step 3: Generate summary
        print(f"[ANALYSIS] Generating summary for session {session_id}")
        summary = generate_session_summary(analysis, transcript)
        
        # Step 4: Update session with results
        # Update totals based on analysis
        current_totals = session_data.get('totals', {})
        if not isinstance(current_totals, dict):
            current_totals = {}
        
        updated_totals = {
            'totalSeconds': current_totals.get('totalSeconds', 0),
            'flaggedCount': analysis.get('flaggedCount', 0),
            'positiveCount': analysis.get('positiveCount', 0),
        }
        
        # Prepare update data
        update_data = {
            'analysisStatus': 'COMPLETED',
            'audioProcessed': True,
            'totals': updated_totals,
            'summary': summary,
            'gossipScore': analysis.get('score', 50),
        }
        
        # Add verification results (v1: store for UX and internal logging)
        update_data['voiceVerificationEnabled'] = verification_enabled  # Whether verification was performed
        if verification_result:
            update_data['voiceVerificationResult'] = verification_result  # User-facing: OWNER, OTHER, or NOT_APPLICABLE
            if verification_enabled:
                update_data['voiceVerificationOwnerRatio'] = owner_ratio  # For UX display (only if verification enabled)
            else:
                # No verification - add disclaimer
                update_data['voiceVerificationDisclaimer'] = "Voice recognition not enabled. All speech analyzed transparently."
        
        # Store chunk-level verification data (only if verification was enabled)
        if verification_enabled and chunk_verifications:
            # Store chunk-level decisions (for internal logging and future filtering)
            update_data['chunkVerifications'] = [
                {
                    'startTime': cv.startTime,
                    'endTime': cv.endTime,
                    'decision': cv.decision.decision,  # User-facing: OWNER or OTHER
                    'internalState': cv.decision.internalState,  # Internal: OWNER, UNCERTAIN, OTHER, SKIPPED
                    'maxSimilarity': cv.decision.maxSimilarity,
                    'topKMean': cv.decision.topKMean,
                    # Store all similarities internally for research
                    'allSimilarities': cv.decision.allSimilarities,
                    'thresholdUsed': cv.decision.thresholdUsed,
                }
                for cv in chunk_verifications
            ]
            # Calculate aggregate metrics (for internal use)
            max_sims = [cv.decision.maxSimilarity for cv in chunk_verifications if cv.decision.maxSimilarity > 0]
            if max_sims:
                update_data['voiceVerificationMaxSimilarity'] = max(max_sims)  # Internal metric
                update_data['voiceVerificationTopKMean'] = np.mean(sorted(max_sims, reverse=True)[:2]) if len(max_sims) >= 2 else max_sims[0]
            
            # Store chunk aggregation for UX
            total_chunks = len(chunk_verifications)
            owner_chunks = sum(1 for cv in chunk_verifications if cv.decision.decision == "OWNER")
            other_chunks = total_chunks - owner_chunks
            update_data['voiceVerificationChunkStats'] = {
                'totalChunks': total_chunks,
                'ownerChunks': owner_chunks,
                'otherChunks': other_chunks,
                'ownerRatio': owner_ratio,
            }
        
        session_ref.update(update_data)
        
        # Cleanup chunks
        if chunks:
            cleanup_chunks(chunks)
        
        print(f"[ANALYSIS] Completed analysis for session {session_id}: score={analysis.get('score')}, flagged={analysis.get('flaggedCount')}, positive={analysis.get('positiveCount')}, verification={verification_result}")
        
        # Cleanup: Delete audio file after successful analysis (optional - can be deferred)
        # Audio files are kept for 30 days, then cleaned up by scheduled job
        # Uncomment below to delete immediately after analysis:
        # try:
        #     if os.path.exists(audio_url):
        #         os.remove(audio_url)
        #         print(f"[ANALYSIS] Deleted audio file after analysis: {audio_url}")
        # except Exception as cleanup_error:
        #     print(f"[ANALYSIS] Failed to delete audio file: {cleanup_error}")
        
    except Exception as e:
        print(f"[ANALYSIS] Error processing audio for session {session_id}: {e}")
        try:
            session_ref.update({
                'analysisStatus': 'FAILED',
                'errorReason': f'Analysis failed: {str(e)}',
            })
        except:
            pass


@router.post(
    "/process/{session_id}",
    summary="Process audio analysis for a session",
    description="Triggers AI analysis of the audio file for a session. This is an async operation that processes in the background.",
    responses={
        202: {
            "description": "Analysis started successfully",
        },
        400: {
            "description": "Bad request - Session has no audio or analysis already completed",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
        403: {
            "description": "Forbidden - Session does not belong to current user or data analysis disabled",
        },
        404: {
            "description": "Session not found",
        },
    },
)
async def process_analysis(
    session_id: str,
    background_tasks: BackgroundTasks,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> Dict[str, Any]:
    """Process audio analysis for a session.
    
    This endpoint triggers background processing of the audio file:
    1. Transcribes audio to text
    2. Analyzes speech for gossip patterns
    3. Generates a summary
    4. Updates session with results
    
    The processing happens asynchronously in the background.
    
    Args:
        session_id: The ID of the session to process
        background_tasks: FastAPI background tasks
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        Dict with success status and message
        
    Raises:
        HTTPException: 400 if session has no audio, 403 if user doesn't own session, 404 if session not found
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
                detail="You do not have permission to process this session"
            )
        
        # Check if audio exists
        audio_url = session_data.get('audioUrl')
        if not audio_url:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Session has no audio file to process"
            )
        
        # Check if already processed or processing
        analysis_status = session_data.get('analysisStatus', 'PENDING')
        if analysis_status == 'PROCESSING':
            return {
                "success": True,
                "message": "Analysis is already in progress"
            }
        
        if analysis_status == 'COMPLETED':
            return {
                "success": True,
                "message": "Analysis already completed"
            }
        
        # Check privacy preferences
        privacy_prefs = get_user_privacy_preferences(uid, db)
        if not privacy_prefs.get('dataAnalysisEnabled', False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Data analysis is disabled in privacy settings"
            )
        
        # Start background processing
        background_tasks.add_task(process_audio_analysis, session_id, uid)
        
        print(f"[ANALYSIS] Started background processing for session {session_id}")
        
        return {
            "success": True,
            "message": "Analysis started. Results will be available shortly."
        }
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[ANALYSIS] Error starting analysis: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start analysis"
        )

