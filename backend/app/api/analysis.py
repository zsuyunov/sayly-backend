from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status, BackgroundTasks
from datetime import datetime, timezone
import os

from app.auth.dependencies import get_current_user
from app.services.ai_service import transcribe_audio, analyze_speech, generate_session_summary
from firebase_admin import firestore

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
            })
            return
        
        # Check privacy preferences
        privacy_prefs = get_user_privacy_preferences(uid, db)
        if not privacy_prefs.get('dataAnalysisEnabled', False):
            print(f"[ANALYSIS] Data analysis disabled for user {uid}")
            session_ref.update({
                'analysisStatus': 'FAILED',
            })
            return
        
        # Check if audio file exists
        if not os.path.exists(audio_url):
            print(f"[ANALYSIS] Audio file not found: {audio_url}")
            session_ref.update({
                'analysisStatus': 'FAILED',
            })
            return
        
        # Step 1: Transcribe audio
        print(f"[ANALYSIS] Transcribing audio for session {session_id}")
        transcript = transcribe_audio(audio_url)
        
        if not transcript or not transcript.strip():
            print(f"[ANALYSIS] Empty transcript for session {session_id}")
            session_ref.update({
                'analysisStatus': 'COMPLETED',
                'audioProcessed': True,
                'summary': 'No speech detected in this session.',
                'gossipScore': 50,
            })
            return
        
        # Step 2: Analyze speech
        print(f"[ANALYSIS] Analyzing speech for session {session_id}")
        analysis = analyze_speech(transcript)
        
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
        
        session_ref.update({
            'analysisStatus': 'COMPLETED',
            'audioProcessed': True,
            'totals': updated_totals,
            'summary': summary,
            'gossipScore': analysis.get('score', 50),
        })
        
        print(f"[ANALYSIS] Completed analysis for session {session_id}: score={analysis.get('score')}, flagged={analysis.get('flaggedCount')}, positive={analysis.get('positiveCount')}")
        
    except Exception as e:
        print(f"[ANALYSIS] Error processing audio for session {session_id}: {e}")
        try:
            session_ref.update({
                'analysisStatus': 'FAILED',
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

