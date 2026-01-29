from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime, timezone
import uuid

from app.auth.dependencies import get_current_user
from app.models.session import (
    ListeningSession,
    StartSessionRequest,
    StartSessionResponse,
    StopSessionResponse,
    LastSessionResponse,
    SessionSummary,
    SessionsListResponse,
    SessionDetailResponse,
    UpdateNoteRequest,
)
from firebase_admin import firestore

router = APIRouter(
    prefix="/sessions",
    tags=["sessions"],
)


def get_firestore_db():
    """Get Firestore database instance."""
    try:
        return firestore.client()
    except Exception as e:
        raise RuntimeError(f"Firestore not available: {e}")


def get_user_privacy_preferences(uid: str, db) -> Dict[str, Any]:
    """Get user's privacy preferences from Firestore.
    
    Returns default values if no record exists:
    - listeningEnabled = False
    - dataAnalysisEnabled = False
    - analyticsEnabled = False
    
    Args:
        uid: User ID
        db: Firestore database instance
        
    Returns:
        Dict with privacy preferences
    """
    try:
        doc_ref = db.collection('user_privacy').document(uid)
        doc = doc_ref.get()
        
        if not doc.exists:
            # Return defaults
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
        print(f"[SESSIONS] Error fetching privacy preferences: {e}")
        # Return safe defaults on error
        return {
            'listeningEnabled': False,
            'dataAnalysisEnabled': False,
            'analyticsEnabled': False,
        }


@router.post(
    "/start",
    response_model=StartSessionResponse,
    summary="Start a listening session",
    description="Creates a new listening session or returns the existing active session for the user.",
    responses={
        200: {
            "description": "Session created or existing active session returned",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
    },
)
def start_session(
    request: StartSessionRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> StartSessionResponse:
    """Start a new listening session.
    
    If the user already has an ACTIVE session, returns that instead of creating a new one.
    Only one ACTIVE session is allowed per user.
    
    Args:
        request: Session start request containing device type
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        StartSessionResponse: The created or existing active session
    """
    try:
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Check privacy preferences - enforce listeningEnabled
        privacy_prefs = get_user_privacy_preferences(uid, db)
        if not privacy_prefs.get('listeningEnabled', False):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="Listening is disabled in privacy settings."
            )
        
        # Check for existing ACTIVE session
        active_sessions = db.collection('listening_sessions') \
            .where('uid', '==', uid) \
            .where('status', '==', 'ACTIVE') \
            .limit(1) \
            .stream()
        
        active_session_list = list(active_sessions)
        if active_session_list:
            # Return existing active session
            existing_doc = active_session_list[0]
            existing_data = existing_doc.to_dict()
            existing_data['id'] = existing_doc.id
            
            # Ensure totals is properly structured
            if 'totals' not in existing_data or not isinstance(existing_data.get('totals'), dict):
                existing_data['totals'] = {
                    'totalSeconds': 0,
                    'flaggedCount': 0,
                    'positiveCount': 0,
                }
            else:
                # Ensure all required fields are present
                if 'totalSeconds' not in existing_data['totals']:
                    existing_data['totals']['totalSeconds'] = 0
                if 'flaggedCount' not in existing_data['totals']:
                    existing_data['totals']['flaggedCount'] = 0
                if 'positiveCount' not in existing_data['totals']:
                    existing_data['totals']['positiveCount'] = 0
            
            # Convert Firestore Timestamp to datetime if needed
            if 'startedAt' in existing_data:
                started_at = existing_data['startedAt']
                if hasattr(started_at, 'timestamp'):
                    existing_data['startedAt'] = datetime.fromtimestamp(started_at.timestamp(), tz=timezone.utc)
            
            if 'endedAt' in existing_data and existing_data['endedAt']:
                ended_at = existing_data['endedAt']
                if hasattr(ended_at, 'timestamp'):
                    existing_data['endedAt'] = datetime.fromtimestamp(ended_at.timestamp(), tz=timezone.utc)
            
            session = ListeningSession(**existing_data)
            print(f"[SESSIONS] Returning existing active session {session.id} for user {uid}")
            return StartSessionResponse(session=session)
        
        # Create new session
        session_id = str(uuid.uuid4())
        started_at = datetime.now(timezone.utc)
        
        # Initialize counts based on dataAnalysisEnabled preference
        # If dataAnalysisEnabled is false, counts must remain 0
        data_analysis_enabled = privacy_prefs.get('dataAnalysisEnabled', False)
        
        session_data = {
            'uid': uid,
            'startedAt': started_at,
            'endedAt': None,
            'status': 'ACTIVE',
            'device': request.device,
            'totals': {
                'totalSeconds': 0,
                'flaggedCount': 0 if not data_analysis_enabled else 0,  # Always 0 initially, enforced server-side
                'positiveCount': 0 if not data_analysis_enabled else 0,  # Always 0 initially, enforced server-side
            },
        }
        
        # Store in Firestore
        doc_ref = db.collection('listening_sessions').document(session_id)
        doc_ref.set(session_data)
        
        session = ListeningSession(
            id=session_id,
            **session_data
        )
        
        print(f"[SESSIONS] Created new session {session_id} for user {uid}")
        return StartSessionResponse(session=session)
        
    except Exception as e:
        print(f"[SESSIONS] Error starting session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to start session"
        )


@router.post(
    "/{session_id}/stop",
    response_model=StopSessionResponse,
    summary="Stop a listening session",
    description="Stops an active session by setting endedAt and computing totalSeconds.",
    responses={
        200: {
            "description": "Session stopped successfully",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
        404: {
            "description": "Session not found",
        },
        403: {
            "description": "Forbidden - Session does not belong to current user",
        },
    },
)
def stop_session(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> StopSessionResponse:
    """Stop an active listening session.
    
    Computes totalSeconds from startedAt to endedAt and updates the session status.
    Only the session owner can stop their own session.
    
    Args:
        session_id: The ID of the session to stop
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        StopSessionResponse: The stopped session with computed totalSeconds
        
    Raises:
        HTTPException: 404 if session not found, 403 if user doesn't own the session
    """
    try:
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Get session document
        doc_ref = db.collection('listening_sessions').document(session_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        session_data = doc.to_dict()
        
        # Enforce uid ownership
        if session_data.get('uid') != uid:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to stop this session"
            )
        
        # Check if already stopped
        if session_data.get('status') == 'STOPPED':
            # Return existing stopped session
            session_data['id'] = doc.id
            
            # Convert timestamps
            if 'startedAt' in session_data:
                started_at = session_data['startedAt']
                if hasattr(started_at, 'timestamp'):
                    session_data['startedAt'] = datetime.fromtimestamp(started_at.timestamp(), tz=timezone.utc)
            
            if 'endedAt' in session_data and session_data['endedAt']:
                ended_at = session_data['endedAt']
                if hasattr(ended_at, 'timestamp'):
                    session_data['endedAt'] = datetime.fromtimestamp(ended_at.timestamp(), tz=timezone.utc)
            
            session = ListeningSession(**session_data)
            return StopSessionResponse(session=session)
        
        # Stop the session
        ended_at = datetime.now(timezone.utc)
        
        # Get startedAt and compute totalSeconds
        started_at = session_data.get('startedAt')
        if hasattr(started_at, 'timestamp'):
            started_at_dt = datetime.fromtimestamp(started_at.timestamp(), tz=timezone.utc)
        elif isinstance(started_at, datetime):
            started_at_dt = started_at
            if started_at_dt.tzinfo is None:
                started_at_dt = started_at_dt.replace(tzinfo=timezone.utc)
        else:
            started_at_dt = ended_at  # Fallback
        
        # Calculate total seconds
        total_seconds = int((ended_at - started_at_dt).total_seconds())
        total_seconds = max(0, total_seconds)  # Ensure non-negative
        
        # Get current totals (preserve existing counts)
        current_totals = session_data.get('totals', {})
        if not isinstance(current_totals, dict):
            current_totals = {}
        
        # Check privacy preferences - enforce dataAnalysisEnabled
        # If dataAnalysisEnabled is false, counts must remain 0 (server-side enforcement)
        privacy_prefs = get_user_privacy_preferences(uid, db)
        data_analysis_enabled = privacy_prefs.get('dataAnalysisEnabled', False)
        
        # Update session with new totals
        # If dataAnalysisEnabled is false, force counts to 0 regardless of what's stored
        updated_totals = {
            'totalSeconds': total_seconds,
            'flaggedCount': 0 if not data_analysis_enabled else current_totals.get('flaggedCount', 0),
            'positiveCount': 0 if not data_analysis_enabled else current_totals.get('positiveCount', 0),
        }
        
        update_data = {
            'endedAt': ended_at,
            'status': 'STOPPED',
            'totals': updated_totals,
        }
        
        doc_ref.update(update_data)
        
        # Get updated session data
        updated_doc = doc_ref.get()
        updated_data = updated_doc.to_dict()
        updated_data['id'] = updated_doc.id
        
        # Ensure totals is properly structured
        if 'totals' not in updated_data or not isinstance(updated_data['totals'], dict):
            updated_data['totals'] = updated_totals
        else:
            # Ensure all required fields are present
            if 'totalSeconds' not in updated_data['totals']:
                updated_data['totals']['totalSeconds'] = total_seconds
            if 'flaggedCount' not in updated_data['totals']:
                updated_data['totals']['flaggedCount'] = 0
            if 'positiveCount' not in updated_data['totals']:
                updated_data['totals']['positiveCount'] = 0
        
        # Convert timestamps
        if 'startedAt' in updated_data:
            started_at = updated_data['startedAt']
            if hasattr(started_at, 'timestamp'):
                updated_data['startedAt'] = datetime.fromtimestamp(started_at.timestamp(), tz=timezone.utc)
        
        if 'endedAt' in updated_data and updated_data['endedAt']:
            ended_at = updated_data['endedAt']
            if hasattr(ended_at, 'timestamp'):
                updated_data['endedAt'] = datetime.fromtimestamp(ended_at.timestamp(), tz=timezone.utc)
        
        session = ListeningSession(**updated_data)
        
        print(f"[SESSIONS] Stopped session {session_id} for user {uid} (duration: {total_seconds}s)")
        return StopSessionResponse(session=session)
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[SESSIONS] Error stopping session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to stop session"
        )


@router.get(
    "/last",
    response_model=LastSessionResponse,
    summary="Get last session",
    description="Returns the most recent session for the current user, sorted by startedAt descending.",
    responses={
        200: {
            "description": "Last session retrieved successfully (may be null if no sessions exist)",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
    },
)
def get_last_session(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> LastSessionResponse:
    """Get the most recent session for the current user.
    
    Returns the session with the latest startedAt timestamp, or null if no sessions exist.
    
    Args:
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        LastSessionResponse: The most recent session, or null if none exists
    """
    try:
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Query for sessions by this user, ordered by startedAt descending, limit 1
        sessions_query = db.collection('listening_sessions') \
            .where('uid', '==', uid) \
            .order_by('startedAt', direction=firestore.Query.DESCENDING) \
            .limit(1) \
            .stream()
        
        sessions_list = list(sessions_query)
        
        if not sessions_list:
            print(f"[SESSIONS] No sessions found for user {uid}")
            return LastSessionResponse(session=None)
        
        # Get the most recent session
        last_doc = sessions_list[0]
        session_data = last_doc.to_dict()
        session_data['id'] = last_doc.id
        
        # Enforce dataAnalysisEnabled privacy preference
        # If disabled, counts must be 0 (server-side enforcement)
        privacy_prefs = get_user_privacy_preferences(uid, db)
        data_analysis_enabled = privacy_prefs.get('dataAnalysisEnabled', False)
        
        # Ensure totals is properly structured
        if 'totals' not in session_data or not isinstance(session_data.get('totals'), dict):
            session_data['totals'] = {
                'totalSeconds': 0,
                'flaggedCount': 0,
                'positiveCount': 0,
            }
        else:
            # Ensure all required fields are present
            if 'totalSeconds' not in session_data['totals']:
                session_data['totals']['totalSeconds'] = 0
            # Enforce privacy: if dataAnalysisEnabled is false, force counts to 0
            if not data_analysis_enabled:
                session_data['totals']['flaggedCount'] = 0
                session_data['totals']['positiveCount'] = 0
            else:
                if 'flaggedCount' not in session_data['totals']:
                    session_data['totals']['flaggedCount'] = 0
                if 'positiveCount' not in session_data['totals']:
                    session_data['totals']['positiveCount'] = 0
        
        # Convert Firestore Timestamp to datetime if needed
        if 'startedAt' in session_data:
            started_at = session_data['startedAt']
            if hasattr(started_at, 'timestamp'):
                session_data['startedAt'] = datetime.fromtimestamp(started_at.timestamp(), tz=timezone.utc)
        
        if 'endedAt' in session_data and session_data['endedAt']:
            ended_at = session_data['endedAt']
            if hasattr(ended_at, 'timestamp'):
                session_data['endedAt'] = datetime.fromtimestamp(ended_at.timestamp(), tz=timezone.utc)
        
        session = ListeningSession(**session_data)
        print(f"[SESSIONS] Retrieved last session {session.id} for user {uid}")
        return LastSessionResponse(session=session)
        
    except Exception as e:
        print(f"[SESSIONS] Error getting last session: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve last session"
        )


@router.get(
    "/",
    response_model=SessionsListResponse,
    summary="List all sessions",
    description="Returns all listening sessions for the current user, sorted by startedAt descending (latest first).",
    responses={
        200: {
            "description": "List of sessions retrieved successfully",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
    },
)
def list_sessions(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> SessionsListResponse:
    """Get all listening sessions for the current user.
    
    Returns sessions sorted by startedAt in descending order (latest first).
    Only returns sessions owned by the authenticated user.
    
    Args:
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        SessionsListResponse: List of session summaries, sorted by startedAt DESC
    """
    try:
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Query for all sessions by this user, ordered by startedAt descending
        sessions_query = db.collection('listening_sessions') \
            .where('uid', '==', uid) \
            .order_by('startedAt', direction=firestore.Query.DESCENDING) \
            .stream()
        
        # Enforce dataAnalysisEnabled privacy preference (check once for all sessions)
        privacy_prefs = get_user_privacy_preferences(uid, db)
        data_analysis_enabled = privacy_prefs.get('dataAnalysisEnabled', False)
        
        sessions_list = []
        
        for doc in sessions_query:
            session_data = doc.to_dict()
            
            # Ensure totals is properly structured
            totals = session_data.get('totals', {})
            if not isinstance(totals, dict):
                totals = {}
            
            # Extract required fields
            total_seconds = totals.get('totalSeconds', 0)
            # Force counts to 0 if data analysis is disabled (server-side enforcement)
            if not data_analysis_enabled:
                flagged_count = 0
                positive_count = 0
            else:
                flagged_count = totals.get('flaggedCount', 0)
                positive_count = totals.get('positiveCount', 0)
            
            # Convert timestamps
            started_at = session_data.get('startedAt')
            if hasattr(started_at, 'timestamp'):
                started_at = datetime.fromtimestamp(started_at.timestamp(), tz=timezone.utc)
            elif isinstance(started_at, datetime):
                if started_at.tzinfo is None:
                    started_at = started_at.replace(tzinfo=timezone.utc)
            
            ended_at = session_data.get('endedAt')
            if ended_at:
                if hasattr(ended_at, 'timestamp'):
                    ended_at = datetime.fromtimestamp(ended_at.timestamp(), tz=timezone.utc)
                elif isinstance(ended_at, datetime):
                    if ended_at.tzinfo is None:
                        ended_at = ended_at.replace(tzinfo=timezone.utc)
            else:
                ended_at = None
            
            # Create session summary
            session_summary = SessionSummary(
                id=doc.id,
                startedAt=started_at,
                endedAt=ended_at,
                totalSeconds=total_seconds,
                flaggedCount=flagged_count,
                positiveCount=positive_count,
                status=session_data.get('status', 'STOPPED'),
            )
            
            sessions_list.append(session_summary)
        
        print(f"[SESSIONS] Retrieved {len(sessions_list)} sessions for user {uid}")
        return SessionsListResponse(sessions=sessions_list)
        
    except Exception as e:
        print(f"[SESSIONS] Error listing sessions: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve sessions"
        )


@router.get(
    "/{session_id}",
    response_model=SessionDetailResponse,
    summary="Get session detail",
    description="Returns detailed information about a specific session.",
    responses={
        200: {
            "description": "Session detail retrieved successfully",
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
def get_session_detail(
    session_id: str,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> SessionDetailResponse:
    """Get detailed information about a specific session.
    
    Only the session owner can view their own session details.
    
    Args:
        session_id: The ID of the session to retrieve
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        SessionDetailResponse: Detailed session information
        
    Raises:
        HTTPException: 404 if session not found, 403 if user doesn't own the session
    """
    try:
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Get session document
        doc_ref = db.collection('listening_sessions').document(session_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        session_data = doc.to_dict()
        
        # Enforce uid ownership
        if session_data.get('uid') != uid:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to view this session"
            )
        
        # Ensure totals is properly structured
        totals = session_data.get('totals', {})
        if not isinstance(totals, dict):
            totals = {}
        
        # Extract required fields
        total_seconds = totals.get('totalSeconds', 0)
        
        # Enforce dataAnalysisEnabled privacy preference
        # If disabled, counts must be 0 (server-side enforcement)
        privacy_prefs = get_user_privacy_preferences(uid, db)
        data_analysis_enabled = privacy_prefs.get('dataAnalysisEnabled', False)
        
        # Force counts to 0 if data analysis is disabled
        if not data_analysis_enabled:
            flagged_count = 0
            positive_count = 0
        else:
            flagged_count = totals.get('flaggedCount', 0)
            positive_count = totals.get('positiveCount', 0)
        
        # Convert timestamps
        started_at = session_data.get('startedAt')
        if hasattr(started_at, 'timestamp'):
            started_at = datetime.fromtimestamp(started_at.timestamp(), tz=timezone.utc)
        elif isinstance(started_at, datetime):
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=timezone.utc)
        
        ended_at = session_data.get('endedAt')
        if ended_at:
            if hasattr(ended_at, 'timestamp'):
                ended_at = datetime.fromtimestamp(ended_at.timestamp(), tz=timezone.utc)
            elif isinstance(ended_at, datetime):
                if ended_at.tzinfo is None:
                    ended_at = ended_at.replace(tzinfo=timezone.utc)
        else:
            ended_at = None
        
        # Extract note and updatedAt
        note = session_data.get('note')
        if note is not None:
            note = note.strip() if isinstance(note, str) else None
            if note == '':
                note = None
        
        updated_at = session_data.get('updatedAt')
        if updated_at:
            if hasattr(updated_at, 'timestamp'):
                updated_at = datetime.fromtimestamp(updated_at.timestamp(), tz=timezone.utc)
            elif isinstance(updated_at, datetime):
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)
        else:
            updated_at = None
        
        # Extract analysis fields
        audio_url = session_data.get('audioUrl')
        audio_processed = session_data.get('audioProcessed', False)
        analysis_status = session_data.get('analysisStatus', 'PENDING')
        summary = session_data.get('summary')
        gossip_score = session_data.get('gossipScore')
        
        print(f"[SESSIONS] Retrieved session detail {session_id} for user {uid}")
        
        return SessionDetailResponse(
            id=session_id,
            startedAt=started_at,
            endedAt=ended_at,
            totalSeconds=total_seconds,
            flaggedCount=flagged_count,
            positiveCount=positive_count,
            status=session_data.get('status', 'STOPPED'),
            device=session_data.get('device', 'unknown'),
            note=note,
            updatedAt=updated_at,
            audioUrl=audio_url,
            audioProcessed=audio_processed,
            analysisStatus=analysis_status,
            summary=summary,
            gossipScore=gossip_score,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[SESSIONS] Error getting session detail: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve session detail"
        )


@router.put(
    "/{session_id}/note",
    response_model=SessionDetailResponse,
    summary="Update session note",
    description="Updates the reflection note for a specific session.",
    responses={
        200: {
            "description": "Note updated successfully",
        },
        400: {
            "description": "Bad request - Invalid note (e.g., exceeds max length)",
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
def update_session_note(
    session_id: str,
    request: UpdateNoteRequest,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> SessionDetailResponse:
    """Update the reflection note for a specific session.
    
    Only the session owner can update their own session notes.
    
    Args:
        session_id: The ID of the session to update
        request: The note update request containing the note text
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        SessionDetailResponse: Updated session information including the new note
        
    Raises:
        HTTPException: 404 if session not found, 403 if user doesn't own the session, 400 if note is invalid
    """
    try:
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Get session document
        doc_ref = db.collection('listening_sessions').document(session_id)
        doc = doc_ref.get()
        
        if not doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="Session not found"
            )
        
        session_data = doc.to_dict()
        
        # Enforce uid ownership
        if session_data.get('uid') != uid:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="You do not have permission to update this session"
            )
        
        # Trim and validate note
        note_text = request.note.strip() if request.note else None
        
        # If note is empty after trimming, set to None
        if note_text == '':
            note_text = None
        
        # Validate max length (Pydantic already validates, but double-check)
        if note_text and len(note_text) > 500:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Note exceeds maximum length of 500 characters"
            )
        
        # Update note and updatedAt
        update_data = {
            'note': note_text,
            'updatedAt': firestore.SERVER_TIMESTAMP,
        }
        
        doc_ref.update(update_data)
        
        # Refresh document to get updated data
        updated_doc = doc_ref.get()
        updated_session_data = updated_doc.to_dict()
        
        # Ensure totals is properly structured
        totals = updated_session_data.get('totals', {})
        if not isinstance(totals, dict):
            totals = {}
        
        # Extract required fields
        total_seconds = totals.get('totalSeconds', 0)
        flagged_count = totals.get('flaggedCount', 0)
        positive_count = totals.get('positiveCount', 0)
        
        # Convert timestamps
        started_at = updated_session_data.get('startedAt')
        if hasattr(started_at, 'timestamp'):
            started_at = datetime.fromtimestamp(started_at.timestamp(), tz=timezone.utc)
        elif isinstance(started_at, datetime):
            if started_at.tzinfo is None:
                started_at = started_at.replace(tzinfo=timezone.utc)
        
        ended_at = updated_session_data.get('endedAt')
        if ended_at:
            if hasattr(ended_at, 'timestamp'):
                ended_at = datetime.fromtimestamp(ended_at.timestamp(), tz=timezone.utc)
            elif isinstance(ended_at, datetime):
                if ended_at.tzinfo is None:
                    ended_at = ended_at.replace(tzinfo=timezone.utc)
        else:
            ended_at = None
        
        # Extract updated note and updatedAt
        updated_note = updated_session_data.get('note')
        if updated_note is not None:
            updated_note = updated_note.strip() if isinstance(updated_note, str) else None
            if updated_note == '':
                updated_note = None
        
        updated_at = updated_session_data.get('updatedAt')
        if updated_at:
            if hasattr(updated_at, 'timestamp'):
                updated_at = datetime.fromtimestamp(updated_at.timestamp(), tz=timezone.utc)
            elif isinstance(updated_at, datetime):
                if updated_at.tzinfo is None:
                    updated_at = updated_at.replace(tzinfo=timezone.utc)
        else:
            updated_at = None
        
        # Extract analysis fields
        audio_url = updated_session_data.get('audioUrl')
        audio_processed = updated_session_data.get('audioProcessed', False)
        analysis_status = updated_session_data.get('analysisStatus', 'PENDING')
        summary = updated_session_data.get('summary')
        gossip_score = updated_session_data.get('gossipScore')
        
        print(f"[SESSIONS] Updated note for session {session_id} for user {uid}")
        
        return SessionDetailResponse(
            id=session_id,
            startedAt=started_at,
            endedAt=ended_at,
            totalSeconds=total_seconds,
            flaggedCount=flagged_count,
            positiveCount=positive_count,
            status=updated_session_data.get('status', 'STOPPED'),
            device=updated_session_data.get('device', 'unknown'),
            note=updated_note,
            updatedAt=updated_at,
            audioUrl=audio_url,
            audioProcessed=audio_processed,
            analysisStatus=analysis_status,
            summary=summary,
            gossipScore=gossip_score,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[SESSIONS] Error updating session note: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to update session note"
        )

