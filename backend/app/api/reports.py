from typing import Dict, Any
from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime, timezone, timedelta

from app.auth.dependencies import get_current_user
from app.models.report import WeeklyReportResponse, MonthlyReportResponse
from firebase_admin import firestore

router = APIRouter(
    prefix="/reports",
    tags=["reports"],
)


def get_firestore_db():
    """Get Firestore database instance."""
    try:
        return firestore.client()
    except Exception as e:
        raise RuntimeError(f"Firestore not available: {e}")


@router.get(
    "/weekly",
    response_model=WeeklyReportResponse,
    summary="Get weekly report",
    description="Returns aggregated statistics for the last 7 days of listening sessions.",
    responses={
        200: {
            "description": "Weekly report retrieved successfully",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
    },
)
def get_weekly_report(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> WeeklyReportResponse:
    """Get weekly report for the current user.
    
    Calculates statistics for the last 7 days, including:
    - Total stopped sessions
    - Total listening minutes
    - Total flagged and positive interactions
    - Average minutes per session
    
    Only includes sessions with status STOPPED (ignores unfinished/active sessions).
    
    Args:
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        WeeklyReportResponse: Aggregated weekly statistics
    """
    try:
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Calculate date range (last 7 days)
        week_end = datetime.now(timezone.utc)
        week_start = week_end - timedelta(days=7)
        
        # Query for all stopped sessions for this user
        # We'll filter by date in Python to avoid Firestore index requirements
        sessions_query = db.collection('listening_sessions') \
            .where('uid', '==', uid) \
            .where('status', '==', 'STOPPED') \
            .stream()
        
        # Aggregate statistics
        total_sessions = 0
        total_listening_seconds = 0
        total_flagged = 0
        total_positive = 0
        
        for doc in sessions_query:
            session_data = doc.to_dict()
            
            # Only process stopped sessions (already filtered, but double-check)
            if session_data.get('status') != 'STOPPED':
                continue
            
            # Filter by date range (last 7 days)
            started_at = session_data.get('startedAt')
            if not started_at:
                continue
            
            # Convert Firestore Timestamp to datetime if needed
            if hasattr(started_at, 'timestamp'):
                started_at_dt = datetime.fromtimestamp(started_at.timestamp(), tz=timezone.utc)
            elif isinstance(started_at, datetime):
                started_at_dt = started_at
                if started_at_dt.tzinfo is None:
                    started_at_dt = started_at_dt.replace(tzinfo=timezone.utc)
            else:
                continue
            
            # Skip if outside date range
            if started_at_dt < week_start or started_at_dt > week_end:
                continue
            
            # Ensure totals is properly structured
            totals = session_data.get('totals', {})
            if not isinstance(totals, dict):
                totals = {}
            
            # Extract statistics
            total_seconds = totals.get('totalSeconds', 0)
            flagged_count = totals.get('flaggedCount', 0)
            positive_count = totals.get('positiveCount', 0)
            
            # Aggregate
            total_sessions += 1
            total_listening_seconds += total_seconds
            total_flagged += flagged_count
            total_positive += positive_count
        
        # Convert seconds to minutes
        total_listening_minutes = total_listening_seconds / 60.0
        
        # Calculate average minutes per session
        if total_sessions > 0:
            average_minutes_per_session = total_listening_minutes / total_sessions
        else:
            average_minutes_per_session = 0.0
        
        # Calculate productivity score (0-100)
        # Based on ratio of positive to total interactions
        # Score increases for positiveCount, decreases for flaggedCount
        total_interactions = total_positive + total_flagged
        
        if total_interactions == 0:
            # No interactions means neutral score
            productivity_score = 50
        else:
            # Calculate ratio of positive to total interactions
            positive_ratio = total_positive / total_interactions
            # Scale to 0-100 range
            productivity_score = int(positive_ratio * 100)
        
        # Ensure score is within bounds
        productivity_score = max(0, min(100, productivity_score))
        
        print(f"[REPORTS] Weekly report for user {uid}: {total_sessions} sessions, {total_listening_minutes:.2f} minutes, score: {productivity_score}")
        
        return WeeklyReportResponse(
            weekStart=week_start,
            weekEnd=week_end,
            totalSessions=total_sessions,
            totalListeningMinutes=round(total_listening_minutes, 2),
            totalFlagged=total_flagged,
            totalPositive=total_positive,
            averageMinutesPerSession=round(average_minutes_per_session, 2),
            productivityScore=productivity_score,
        )
        
    except Exception as e:
        print(f"[REPORTS] Error generating weekly report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate weekly report"
        )


@router.get(
    "/monthly",
    response_model=MonthlyReportResponse,
    summary="Get monthly report",
    description="Returns aggregated statistics for the last 30 days of listening sessions.",
    responses={
        200: {
            "description": "Monthly report retrieved successfully",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
    },
)
def get_monthly_report(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> MonthlyReportResponse:
    """Get monthly report for the current user.
    
    Calculates statistics for the last 30 days, including:
    - Total stopped sessions
    - Total listening minutes
    - Total flagged and positive interactions
    - Average minutes per session
    - Best day (day with highest positiveCount)
    
    Only includes sessions with status STOPPED (ignores unfinished/active sessions).
    
    Args:
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        MonthlyReportResponse: Aggregated monthly statistics
    """
    try:
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Calculate date range (last 30 days)
        month_end = datetime.now(timezone.utc)
        month_start = month_end - timedelta(days=30)
        
        # Query for all stopped sessions for this user
        # We'll filter by date in Python to avoid Firestore index requirements
        sessions_query = db.collection('listening_sessions') \
            .where('uid', '==', uid) \
            .where('status', '==', 'STOPPED') \
            .stream()
        
        # Aggregate statistics
        total_sessions = 0
        total_listening_seconds = 0
        total_flagged = 0
        total_positive = 0
        
        # Track positive counts by day to find best day
        daily_positive_counts: Dict[str, int] = {}  # date string -> positive count
        
        for doc in sessions_query:
            session_data = doc.to_dict()
            
            # Only process stopped sessions (already filtered, but double-check)
            if session_data.get('status') != 'STOPPED':
                continue
            
            # Filter by date range (last 30 days)
            started_at = session_data.get('startedAt')
            if not started_at:
                continue
            
            # Convert Firestore Timestamp to datetime if needed
            if hasattr(started_at, 'timestamp'):
                started_at_dt = datetime.fromtimestamp(started_at.timestamp(), tz=timezone.utc)
            elif isinstance(started_at, datetime):
                started_at_dt = started_at
                if started_at_dt.tzinfo is None:
                    started_at_dt = started_at_dt.replace(tzinfo=timezone.utc)
            else:
                continue
            
            # Skip if outside date range
            if started_at_dt < month_start or started_at_dt > month_end:
                continue
            
            # Ensure totals is properly structured
            totals = session_data.get('totals', {})
            if not isinstance(totals, dict):
                totals = {}
            
            # Extract statistics
            total_seconds = totals.get('totalSeconds', 0)
            flagged_count = totals.get('flaggedCount', 0)
            positive_count = totals.get('positiveCount', 0)
            
            # Aggregate
            total_sessions += 1
            total_listening_seconds += total_seconds
            total_flagged += flagged_count
            total_positive += positive_count
            
            # Track positive counts by day for best day calculation
            # Use date string (YYYY-MM-DD) as key
            date_key = started_at_dt.date().isoformat()
            if date_key not in daily_positive_counts:
                daily_positive_counts[date_key] = 0
            daily_positive_counts[date_key] += positive_count
        
        # Convert seconds to minutes
        total_listening_minutes = total_listening_seconds / 60.0
        
        # Calculate average minutes per session
        if total_sessions > 0:
            average_minutes_per_session = total_listening_minutes / total_sessions
        else:
            average_minutes_per_session = 0.0
        
        # Find best day (day with highest positiveCount)
        best_day = None
        if daily_positive_counts:
            best_date_key = max(daily_positive_counts.items(), key=lambda x: x[1])[0]
            # Convert date string back to datetime (start of day in UTC)
            from datetime import date as date_type
            best_date = date_type.fromisoformat(best_date_key)
            best_day = datetime.combine(best_date, datetime.min.time(), tzinfo=timezone.utc)
        
        print(f"[REPORTS] Monthly report for user {uid}: {total_sessions} sessions, {total_listening_minutes:.2f} minutes, best day: {best_day}")
        
        return MonthlyReportResponse(
            monthStart=month_start,
            monthEnd=month_end,
            totalSessions=total_sessions,
            totalListeningMinutes=round(total_listening_minutes, 2),
            totalFlagged=total_flagged,
            totalPositive=total_positive,
            averageMinutesPerSession=round(average_minutes_per_session, 2),
            bestDay=best_day,
        )
        
    except Exception as e:
        print(f"[REPORTS] Error generating monthly report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate monthly report"
        )

