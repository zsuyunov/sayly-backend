from typing import Dict, Any, Set
from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime, timezone, timedelta, date

from app.auth.dependencies import get_current_user
from app.models.rewards import RewardsStatusResponse
from firebase_admin import firestore

router = APIRouter(
    prefix="/rewards",
    tags=["rewards"],
)

# Available badges
ALL_BADGES = [
    "First Session",
    "3-Day Streak",
    "7-Day Streak",
    "Speech Guardian",
    "Consistency Builder",
]


def get_firestore_db():
    """Get Firestore database instance."""
    try:
        return firestore.client()
    except Exception as e:
        raise RuntimeError(f"Firestore not available: {e}")


def get_session_dates(sessions_data: list) -> Set[str]:
    """Extract unique dates (YYYY-MM-DD) from sessions."""
    dates = set()
    for session_data in sessions_data:
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
        
        # Get date string (YYYY-MM-DD)
        date_key = started_at_dt.date().isoformat()
        dates.add(date_key)
    
    return dates


def calculate_current_streak(session_dates: Set[str]) -> int:
    """Calculate current streak of consecutive days with sessions.
    
    Starts from today and counts backwards until a day without a session is found.
    """
    if not session_dates:
        return 0
    
    current_date = datetime.now(timezone.utc).date()
    streak = 0
    
    # Check consecutive days starting from today
    check_date = current_date
    while True:
        date_key = check_date.isoformat()
        if date_key in session_dates:
            streak += 1
            check_date = check_date - timedelta(days=1)
        else:
            break
    
    return streak


def calculate_best_streak(session_dates: Set[str]) -> int:
    """Calculate the best (longest) streak in history.
    
    Finds the longest sequence of consecutive days with sessions.
    """
    if not session_dates:
        return 0
    
    # Convert to sorted list of dates
    sorted_dates = sorted([date.fromisoformat(d) for d in session_dates])
    
    if not sorted_dates:
        return 0
    
    best_streak = 1
    current_streak = 1
    
    # Iterate through sorted dates and find longest consecutive sequence
    for i in range(1, len(sorted_dates)):
        days_diff = (sorted_dates[i] - sorted_dates[i-1]).days
        if days_diff == 1:
            # Consecutive day
            current_streak += 1
            best_streak = max(best_streak, current_streak)
        else:
            # Gap in dates, reset streak
            current_streak = 1
    
    return best_streak


@router.get(
    "/status",
    response_model=RewardsStatusResponse,
    summary="Get rewards status",
    description="Returns current streak, best streak, and earned/available badges based on listening sessions.",
    responses={
        200: {
            "description": "Rewards status retrieved successfully",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
    },
)
def get_rewards_status(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> RewardsStatusResponse:
    """Get rewards status for the current user.
    
    Calculates:
    - Current streak: Consecutive days with at least one stopped session (from today backwards)
    - Best streak: Maximum historical streak of consecutive days
    - Earned badges: Badges that have been unlocked
    - Available badges: Badges that are available but not yet earned
    
    Badges:
    - "First Session": At least 1 session
    - "3-Day Streak": 3 consecutive days
    - "7-Day Streak": 7 consecutive days
    - "Speech Guardian": At least one session with no flagged speech
    - "Consistency Builder": 10 total sessions
    
    Args:
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        RewardsStatusResponse: Current streak, best streak, and badge information
    """
    try:
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Query for all stopped sessions for this user
        sessions_query = db.collection('listening_sessions') \
            .where('uid', '==', uid) \
            .where('status', '==', 'STOPPED') \
            .stream()
        
        sessions_list = []
        total_sessions = 0
        has_no_flagged_session = False
        
        for doc in sessions_query:
            session_data = doc.to_dict()
            sessions_list.append(session_data)
            total_sessions += 1
            
            # Check for Speech Guardian badge (session with no flagged speech)
            totals = session_data.get('totals', {})
            if isinstance(totals, dict):
                flagged_count = totals.get('flaggedCount', 0)
                if flagged_count == 0:
                    has_no_flagged_session = True
        
        # Get unique dates with sessions
        session_dates = get_session_dates(sessions_list)
        
        # Calculate streaks
        current_streak = calculate_current_streak(session_dates)
        best_streak = calculate_best_streak(session_dates)
        
        # Determine earned badges
        earned_badges = []
        
        # First Session badge
        if total_sessions >= 1:
            earned_badges.append("First Session")
        
        # 3-Day Streak badge
        if current_streak >= 3 or best_streak >= 3:
            earned_badges.append("3-Day Streak")
        
        # 7-Day Streak badge
        if current_streak >= 7 or best_streak >= 7:
            earned_badges.append("7-Day Streak")
        
        # Speech Guardian badge
        if has_no_flagged_session:
            earned_badges.append("Speech Guardian")
        
        # Consistency Builder badge
        if total_sessions >= 10:
            earned_badges.append("Consistency Builder")
        
        # Available badges are those not yet earned
        available_badges = [badge for badge in ALL_BADGES if badge not in earned_badges]
        
        print(f"[REWARDS] Status for user {uid}: streak={current_streak}, best={best_streak}, badges={len(earned_badges)}")
        
        return RewardsStatusResponse(
            currentStreak=current_streak,
            bestStreak=best_streak,
            earnedBadges=earned_badges,
            availableBadges=available_badges,
        )
        
    except Exception as e:
        print(f"[REWARDS] Error getting rewards status: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to retrieve rewards status"
        )

