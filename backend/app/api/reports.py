from typing import Dict, Any, List, Literal
from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime, timezone, timedelta
from collections import defaultdict

from app.auth.dependencies import get_current_user
from app.models.report import WeeklyReportResponse, MonthlyReportResponse
from app.models.progress import ProgressReportResponse, ChartDataResponse, ChartDataPoint, CategoryDistribution
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


def format_date_context(period: str, period_start: datetime, period_end: datetime) -> str:
    """Format date context string for display."""
    if period == "today":
        # Format as "Jan 29"
        return period_start.strftime("%b %d")
    elif period == "week":
        return "This week"
    elif period == "month":
        # Format as "January"
        return period_start.strftime("%B")
    elif period == "lifetime":
        # Format as "Since Jan 2026"
        return f"Since {period_start.strftime('%b %Y')}"
    return ""


@router.get(
    "/progress/{period}",
    response_model=ProgressReportResponse,
    summary="Get progress report",
    description="Returns aggregated statistics for a specific time period (today, week, month, lifetime).",
    responses={
        200: {
            "description": "Progress report retrieved successfully",
        },
        400: {
            "description": "Invalid period parameter",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
    },
)
def get_progress_report(
    period: Literal["today", "week", "month", "lifetime"],
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ProgressReportResponse:
    """Get progress report for a specific time period.
    
    Calculates statistics for the specified period:
    - today: last 24 hours
    - week: last 7 days
    - month: last 30 days
    - lifetime: all sessions
    
    Only includes sessions with status STOPPED.
    
    Args:
        period: The time period to aggregate (today, week, month, lifetime)
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        ProgressReportResponse: Aggregated statistics for the period
    """
    try:
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Calculate date range based on period
        period_end = datetime.now(timezone.utc)
        
        if period == "today":
            period_start = period_end - timedelta(hours=24)
        elif period == "week":
            period_start = period_end - timedelta(days=7)
        elif period == "month":
            period_start = period_end - timedelta(days=30)
        elif period == "lifetime":
            # For lifetime, we'll query all sessions and find the earliest
            period_start = None  # Will be determined from first session
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid period: {period}. Must be one of: today, week, month, lifetime"
            )
        
        # Query for all stopped sessions for this user
        sessions_query = db.collection('listening_sessions') \
            .where('uid', '==', uid) \
            .where('status', '==', 'STOPPED') \
            .stream()
        
        # Aggregate statistics
        total_sessions = 0
        total_listening_seconds = 0
        total_flagged = 0
        total_positive = 0
        earliest_session = None
        
        # Category totals for distribution calculation
        category_totals = {
            "gossip": 0,
            "unethical": 0,
            "waste": 0,
            "productive": 0,
        }
        
        for doc in sessions_query:
            session_data = doc.to_dict()
            
            # Only process stopped sessions with completed analysis
            if session_data.get('status') != 'STOPPED':
                continue
            
            # NOTE: We count ALL STOPPED sessions for totals/minutes.
            # Category distribution is computed only when `classification` is present.
            
            # Get startedAt timestamp
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
            
            # For lifetime, track earliest session
            if period == "lifetime":
                if earliest_session is None or started_at_dt < earliest_session:
                    earliest_session = started_at_dt
            else:
                # Filter by date range for other periods
                if started_at_dt < period_start or started_at_dt > period_end:
                    continue
            
            # Ensure totals is properly structured
            totals = session_data.get('totals', {})
            if not isinstance(totals, dict):
                totals = {}
            
            # Extract statistics
            total_seconds = totals.get('totalSeconds', 0)
            flagged_count = totals.get('flaggedCount', 0)
            positive_count = totals.get('positiveCount', 0)
            total_minutes = total_seconds / 60.0
            
            # Aggregate
            total_sessions += 1
            total_listening_seconds += total_seconds
            total_flagged += int(flagged_count or 0)
            total_positive += int(positive_count or 0)
            
            # Extract category data from AI classification
            # IMPORTANT: Include ALL sessions with classification data, even if scores are low
            classification = session_data.get('classification', {})
            if classification and isinstance(classification, dict) and len(classification) > 0:
                # Map classification labels to our category names
                # Ensure scores are floats (handle string conversion if needed)
                gossip_score = float(classification.get('gossip', 0.0) or 0.0)
                unethical_score = float(classification.get('insult or unethical speech', 0.0) or 0.0)
                waste_score = float(classification.get('wasteful talk', 0.0) or 0.0)
                productive_score = float(classification.get('productive or meaningful speech', 0.0) or 0.0)
                
                # Add scores to category totals (weighted by session duration in minutes)
                # Even if scores are low, they contribute to the distribution
                category_totals["gossip"] += gossip_score * total_minutes
                category_totals["unethical"] += unethical_score * total_minutes
                category_totals["waste"] += waste_score * total_minutes
                category_totals["productive"] += productive_score * total_minutes
        
        # For lifetime, set period_start to earliest session or account creation
        if period == "lifetime":
            if earliest_session:
                period_start = earliest_session
            else:
                # If no sessions, use a default date (e.g., account creation or 1 year ago)
                period_start = period_end - timedelta(days=365)
        
        # Convert seconds to minutes and round
        total_listening_minutes = round(total_listening_seconds / 60.0)

        # Average minutes per session
        average_minutes_per_session = (total_listening_seconds / 60.0) / total_sessions if total_sessions > 0 else 0.0
        
        # Calculate category distribution percentages
        total_category = sum(category_totals.values())
        if total_category > 0:
            category_distribution = CategoryDistribution(
                gossip=round((category_totals["gossip"] / total_category) * 100, 1),
                unethical=round((category_totals["unethical"] / total_category) * 100, 1),
                waste=round((category_totals["waste"] / total_category) * 100, 1),
                productive=round((category_totals["productive"] / total_category) * 100, 1),
            )
        else:
            category_distribution = None
        
        # Format date context
        date_context = format_date_context(period, period_start, period_end)
        
        print(f"[REPORTS] {period.capitalize()} progress for user {uid}: {total_sessions} sessions, {total_listening_minutes} minutes")
        if category_distribution:
            print(f"[REPORTS] Category distribution: gossip={category_distribution.gossip}%, unethical={category_distribution.unethical}%, waste={category_distribution.waste}%, productive={category_distribution.productive}%")
        
        return ProgressReportResponse(
            period=period,
            dateContext=date_context,
            totalSessions=total_sessions,
            totalListeningMinutes=float(total_listening_minutes),
            totalFlagged=total_flagged,
            totalPositive=total_positive,
            averageMinutesPerSession=round(average_minutes_per_session, 2),
            periodStart=period_start,
            periodEnd=period_end,
            categoryDistribution=category_distribution,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[REPORTS] Error generating {period} progress report: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate {period} progress report"
        )


@router.get(
    "/chart/{period}",
    response_model=ChartDataResponse,
    summary="Get chart data",
    description="Returns time series data points for chart visualization for a specific time period.",
    responses={
        200: {
            "description": "Chart data retrieved successfully",
        },
        400: {
            "description": "Invalid period parameter",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
    },
)
def get_chart_data(
    period: Literal["today", "week", "month", "lifetime"],
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> ChartDataResponse:
    """Get chart data for a specific time period.
    
    Groups sessions by time resolution:
    - today: per hour (last 24 hours)
    - week: per day (last 7 days)
    - month: per week (last 4 weeks)
    - lifetime: per month (all months)
    
    Args:
        period: The time period to aggregate (today, week, month, lifetime)
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        ChartDataResponse: Time series data points for the chart
    """
    try:
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Calculate date range based on period
        period_end = datetime.now(timezone.utc)
        
        if period == "today":
            period_start = period_end - timedelta(hours=24)
            # Group by hour
            time_resolution = "hour"
        elif period == "week":
            period_start = period_end - timedelta(days=7)
            # Group by day
            time_resolution = "day"
        elif period == "month":
            period_start = period_end - timedelta(days=30)
            # Group by week
            time_resolution = "week"
        elif period == "lifetime":
            # Will determine from first session
            period_start = None
            time_resolution = "month"
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid period: {period}. Must be one of: today, week, month, lifetime"
            )
        
        # Query for all stopped sessions for this user
        sessions_query = db.collection('listening_sessions') \
            .where('uid', '==', uid) \
            .where('status', '==', 'STOPPED') \
            .stream()
        
        # Dictionary to aggregate minutes by time bucket (for non-today periods)
        time_buckets: Dict[str, float] = defaultdict(float)
        # List to store individual session points (for today period)
        individual_points: List[ChartDataPoint] = []
        earliest_session = None
        
        for doc in sessions_query:
            session_data = doc.to_dict()
            
            # Only process stopped sessions
            if session_data.get('status') != 'STOPPED':
                continue
            
            # Get startedAt timestamp
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
            
            # For lifetime, track earliest session
            if period == "lifetime":
                if earliest_session is None or started_at_dt < earliest_session:
                    earliest_session = started_at_dt
            else:
                # Filter by date range for other periods
                if started_at_dt < period_start or started_at_dt > period_end:
                    continue
            
            # Ensure totals is properly structured
            totals = session_data.get('totals', {})
            if not isinstance(totals, dict):
                totals = {}
            
            # Extract listening time
            total_seconds = totals.get('totalSeconds', 0)
            total_minutes = total_seconds / 60.0
            
            # For "today" period, return individual sessions (not grouped)
            if period == "today":
                # Store individual session data
                individual_points.append(ChartDataPoint(timestamp=started_at_dt, minutes=round(total_minutes, 1)))
            else:
                # Group by time resolution for other periods
                if time_resolution == "day":
                    # Group by day: YYYY-MM-DD
                    bucket_key = started_at_dt.strftime("%Y-%m-%d")
                elif time_resolution == "week":
                    # Group by week: Get Monday of the week
                    days_since_monday = started_at_dt.weekday()
                    monday = started_at_dt - timedelta(days=days_since_monday)
                    bucket_key = monday.strftime("%Y-%m-%d")
                elif time_resolution == "month":
                    # Group by month: YYYY-MM
                    bucket_key = started_at_dt.strftime("%Y-%m")
                
                time_buckets[bucket_key] += total_minutes
        
        # Convert buckets to sorted list of ChartDataPoint
        points: List[ChartDataPoint] = []
        
        if period == "today":
            # For today, use individual session points, sorted by timestamp
            points = sorted(individual_points, key=lambda p: p.timestamp)
        elif time_resolution == "day":
            # Generate all days in the last 7 days
            current = period_start
            while current <= period_end:
                bucket_key = current.strftime("%Y-%m-%d")
                minutes = time_buckets.get(bucket_key, 0.0)
                # Use start of day
                day_start = current.replace(hour=0, minute=0, second=0, microsecond=0)
                points.append(ChartDataPoint(timestamp=day_start, minutes=round(minutes, 1)))
                current += timedelta(days=1)
        elif time_resolution == "week":
            # Generate all weeks in the last 30 days (approximately 4 weeks)
            current = period_start
            seen_weeks = set()
            while current <= period_end:
                days_since_monday = current.weekday()
                monday = current - timedelta(days=days_since_monday)
                bucket_key = monday.strftime("%Y-%m-%d")
                if bucket_key not in seen_weeks:
                    seen_weeks.add(bucket_key)
                    minutes = time_buckets.get(bucket_key, 0.0)
                    points.append(ChartDataPoint(timestamp=monday, minutes=round(minutes, 1)))
                current += timedelta(days=7)
        elif time_resolution == "month":
            # For lifetime, use earliest session or default
            if period == "lifetime":
                if earliest_session:
                    period_start = earliest_session.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
                else:
                    period_start = (period_end - timedelta(days=365)).replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            
            # Generate all months from period_start to period_end
            current = period_start.replace(day=1, hour=0, minute=0, second=0, microsecond=0)
            while current <= period_end:
                bucket_key = current.strftime("%Y-%m")
                minutes = time_buckets.get(bucket_key, 0.0)
                points.append(ChartDataPoint(timestamp=current, minutes=round(minutes, 1)))
                # Move to next month
                if current.month == 12:
                    current = current.replace(year=current.year + 1, month=1)
                else:
                    current = current.replace(month=current.month + 1)
        
        # Sort points by timestamp
        points.sort(key=lambda x: x.timestamp)
        
        print(f"[REPORTS] Chart data for {period} period: {len(points)} data points")
        
        return ChartDataResponse(
            period=period,
            points=points,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[REPORTS] Error generating {period} chart data: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to generate {period} chart data"
        )

