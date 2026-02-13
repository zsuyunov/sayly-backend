from typing import Dict, Any, List
from fastapi import APIRouter, Depends, HTTPException, status
from datetime import datetime, timezone, timedelta
from collections import defaultdict

from app.auth.dependencies import get_current_user
from app.models.weekly_stats import WeeklyStatsResponse, DailyTotal, WeeklyCategoryDistribution
from app.models.monthly_stats import MonthlyStatsResponse, PerDayTotal, MonthlyCategoryDistribution, MonthlyComparisonResponse
from app.models.lifetime_stats import LifetimeStatsResponse, MonthlyAverage, LifetimeCategoryDistribution
from firebase_admin import firestore

router = APIRouter(
    prefix="/stats",
    tags=["stats"],
)


def get_firestore_db():
    """Get Firestore database instance."""
    try:
        return firestore.client()
    except Exception as e:
        raise RuntimeError(f"Firestore not available: {e}")


def get_week_start_end():
    """Get the start (Monday) and end (Sunday) of the current week."""
    now = datetime.now(timezone.utc)
    # Get Monday of current week (weekday() returns 0 for Monday, 6 for Sunday)
    days_since_monday = now.weekday()
    week_start = (now - timedelta(days=days_since_monday)).replace(hour=0, minute=0, second=0, microsecond=0)
    week_end = (week_start + timedelta(days=6)).replace(hour=23, minute=59, second=59, microsecond=999999)
    return week_start, week_end


@router.get(
    "/weekly",
    response_model=WeeklyStatsResponse,
    summary="Get weekly statistics",
    description="Returns aggregated weekly statistics including daily totals and category distribution. Week starts Monday.",
    responses={
        200: {
            "description": "Weekly statistics retrieved successfully",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
    },
)
def get_weekly_stats(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> WeeklyStatsResponse:
    """Get weekly statistics for the current user.
    
    Calculates statistics for the current week (Monday to Sunday), including:
    - Total sessions in the week
    - Total listening minutes in the week
    - Daily totals for each day (Monday through Sunday)
    - Weekly category distribution (gossip, unethical, waste, productive)
    
    Only includes sessions with status STOPPED.
    Week starts on Monday.
    
    Args:
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        WeeklyStatsResponse: Aggregated weekly statistics with daily breakdown
    """
    try:
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Get week boundaries (Monday to Sunday)
        week_start, week_end = get_week_start_end()
        
        # Query for all stopped sessions for this user
        sessions_query = db.collection('listening_sessions') \
            .where('uid', '==', uid) \
            .where('status', '==', 'STOPPED') \
            .stream()
        
        # Aggregate statistics
        total_sessions_week = 0
        total_listening_seconds_week = 0
        
        # Daily totals: date string (YYYY-MM-DD) -> {minutes, sessions}
        daily_totals_dict: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"minutes": 0.0, "sessions": 0})
        
        # Category totals (for now, these will be placeholders until AI analysis is available)
        category_totals = {
            "gossip": 0,
            "unethical": 0,
            "waste": 0,
            "productive": 0,
        }
        
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
            
            # Filter by week range
            if started_at_dt < week_start or started_at_dt > week_end:
                continue
            
            # Ensure totals is properly structured
            totals = session_data.get('totals', {})
            if not isinstance(totals, dict):
                totals = {}
            
            # Extract listening time
            total_seconds = totals.get('totalSeconds', 0)
            total_minutes = total_seconds / 60.0
            
            # Aggregate week totals
            total_sessions_week += 1
            total_listening_seconds_week += total_seconds
            
            # Aggregate daily totals
            date_key = started_at_dt.date().isoformat()  # YYYY-MM-DD
            daily_totals_dict[date_key]["minutes"] += total_minutes
            daily_totals_dict[date_key]["sessions"] += 1
            
            # Extract category data from AI classification
            classification = session_data.get('classification', {})
            if classification and isinstance(classification, dict):
                # Map classification labels to our category names
                gossip_score = classification.get('gossip', 0.0)
                unethical_score = classification.get('insult or unethical speech', 0.0)
                waste_score = classification.get('wasteful talk', 0.0)
                productive_score = classification.get('productive or meaningful speech', 0.0)
                
                # Add scores to category totals (weighted by session duration in minutes)
                category_totals["gossip"] += gossip_score * total_minutes
                category_totals["unethical"] += unethical_score * total_minutes
                category_totals["waste"] += waste_score * total_minutes
                category_totals["productive"] += productive_score * total_minutes
        
        # Convert total seconds to minutes
        total_listening_minutes_week = total_listening_seconds_week / 60.0
        
        # Generate daily totals for all 7 days of the week (Monday through Sunday)
        daily_totals: List[DailyTotal] = []
        current_date = week_start.date()
        for i in range(7):
            date_key = current_date.isoformat()
            daily_data = daily_totals_dict.get(date_key, {"minutes": 0.0, "sessions": 0})
            daily_totals.append(DailyTotal(
                date=date_key,
                minutes=round(daily_data["minutes"], 1),
                sessions=daily_data["sessions"]
            ))
            current_date += timedelta(days=1)
        
        # Calculate category distribution percentages
        # For now, all zeros until AI analysis is available
        total_category = sum(category_totals.values())
        if total_category > 0:
            category_distribution = WeeklyCategoryDistribution(
                gossip=round((category_totals["gossip"] / total_category) * 100, 1),
                unethical=round((category_totals["unethical"] / total_category) * 100, 1),
                waste=round((category_totals["waste"] / total_category) * 100, 1),
                productive=round((category_totals["productive"] / total_category) * 100, 1),
            )
        else:
            category_distribution = WeeklyCategoryDistribution(
                gossip=0.0,
                unethical=0.0,
                waste=0.0,
                productive=0.0,
            )
        
        print(f"[STATS] Weekly stats for user {uid}: {total_sessions_week} sessions, {total_listening_minutes_week:.2f} minutes")
        
        return WeeklyStatsResponse(
            total_sessions_week=total_sessions_week,
            total_listening_minutes_week=round(total_listening_minutes_week, 1),
            daily_totals=daily_totals,
            weekly_category_distribution=category_distribution,
            week_start=week_start,
            week_end=week_end,
        )
        
    except Exception as e:
        print(f"[STATS] Error generating weekly stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate weekly statistics"
        )


def get_month_start_end(year: int, month: int):
    """Get the start and end of a specific month."""
    from calendar import monthrange
    month_start = datetime(year, month, 1, 0, 0, 0, 0, tzinfo=timezone.utc)
    last_day = monthrange(year, month)[1]
    month_end = datetime(year, month, last_day, 23, 59, 59, 999999, tzinfo=timezone.utc)
    return month_start, month_end


@router.get(
    "/monthly",
    response_model=MonthlyStatsResponse,
    summary="Get monthly statistics",
    description="Returns aggregated monthly statistics including daily totals and category distribution. Supports month selection via query parameters.",
    responses={
        200: {
            "description": "Monthly statistics retrieved successfully",
        },
        400: {
            "description": "Invalid year or month parameter",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
    },
)
def get_monthly_stats(
    year: int = None,
    month: int = None,
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> MonthlyStatsResponse:
    """Get monthly statistics for the current user.
    
    Calculates statistics for a specific month (defaults to current month), including:
    - Total sessions in the month
    - Total listening minutes in the month
    - Daily totals for each day of the month
    - Monthly category distribution (gossip, unethical, waste, productive)
    
    Only includes sessions with status STOPPED.
    
    Args:
        year: Year (YYYY format). Defaults to current year if not provided.
        month: Month (1-12). Defaults to current month if not provided.
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        MonthlyStatsResponse: Aggregated monthly statistics with daily breakdown
    """
    try:
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Get month boundaries (default to current month if not specified)
        now = datetime.now(timezone.utc)
        target_year = year if year is not None else now.year
        target_month = month if month is not None else now.month
        
        # Validate month
        if target_month < 1 or target_month > 12:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Invalid month. Must be between 1 and 12."
            )
        
        month_start, month_end = get_month_start_end(target_year, target_month)
        
        # Query for all stopped sessions for this user
        sessions_query = db.collection('listening_sessions') \
            .where('uid', '==', uid) \
            .where('status', '==', 'STOPPED') \
            .stream()
        
        # Aggregate statistics
        total_sessions_month = 0
        total_listening_seconds_month = 0
        
        # Daily totals: date string (YYYY-MM-DD) -> {minutes, sessions}
        daily_totals_dict: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"minutes": 0.0, "sessions": 0})
        
        # Category totals (for now, these will be placeholders until AI analysis is available)
        category_totals = {
            "gossip": 0,
            "unethical": 0,
            "waste": 0,
            "productive": 0,
        }
        
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
            
            # Filter by month range
            if started_at_dt < month_start or started_at_dt > month_end:
                continue
            
            # Ensure totals is properly structured
            totals = session_data.get('totals', {})
            if not isinstance(totals, dict):
                totals = {}
            
            # Extract listening time
            total_seconds = totals.get('totalSeconds', 0)
            total_minutes = total_seconds / 60.0
            
            # Aggregate month totals
            total_sessions_month += 1
            total_listening_seconds_month += total_seconds
            
            # Aggregate daily totals
            date_key = started_at_dt.date().isoformat()  # YYYY-MM-DD
            daily_totals_dict[date_key]["minutes"] += total_minutes
            daily_totals_dict[date_key]["sessions"] += 1
            
            # Extract category data from AI classification
            classification = session_data.get('classification', {})
            if classification and isinstance(classification, dict):
                # Map classification labels to our category names
                gossip_score = classification.get('gossip', 0.0)
                unethical_score = classification.get('insult or unethical speech', 0.0)
                waste_score = classification.get('wasteful talk', 0.0)
                productive_score = classification.get('productive or meaningful speech', 0.0)
                
                # Add scores to category totals (weighted by session duration in minutes)
                category_totals["gossip"] += gossip_score * total_minutes
                category_totals["unethical"] += unethical_score * total_minutes
                category_totals["waste"] += waste_score * total_minutes
                category_totals["productive"] += productive_score * total_minutes
        
        # Convert total seconds to minutes
        total_listening_minutes_month = total_listening_seconds_month / 60.0
        
        # Generate daily totals for all days of the month
        from calendar import monthrange
        per_day_totals: List[PerDayTotal] = []
        num_days = monthrange(target_year, target_month)[1]
        current_date = month_start.date()
        
        for i in range(num_days):
            date_key = current_date.isoformat()
            daily_data = daily_totals_dict.get(date_key, {"minutes": 0.0, "sessions": 0})
            per_day_totals.append(PerDayTotal(
                date=date_key,
                minutes=round(daily_data["minutes"], 1),
                sessions=daily_data["sessions"]
            ))
            current_date += timedelta(days=1)
        
        # Calculate category distribution percentages
        # For now, all zeros until AI analysis is available
        total_category = sum(category_totals.values())
        if total_category > 0:
            category_distribution = MonthlyCategoryDistribution(
                gossip=round((category_totals["gossip"] / total_category) * 100, 1),
                unethical=round((category_totals["unethical"] / total_category) * 100, 1),
                waste=round((category_totals["waste"] / total_category) * 100, 1),
                productive=round((category_totals["productive"] / total_category) * 100, 1),
            )
        else:
            category_distribution = MonthlyCategoryDistribution(
                gossip=0.0,
                unethical=0.0,
                waste=0.0,
                productive=0.0,
            )
        
        # Format month name
        month_names = ["", "January", "February", "March", "April", "May", "June",
                      "July", "August", "September", "October", "November", "December"]
        month_name = f"{month_names[target_month]} {target_year}"
        
        print(f"[STATS] Monthly stats for user {uid}, {month_name}: {total_sessions_month} sessions, {total_listening_minutes_month:.2f} minutes")
        
        return MonthlyStatsResponse(
            total_sessions_month=total_sessions_month,
            total_listening_minutes_month=round(total_listening_minutes_month, 1),
            per_day_totals=per_day_totals,
            month_category_distribution=category_distribution,
            month_start=month_start,
            month_end=month_end,
            month_name=month_name,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[STATS] Error generating monthly stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate monthly statistics"
        )


@router.get(
    "/lifetime",
    response_model=LifetimeStatsResponse,
    summary="Get lifetime statistics",
    description="Returns aggregated lifetime statistics including monthly averages and category distribution.",
    responses={
        200: {
            "description": "Lifetime statistics retrieved successfully",
        },
        401: {
            "description": "Unauthorized - Invalid or missing authentication token",
        },
    },
)
def get_lifetime_stats(
    current_user: Dict[str, Any] = Depends(get_current_user)
) -> LifetimeStatsResponse:
    """Get lifetime statistics for the current user.
    
    Calculates statistics for all time since account creation, including:
    - Total sessions in lifetime
    - Total listening minutes in lifetime
    - Active days (days with at least one session)
    - Missed days (days without sessions since signup)
    - Monthly average series (monthly averages since signup)
    - Lifetime category distribution (gossip, unethical, waste, productive)
    
    Only includes sessions with status STOPPED.
    
    Args:
        current_user: The authenticated user object (injected via dependency)
        
    Returns:
        LifetimeStatsResponse: Aggregated lifetime statistics with monthly breakdown
    """
    try:
        uid = current_user["uid"]
        db = get_firestore_db()
        
        # Get user profile to find account creation date
        user_doc = db.collection('users').document(uid).get()
        if not user_doc.exists:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="User profile not found"
            )
        
        user_data = user_doc.to_dict()
        account_created_at = user_data.get('createdAt')
        if account_created_at:
            if hasattr(account_created_at, 'timestamp'):
                account_created_at = datetime.fromtimestamp(account_created_at.timestamp(), tz=timezone.utc)
            elif isinstance(account_created_at, datetime):
                if account_created_at.tzinfo is None:
                    account_created_at = account_created_at.replace(tzinfo=timezone.utc)
        else:
            # Fallback to a reasonable default (1 year ago)
            account_created_at = datetime.now(timezone.utc) - timedelta(days=365)
        
        # Query for all stopped sessions for this user
        sessions_query = db.collection('listening_sessions') \
            .where('uid', '==', uid) \
            .where('status', '==', 'STOPPED') \
            .stream()
        
        # Aggregate statistics
        total_sessions = 0
        total_listening_seconds = 0
        first_session_date = None
        
        # Track sessions by date
        sessions_by_date: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"minutes": 0.0, "sessions": 0})
        
        # Track sessions by month for monthly averages
        sessions_by_month: Dict[str, Dict[str, Any]] = defaultdict(lambda: {"minutes": 0.0, "sessions": 0, "days": set()})
        
        # Category totals (for now, these will be placeholders until AI analysis is available)
        category_totals = {
            "gossip": 0,
            "unethical": 0,
            "waste": 0,
            "productive": 0,
        }
        
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
            
            # Track first session
            if first_session_date is None or started_at_dt < first_session_date:
                first_session_date = started_at_dt
            
            # Ensure totals is properly structured
            totals = session_data.get('totals', {})
            if not isinstance(totals, dict):
                totals = {}
            
            # Extract listening time
            total_seconds = totals.get('totalSeconds', 0)
            total_minutes = total_seconds / 60.0
            
            # Aggregate lifetime totals
            total_sessions += 1
            total_listening_seconds += total_seconds
            
            # Aggregate by date
            date_key = started_at_dt.date().isoformat()  # YYYY-MM-DD
            sessions_by_date[date_key]["minutes"] += total_minutes
            sessions_by_date[date_key]["sessions"] += 1
            
            # Aggregate by month
            month_key = started_at_dt.strftime("%Y-%m")  # YYYY-MM
            sessions_by_month[month_key]["minutes"] += total_minutes
            sessions_by_month[month_key]["sessions"] += 1
            sessions_by_month[month_key]["days"].add(date_key)
            
            # Extract category data from AI classification
            classification = session_data.get('classification', {})
            if classification and isinstance(classification, dict):
                # Map classification labels to our category names
                gossip_score = classification.get('gossip', 0.0)
                unethical_score = classification.get('insult or unethical speech', 0.0)
                waste_score = classification.get('wasteful talk', 0.0)
                productive_score = classification.get('productive or meaningful speech', 0.0)
                
                # Add scores to category totals (weighted by session duration in minutes)
                category_totals["gossip"] += gossip_score * total_minutes
                category_totals["unethical"] += unethical_score * total_minutes
                category_totals["waste"] += waste_score * total_minutes
                category_totals["productive"] += productive_score * total_minutes
        
        # Convert total seconds to minutes
        total_listening_minutes = total_listening_seconds / 60.0
        
        # Calculate active days
        active_days = len(sessions_by_date)
        
        # Calculate missed days (days since account creation without sessions)
        now = datetime.now(timezone.utc)
        days_since_signup = (now.date() - account_created_at.date()).days + 1
        missed_days = max(0, days_since_signup - active_days)
        
        # Generate monthly average series
        monthly_average_series: List[MonthlyAverage] = []
        month_names = ["", "January", "February", "March", "April", "May", "June",
                      "July", "August", "September", "October", "November", "December"]
        
        # Sort months chronologically
        sorted_months = sorted(sessions_by_month.keys())
        
        for month_key in sorted_months:
            year, month = map(int, month_key.split('-'))
            month_data = sessions_by_month[month_key]
            days_in_month = len(month_data["days"])
            
            # Calculate average minutes per day for this month
            average_minutes_per_day = month_data["minutes"] / days_in_month if days_in_month > 0 else 0.0
            
            monthly_average_series.append(MonthlyAverage(
                year=year,
                month=month,
                month_name=f"{month_names[month]} {year}",
                average_minutes_per_day=round(average_minutes_per_day, 1),
                total_sessions=month_data["sessions"],
                total_minutes=round(month_data["minutes"], 1),
            ))
        
        # Calculate category distribution percentages
        # For now, all zeros until AI analysis is available
        total_category = sum(category_totals.values())
        if total_category > 0:
            category_distribution = LifetimeCategoryDistribution(
                gossip=round((category_totals["gossip"] / total_category) * 100, 1),
                unethical=round((category_totals["unethical"] / total_category) * 100, 1),
                waste=round((category_totals["waste"] / total_category) * 100, 1),
                productive=round((category_totals["productive"] / total_category) * 100, 1),
            )
        else:
            category_distribution = LifetimeCategoryDistribution(
                gossip=0.0,
                unethical=0.0,
                waste=0.0,
                productive=0.0,
            )
        
        print(f"[STATS] Lifetime stats for user {uid}: {total_sessions} sessions, {total_listening_minutes:.2f} minutes, {active_days} active days")
        
        return LifetimeStatsResponse(
            total_sessions=total_sessions,
            total_listening_minutes=round(total_listening_minutes, 1),
            active_days=active_days,
            missed_days=missed_days,
            monthly_average_series=monthly_average_series,
            lifetime_category_distribution=category_distribution,
            account_created_at=account_created_at,
            first_session_date=first_session_date,
        )
        
    except HTTPException:
        raise
    except Exception as e:
        print(f"[STATS] Error generating lifetime stats: {e}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Failed to generate lifetime statistics"
        )

