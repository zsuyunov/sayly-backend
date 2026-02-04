"""Service for cleaning up old audio files and orphaned sessions."""

from datetime import datetime, timezone, timedelta
import os
from typing import List

from firebase_admin import firestore

# Audio files are kept for 30 days after analysis completes
AUDIO_RETENTION_DAYS = 30

# Failed uploads are kept for 7 days
FAILED_UPLOAD_RETENTION_DAYS = 7


def get_firestore_db():
    """Get Firestore database instance."""
    try:
        return firestore.client()
    except Exception as e:
        raise RuntimeError(f"Firestore not available: {e}")


def cleanup_old_audio_files(audio_storage_dir: str = "./audio_storage") -> int:
    """Clean up audio files older than retention period.
    
    Args:
        audio_storage_dir: Directory where audio files are stored
        
    Returns:
        Number of files deleted
    """
    if not os.path.exists(audio_storage_dir):
        return 0
    
    deleted_count = 0
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=AUDIO_RETENTION_DAYS)
    
    try:
        db = get_firestore_db()
        
        # Find sessions with audio files older than retention period
        sessions_query = db.collection('listening_sessions') \
            .where('analysisStatus', '==', 'COMPLETED') \
            .where('audioProcessed', '==', True) \
            .stream()
        
        for session_doc in sessions_query:
            session_data = session_doc.to_dict()
            audio_url = session_data.get('audioUrl')
            
            if not audio_url:
                continue
            
            # Check if file exists and get modification time
            if os.path.exists(audio_url):
                file_mtime = datetime.fromtimestamp(os.path.getmtime(audio_url), tz=timezone.utc)
                
                # Check if file is older than retention period
                if file_mtime < cutoff_date:
                    try:
                        os.remove(audio_url)
                        deleted_count += 1
                        print(f"[CLEANUP] Deleted old audio file: {audio_url}")
                    except Exception as e:
                        print(f"[CLEANUP] Failed to delete {audio_url}: {e}")
        
        # Also clean up orphaned files (files without sessions)
        if os.path.isdir(audio_storage_dir):
            for filename in os.listdir(audio_storage_dir):
                file_path = os.path.join(audio_storage_dir, filename)
                if os.path.isfile(file_path):
                    file_mtime = datetime.fromtimestamp(os.path.getmtime(file_path), tz=timezone.utc)
                    if file_mtime < cutoff_date:
                        # Check if this file is referenced by any session
                        sessions_with_file = db.collection('listening_sessions') \
                            .where('audioUrl', '==', file_path) \
                            .limit(1) \
                            .stream()
                        
                        if not list(sessions_with_file):
                            # Orphaned file - delete it
                            try:
                                os.remove(file_path)
                                deleted_count += 1
                                print(f"[CLEANUP] Deleted orphaned audio file: {file_path}")
                            except Exception as e:
                                print(f"[CLEANUP] Failed to delete orphaned file {file_path}: {e}")
    
    except Exception as e:
        print(f"[CLEANUP] Error cleaning up audio files: {e}")
    
    return deleted_count


def cleanup_failed_analysis_sessions() -> int:
    """Clean up sessions with failed analysis that are older than retention period.
    
    Returns:
        Number of sessions cleaned up
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=FAILED_UPLOAD_RETENTION_DAYS)
    cleaned_count = 0
    
    try:
        db = get_firestore_db()
        
        # Find failed analysis sessions older than retention period
        sessions_query = db.collection('listening_sessions') \
            .where('analysisStatus', '==', 'FAILED') \
            .stream()
        
        for session_doc in sessions_query:
            session_data = session_doc.to_dict()
            started_at = session_data.get('startedAt')
            
            if not started_at:
                continue
            
            # Convert Firestore timestamp to datetime if needed
            if hasattr(started_at, 'timestamp'):
                started_at_dt = datetime.fromtimestamp(started_at.timestamp(), tz=timezone.utc)
            elif isinstance(started_at, datetime):
                started_at_dt = started_at
                if started_at_dt.tzinfo is None:
                    started_at_dt = started_at_dt.replace(tzinfo=timezone.utc)
            else:
                continue
            
            if started_at_dt < cutoff_date:
                # Delete associated audio file if exists
                audio_url = session_data.get('audioUrl')
                if audio_url and os.path.exists(audio_url):
                    try:
                        os.remove(audio_url)
                        print(f"[CLEANUP] Deleted audio file for failed session: {audio_url}")
                    except Exception as e:
                        print(f"[CLEANUP] Failed to delete audio file: {e}")
                
                # Optionally delete the session document (or just mark for deletion)
                # For now, we'll keep the session but remove the audio file
                cleaned_count += 1
    
    except Exception as e:
        print(f"[CLEANUP] Error cleaning up failed sessions: {e}")
    
    return cleaned_count


def run_cleanup_job(audio_storage_dir: str = "./audio_storage") -> dict:
    """Run all cleanup jobs.
    
    Args:
        audio_storage_dir: Directory where audio files are stored
        
    Returns:
        Dict with cleanup statistics
    """
    print(f"[CLEANUP] Starting cleanup job at {datetime.now(timezone.utc)}")
    
    audio_files_deleted = cleanup_old_audio_files(audio_storage_dir)
    failed_sessions_cleaned = cleanup_failed_analysis_sessions()
    
    result = {
        'audio_files_deleted': audio_files_deleted,
        'failed_sessions_cleaned': failed_sessions_cleaned,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }
    
    print(f"[CLEANUP] Cleanup job completed: {result}")
    return result

