"""
Threshold Service for Dynamic Speaker Verification Thresholds

This service manages verification thresholds dynamically, allowing for
environment-specific configuration and calibration based on similarity distributions.
No hardcoded thresholds are used - all values come from configuration.
"""
import os
from typing import Dict, Any, Optional
from datetime import datetime, timezone
from firebase_admin import firestore

from app.models.threshold_config import ThresholdConfig, SimilarityDistribution


# Default thresholds (can be overridden by environment variables or Firestore)
DEFAULT_OWNER_THRESHOLD = float(os.getenv("VERIFICATION_OWNER_THRESHOLD", "0.75"))
DEFAULT_UNCERTAIN_THRESHOLD = float(os.getenv("VERIFICATION_UNCERTAIN_THRESHOLD", "0.6"))
DEFAULT_ENVIRONMENT = os.getenv("VERIFICATION_ENVIRONMENT", "dev")


def get_firestore_db():
    """Get Firestore database instance."""
    try:
        return firestore.client()
    except Exception as e:
        raise RuntimeError(f"Firestore not available: {e}")


def get_threshold_config(environment: Optional[str] = None) -> ThresholdConfig:
    """Get threshold configuration for the specified environment.
    
    Priority:
    1. Firestore configuration (if exists)
    2. Environment variables
    3. Default values
    
    Args:
        environment: Environment name (dev/test/prod). If None, uses DEFAULT_ENVIRONMENT.
        
    Returns:
        ThresholdConfig: The threshold configuration
    """
    env = environment or DEFAULT_ENVIRONMENT
    
    # Try to get from Firestore first
    try:
        db = get_firestore_db()
        config_ref = db.collection('verification_thresholds').document(env)
        config_doc = config_ref.get()
        
        if config_doc.exists:
            config_data = config_doc.to_dict()
            # Convert Firestore timestamp to datetime
            if 'calibratedAt' in config_data and hasattr(config_data['calibratedAt'], 'timestamp'):
                config_data['calibratedAt'] = datetime.fromtimestamp(
                    config_data['calibratedAt'].timestamp(),
                    tz=timezone.utc
                )
            return ThresholdConfig(**config_data)
    except Exception as e:
        print(f"[THRESHOLD] Error loading from Firestore: {e}, using defaults")
    
    # Fallback to environment variables or defaults
    owner_threshold = float(os.getenv(f"VERIFICATION_OWNER_THRESHOLD_{env.upper()}", str(DEFAULT_OWNER_THRESHOLD)))
    uncertain_threshold = float(os.getenv(f"VERIFICATION_UNCERTAIN_THRESHOLD_{env.upper()}", str(DEFAULT_UNCERTAIN_THRESHOLD)))
    
    return ThresholdConfig(
        environment=env,
        ownerThreshold=owner_threshold,
        uncertainThreshold=uncertain_threshold,
        calibratedAt=datetime.now(timezone.utc),
        similarityDistribution=None,
        notes="Default configuration from environment variables"
    )


def update_threshold_config(config: ThresholdConfig) -> None:
    """Update threshold configuration in Firestore.
    
    Args:
        config: The threshold configuration to store
    """
    try:
        db = get_firestore_db()
        config_ref = db.collection('verification_thresholds').document(config.environment)
        config_ref.set(config.dict(), merge=False)
        print(f"[THRESHOLD] Updated threshold configuration for {config.environment}")
    except Exception as e:
        print(f"[THRESHOLD] Error updating threshold configuration: {e}")
        raise


def log_similarity_score(uid: str, similarity: float, decision: str, environment: Optional[str] = None) -> None:
    """Log a similarity score for distribution analysis.
    
    This helps with threshold calibration by tracking actual similarity
    distributions per user and globally.
    
    Args:
        uid: User ID
        similarity: Similarity score (0.0 to 1.0)
        decision: Decision made (OWNER/UNCERTAIN/OTHER)
        environment: Environment name (optional)
    """
    try:
        db = get_firestore_db()
        env = environment or DEFAULT_ENVIRONMENT
        
        # Log per user
        user_log_ref = db.collection('verification_logs').document(uid).collection('scores')
        user_log_ref.add({
            'similarity': similarity,
            'decision': decision,
            'environment': env,
            'timestamp': firestore.SERVER_TIMESTAMP,
        })
        
        # Log globally (for distribution analysis)
        global_log_ref = db.collection('verification_logs').document('_global').collection('scores')
        global_log_ref.add({
            'similarity': similarity,
            'decision': decision,
            'environment': env,
            'uid': uid,
            'timestamp': firestore.SERVER_TIMESTAMP,
        })
        
    except Exception as e:
        # Non-critical - logging failure shouldn't break verification
        print(f"[THRESHOLD] Error logging similarity score: {e}")


def compute_similarity_distribution(uid: Optional[str] = None, environment: Optional[str] = None) -> SimilarityDistribution:
    """Compute similarity distribution statistics from logged scores.
    
    Args:
        uid: User ID (if None, computes global distribution)
        environment: Environment name (optional)
        
    Returns:
        SimilarityDistribution: Distribution statistics
    """
    try:
        db = get_firestore_db()
        env = environment or DEFAULT_ENVIRONMENT
        
        # Query scores
        if uid:
            scores_ref = db.collection('verification_logs').document(uid).collection('scores')
        else:
            scores_ref = db.collection('verification_logs').document('_global').collection('scores')
        
        # Filter by environment if specified
        query = scores_ref.where('environment', '==', env).order_by('timestamp', direction=firestore.Query.DESCENDING).limit(1000)
        scores_docs = query.stream()
        
        similarities = []
        for doc in scores_docs:
            data = doc.to_dict()
            if 'similarity' in data:
                similarities.append(data['similarity'])
        
        if not similarities:
            # Return default distribution if no data
            return SimilarityDistribution(
                mean=0.5,
                std=0.2,
                min=0.0,
                max=1.0,
                percentiles={'p50': 0.5, 'p75': 0.7, 'p95': 0.9},
                sampleCount=0
            )
        
        # Compute statistics
        import numpy as np
        similarities_array = np.array(similarities)
        
        percentiles = {
            'p25': float(np.percentile(similarities_array, 25)),
            'p50': float(np.percentile(similarities_array, 50)),
            'p75': float(np.percentile(similarities_array, 75)),
            'p95': float(np.percentile(similarities_array, 95)),
        }
        
        return SimilarityDistribution(
            mean=float(np.mean(similarities_array)),
            std=float(np.std(similarities_array)),
            min=float(np.min(similarities_array)),
            max=float(np.max(similarities_array)),
            percentiles=percentiles,
            sampleCount=len(similarities)
        )
        
    except Exception as e:
        print(f"[THRESHOLD] Error computing similarity distribution: {e}")
        # Return default on error
        return SimilarityDistribution(
            mean=0.5,
            std=0.2,
            min=0.0,
            max=1.0,
            percentiles={'p50': 0.5, 'p75': 0.7, 'p95': 0.9},
            sampleCount=0
        )

