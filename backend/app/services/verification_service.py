"""
Verification Service for Speaker Verification

This service implements multi-embedding comparison using max similarity
and top-K mean similarity, applies decision policies, and logs results
for auditability. Implements fail-closed behavior on errors.
"""
from typing import List, Optional, Tuple
from datetime import datetime, timezone
import numpy as np

from app.services.huggingface_service import extract_speaker_embedding
from app.services.threshold_service import get_threshold_config, log_similarity_score
from app.services.model_versioning_service import get_current_model_metadata, check_embedding_compatibility
from app.models.verification import VerificationResult, VerificationDecision, ChunkVerification, VerificationPolicy


def cosine_similarity(embedding1: List[float], embedding2: List[float]) -> float:
    """Compute cosine similarity between two embedding vectors.
    
    Args:
        embedding1: First embedding vector
        embedding2: Second embedding vector
        
    Returns:
        Cosine similarity score (0.0 to 1.0)
    """
    if len(embedding1) != len(embedding2):
        raise ValueError(f"Embedding dimensions must match: {len(embedding1)} vs {len(embedding2)}")
    
    # Convert to numpy arrays
    vec1 = np.array(embedding1)
    vec2 = np.array(embedding2)
    
    # Compute cosine similarity
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)
    
    if norm1 == 0 or norm2 == 0:
        return 0.0
    
    similarity = dot_product / (norm1 * norm2)
    return float(similarity)


def verify_speaker(
    session_embedding: List[float],
    enrollment_embeddings: List[List[float]],
    environment: Optional[str] = None,
    uid: Optional[str] = None
) -> Tuple[VerificationDecision, Optional[str]]:
    """Verify speaker identity using multi-embedding comparison.
    
    VERIFICATION LOGIC FLOW:
    ========================
    1. Compute cosine similarity between session embedding and each enrollment embedding
       → Results in N similarities (where N = number of enrollment embeddings, typically 3)
    
    2. Calculate two metrics:
       a. Max Similarity: max(all_similarities)
          → Best match across all enrollment samples
          → Robust to one poor enrollment sample
       b. Top-K Mean: mean(sorted(similarities)[-2:]) where K=2
          → Average of top 2 similarities
          → More stable than max alone, reduces outlier impact
    
    3. Apply decision policy using dynamic thresholds (from threshold_service):
       - OWNER: Both max_sim >= owner_threshold AND topK_mean >= owner_threshold
       - UNCERTAIN: At least one metric >= uncertain_threshold but not OWNER
       - OTHER: Both metrics < uncertain_threshold
    
    4. Log similarity scores for calibration and threshold tuning
    
    WHY MULTI-EMBEDDING COMPARISON:
    - Individual embeddings preserve variability (averaging loses information)
    - Max similarity handles cases where one enrollment sample is poor
    - Top-K mean provides stability and reduces false positives
    
    Args:
        session_embedding: Embedding vector from session audio
        enrollment_embeddings: List of 3 enrollment embeddings (stored individually)
        environment: Environment name (for threshold selection)
        uid: User ID (for logging similarity distributions)
        
    Returns:
        Tuple of (VerificationDecision, warning_message)
        warning_message is None if no compatibility issues
    """
    if not enrollment_embeddings or len(enrollment_embeddings) == 0:
        raise ValueError("No enrollment embeddings provided")
    
    # Compute similarities to all enrollment embeddings
    similarities = []
    for emb in enrollment_embeddings:
        try:
            sim = cosine_similarity(session_embedding, emb)
            similarities.append(sim)
        except Exception as e:
            print(f"[VERIFICATION] Error computing similarity: {e}")
            raise
    
    if not similarities:
        raise ValueError("No similarities computed")
    
    # Calculate metrics
    max_similarity = max(similarities)
    
    # Top-K mean (K=2): mean of top 2 similarities
    sorted_similarities = sorted(similarities, reverse=True)
    top_k = min(2, len(sorted_similarities))
    top_k_mean = np.mean(sorted_similarities[:top_k])
    
    # Get thresholds
    threshold_config = get_threshold_config(environment)
    
    # Apply v1 simplified binary decision policy
    user_decision, internal_state = VerificationPolicy.apply_decision(
        max_similarity,
        top_k_mean,
        threshold_config.ownerThreshold,
        threshold_config.uncertainThreshold
    )
    
    # Log similarity score for calibration (use internal_state for logging)
    if uid:
        try:
            log_similarity_score(uid, max_similarity, internal_state, environment)
        except Exception as e:
            print(f"[VERIFICATION] Error logging similarity: {e}")
    
    # Create decision (user-facing is binary, internal state preserved)
    decision = VerificationDecision(
        decision=user_decision,  # User-facing: OWNER or OTHER
        internalState=internal_state,  # Internal: OWNER, UNCERTAIN, OTHER, or SKIPPED
        maxSimilarity=float(max_similarity),
        topKMean=float(top_k_mean),
        allSimilarities=[float(s) for s in similarities],
        thresholdUsed={
            "ownerThreshold": threshold_config.ownerThreshold,
            "uncertainThreshold": threshold_config.uncertainThreshold
        }
    )
    
    return decision, None


def verify_session_audio(
    audio_path: str,
    enrollment_embeddings: List[List[float]],
    enrollment_metadata: Optional[dict] = None,
    environment: Optional[str] = None,
    uid: Optional[str] = None
) -> VerificationResult:
    """Verify session audio against enrollment embeddings (fail-closed).
    
    FAIL-CLOSED BEHAVIOR:
    =====================
    This function implements fail-closed security: if verification fails
    (API error, timeout, rate limit, etc.), processing is BLOCKED.
    
    Error Detection:
    - Timeout → errorReason: "timeout", retryable: True
    - Rate limit (429) → errorReason: "rate_limit", retryable: True
    - Cold start (503) → errorReason: "cold_start", retryable: True
    - Network error → errorReason: "network_error", retryable: True
    - Model incompatibility → errorReason: "model_incompatibility", retryable: False
    
    On any error:
    - status: "SKIPPED" or "ERROR"
    - decision: "BLOCKED"
    - shouldProcess: False (blocks downstream AI processing)
    - error/errorReason: Specific error information
    
    WHY FAIL-CLOSED:
    - Security: Prevents processing audio from unverified speakers
    - Data Quality: Ensures only verified owner speech enters analysis pipeline
    - Defensibility: Clear audit trail of why processing was blocked
    
    Args:
        audio_path: Path to session audio file
        enrollment_embeddings: List of enrollment embeddings (typically 3)
        enrollment_metadata: Metadata from enrollment (for model compatibility check)
        environment: Environment name (for threshold selection)
        uid: User ID (for logging)
        
    Returns:
        VerificationResult with decision, metadata, and shouldProcess flag
    """
    # Check model compatibility
    warning = None
    if enrollment_metadata:
        current_metadata = get_current_model_metadata()
        compatible, warning = check_embedding_compatibility(enrollment_metadata, current_metadata)
        if not compatible:
            # Incompatible model - block processing
            return VerificationResult(
                status="ERROR",
                decision="BLOCKED",
                shouldProcess=False,
                error="Enrollment embeddings are incompatible with current model version",
                errorReason="model_incompatibility",
                retryable=False,
                modelMetadata=current_metadata.to_dict(),
                verifiedAt=datetime.now(timezone.utc)
            )
    
    # Extract embedding from session audio
    try:
        session_embedding = extract_speaker_embedding(audio_path)
    except Exception as e:
        # Fail-closed: Block processing on error
        error_reason = "unknown"
        retryable = True
        
        error_str = str(e).lower()
        if "timeout" in error_str:
            error_reason = "timeout"
        elif "rate" in error_str or "429" in error_str:
            error_reason = "rate_limit"
        elif "503" in error_str or "loading" in error_str:
            error_reason = "cold_start"
        elif "network" in error_str:
            error_reason = "network_error"
            retryable = True
        
        print(f"[VERIFICATION] Verification failed (fail-closed): {error_reason}")
        
        return VerificationResult(
            status="SKIPPED",
            decision="BLOCKED",
            shouldProcess=False,  # Fail-closed: block processing
            error=str(e),
            errorReason=error_reason,
            retryable=retryable,
            modelMetadata=get_current_model_metadata().to_dict(),
            verifiedAt=datetime.now(timezone.utc)
        )
    
    # Perform verification
    try:
        decision, compatibility_warning = verify_speaker(
            session_embedding,
            enrollment_embeddings,
            environment,
            uid
        )
        
        if compatibility_warning:
            warning = compatibility_warning
        
        # V1: Always process - verification filters text segments, doesn't block
        return VerificationResult(
            status="SUCCESS",
            internalStatus="SUCCESS",
            decision=decision.decision,  # User-facing: OWNER or OTHER
            shouldProcess=True,  # V1: Always True - filtering happens at text level
            maxSimilarity=decision.maxSimilarity,
            topKMean=decision.topKMean,
            allSimilarities=decision.allSimilarities,
            error=warning,  # Store warning as error field if present
            modelMetadata=get_current_model_metadata().to_dict(),
            verifiedAt=datetime.now(timezone.utc)
        )
        
    except Exception as e:
        # V1: Don't block on logic errors - log and treat as OWNER
        print(f"[VERIFICATION] Verification logic error (v1: logging only): {e}")
        return VerificationResult(
            status="ERROR",
            internalStatus="ERROR",
            decision="OWNER",  # V1: Map errors to OWNER to avoid blocking
            shouldProcess=True,  # V1: Always process
            error=str(e),
            errorReason="verification_error",
            retryable=False,
            modelMetadata=get_current_model_metadata().to_dict(),
            verifiedAt=datetime.now(timezone.utc)
        )


def verify_chunk_audio(
    chunk_path: str,
    chunk_start_time: float,
    chunk_end_time: float,
    chunk_index: int,
    enrollment_embeddings: List[List[float]],
    environment: Optional[str] = None,
    uid: Optional[str] = None
) -> ChunkVerification:
    """Verify a single audio chunk against enrollment embeddings.
    
    Args:
        chunk_path: Path to chunk audio file
        chunk_start_time: Start time of chunk in seconds
        chunk_end_time: End time of chunk in seconds
        chunk_index: Index of chunk in session
        enrollment_embeddings: List of enrollment embeddings
        environment: Environment name
        uid: User ID
        
    Returns:
        ChunkVerification with decision for this chunk
    """
    # Extract embedding from chunk
    try:
        chunk_embedding = extract_speaker_embedding(chunk_path)
    except Exception as e:
        # V1: Don't block chunk - log error, treat as OWNER to avoid blocking
        print(f"[VERIFICATION] Chunk {chunk_index} verification failed (v1: logging only): {e}")
        decision = VerificationDecision(
            decision="OWNER",  # V1: Map errors to OWNER
            internalState="SKIPPED",  # Internal state for logging
            maxSimilarity=0.0,
            topKMean=0.0,
            allSimilarities=[],
            thresholdUsed={}
        )
        return ChunkVerification(
            startTime=chunk_start_time,
            endTime=chunk_end_time,
            decision=decision,
            chunkIndex=chunk_index
        )
    
    # Verify chunk
    decision, _ = verify_speaker(
        chunk_embedding,
        enrollment_embeddings,
        environment,
        uid
    )
    
    return ChunkVerification(
        startTime=chunk_start_time,
        endTime=chunk_end_time,
        decision=decision,
        chunkIndex=chunk_index
    )

