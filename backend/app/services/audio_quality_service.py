"""
Audio Quality Service for Enrollment Validation

This service validates audio quality before accepting enrollment samples.
Prevents poor quality audio from being stored as embeddings.

Rule-based validation (no ML):
- Duration check
- Silence ratio detection
- RMS amplitude (loudness) measurement
- Clipping detection
- Sample rate and channel validation
"""
import os
from typing import Dict, Any, List, Literal
from pydantic import BaseModel, Field
from pydub import AudioSegment
from pydub.silence import detect_nonsilent
import numpy as np


# Configurable thresholds (via environment variables)
MIN_DURATION_SECONDS = float(os.getenv("AUDIO_MIN_DURATION_SECONDS", "4.0"))
MAX_SILENCE_RATIO = float(os.getenv("AUDIO_MAX_SILENCE_RATIO", "0.30"))
MIN_RMS_THRESHOLD = float(os.getenv("AUDIO_MIN_RMS", "500.0"))
MAX_CLIPPING_RATIO = float(os.getenv("AUDIO_MAX_CLIPPING_RATIO", "0.05"))  # 5% max clipping
EXPECTED_SAMPLE_RATE = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))
SAMPLE_RATE_TOLERANCE = int(os.getenv("AUDIO_SAMPLE_RATE_TOLERANCE", "1000"))


# Rejection reason codes
class RejectionReason:
    TOO_SHORT = "TOO_SHORT"
    TOO_QUIET = "TOO_QUIET"
    TOO_SILENT = "TOO_SILENT"
    TOO_MUCH_CLIPPING = "TOO_MUCH_CLIPPING"
    INVALID_SAMPLE_RATE = "INVALID_SAMPLE_RATE"
    INVALID_CHANNELS = "INVALID_CHANNELS"
    LOAD_ERROR = "LOAD_ERROR"


class AudioQualityResult(BaseModel):
    """Structured audio quality validation result."""
    status: Literal["PASS", "FAIL"] = Field(..., description="Validation status")
    reasons: List[str] = Field(default_factory=list, description="Rejection reason codes (if FAIL)")
    metrics: Dict[str, Any] = Field(default_factory=dict, description="Quality metrics")
    message: str = Field(..., description="User-friendly message")


def calculate_silence_ratio(audio: AudioSegment, silence_thresh_db: int = -50) -> float:
    """Calculate the ratio of silence in audio.
    
    Args:
        audio: AudioSegment to analyze
        silence_thresh_db: Silence threshold in dBFS (default: -50)
        
    Returns:
        Ratio of silence (0.0 to 1.0)
    """
    if len(audio) == 0:
        return 1.0
    
    # Detect non-silent chunks
    nonsilent_chunks = detect_nonsilent(
        audio,
        min_silence_len=100,  # 100ms minimum silence
        silence_thresh=silence_thresh_db
    )
    
    if not nonsilent_chunks:
        return 1.0  # All silence
    
    # Calculate total non-silent duration
    nonsilent_duration = sum(end - start for start, end in nonsilent_chunks)
    total_duration = len(audio)
    
    # Silence ratio = (total - nonsilent) / total
    silence_ratio = 1.0 - (nonsilent_duration / total_duration)
    return max(0.0, min(1.0, silence_ratio))


def calculate_clipping_ratio(audio: AudioSegment) -> float:
    """Calculate the ratio of clipped samples in audio.
    
    Clipping occurs when samples reach the maximum/minimum value (typically ±32767 for 16-bit).
    
    Args:
        audio: AudioSegment to analyze
        
    Returns:
        Ratio of clipped samples (0.0 to 1.0)
    """
    if len(audio) == 0:
        return 0.0
    
    # Get raw audio data as numpy array
    samples = np.array(audio.get_array_of_samples())
    
    # For mono audio, samples is 1D. For stereo, it's interleaved.
    if audio.channels == 2:
        # Take only left channel (every other sample starting at 0)
        samples = samples[::2]
    
    # Find maximum possible value for this bit depth
    max_value = audio.max_possible_amplitude
    
    # Count clipped samples (at maximum or minimum)
    clipped_count = np.sum((samples >= max_value) | (samples <= -max_value))
    total_samples = len(samples)
    
    if total_samples == 0:
        return 0.0
    
    clipping_ratio = clipped_count / total_samples
    return float(clipping_ratio)


def validate_audio_quality(audio_path: str) -> AudioQualityResult:
    """Validate enrollment audio quality using rule-based checks.
    
    Validation rules:
    - duration >= MIN_DURATION_SECONDS (default: 8.0s)
    - silence_ratio <= MAX_SILENCE_RATIO (default: 0.30)
    - RMS >= MIN_RMS_THRESHOLD (default: 500.0)
    - clipping_ratio <= MAX_CLIPPING_RATIO (default: 0.05)
    - sample_rate within tolerance of EXPECTED_SAMPLE_RATE (default: 16000Hz)
    - channels == 1 (mono)
    
    Args:
        audio_path: Path to audio file to validate
        
    Returns:
        AudioQualityResult with status, reasons, metrics, and message
    """
    reasons: List[str] = []
    metrics: Dict[str, Any] = {}
    
    try:
        # Load audio
        audio = AudioSegment.from_file(audio_path)
        
        # Extract metrics
        duration_seconds = len(audio) / 1000.0
        silence_ratio = calculate_silence_ratio(audio)
        rms = audio.rms
        clipping_ratio = calculate_clipping_ratio(audio)
        sample_rate = audio.frame_rate
        channels = audio.channels
        
        # Store all metrics
        metrics = {
            "durationSeconds": round(duration_seconds, 2),
            "silenceRatio": round(silence_ratio, 3),
            "rms": round(rms, 2),
            "clippingRatio": round(clipping_ratio, 3),
            "sampleRate": sample_rate,
            "channels": channels,
            "maxPossibleAmplitude": audio.max_possible_amplitude,
        }
        
        # Validate duration
        if duration_seconds < MIN_DURATION_SECONDS:
            reasons.append(RejectionReason.TOO_SHORT)
        
        # Validate silence ratio
        if silence_ratio > MAX_SILENCE_RATIO:
            reasons.append(RejectionReason.TOO_SILENT)
        
        # Validate RMS/loudness
        if rms < MIN_RMS_THRESHOLD:
            reasons.append(RejectionReason.TOO_QUIET)
        
        # Validate clipping ratio
        if clipping_ratio > MAX_CLIPPING_RATIO:
            reasons.append(RejectionReason.TOO_MUCH_CLIPPING)
        
        # Validate sample rate
        if abs(sample_rate - EXPECTED_SAMPLE_RATE) > SAMPLE_RATE_TOLERANCE:
            reasons.append(RejectionReason.INVALID_SAMPLE_RATE)
        
        # Validate channels (must be mono)
        if channels != 1:
            reasons.append(RejectionReason.INVALID_CHANNELS)
        
        # Determine status and message
        if reasons:
            status = "FAIL"
            message = _build_user_message(reasons, metrics)
        else:
            status = "PASS"
            message = "Audio quality is good!"
        
        return AudioQualityResult(
            status=status,
            reasons=reasons,
            metrics=metrics,
            message=message
        )
        
    except Exception as e:
        # Error loading or analyzing audio
        reasons.append(RejectionReason.LOAD_ERROR)
        return AudioQualityResult(
            status="FAIL",
            reasons=reasons,
            metrics=metrics,
            message=f"Error analyzing audio: {str(e)}"
        )


def _build_user_message(reasons: List[str], metrics: Dict[str, Any]) -> str:
    """Build user-friendly error message from rejection reasons.
    
    Args:
        reasons: List of rejection reason codes
        metrics: Quality metrics dictionary
        
    Returns:
        User-friendly error message
    """
    messages = []
    
    if RejectionReason.TOO_SHORT in reasons:
        duration = metrics.get("durationSeconds", 0)
        messages.append(f"Recording too short ({duration:.1f}s). Please record for at least {MIN_DURATION_SECONDS}s.")
    
    if RejectionReason.TOO_SILENT in reasons:
        silence_pct = metrics.get("silenceRatio", 0) * 100
        messages.append(f"Too much silence detected ({silence_pct:.0f}%). Please speak more clearly.")
    
    if RejectionReason.TOO_QUIET in reasons:
        messages.append("Voice too quiet. Please speak louder or hold the phone closer.")
    
    if RejectionReason.TOO_MUCH_CLIPPING in reasons:
        clipping_pct = metrics.get("clippingRatio", 0) * 100
        messages.append(f"Audio is distorted ({clipping_pct:.1f}% clipping). Please speak at a normal volume.")
    
    if RejectionReason.INVALID_SAMPLE_RATE in reasons:
        sample_rate = metrics.get("sampleRate", 0)
        messages.append(f"Invalid audio format (sample rate: {sample_rate}Hz). Expected 16kHz.")
    
    if RejectionReason.INVALID_CHANNELS in reasons:
        channels = metrics.get("channels", 0)
        messages.append(f"Invalid audio format ({channels} channels). Expected mono (1 channel).")
    
    if RejectionReason.LOAD_ERROR in reasons:
        messages.append("Could not analyze audio file. Please try recording again.")
    
    if not messages:
        messages.append("Audio quality check failed. Please try recording again.")
    
    return " ".join(messages)


# Legacy function for backward compatibility
def validate_enrollment_audio(audio_path: str):
    """Legacy function - use validate_audio_quality instead."""
    result = validate_audio_quality(audio_path)
    
    # Convert to old format for backward compatibility
    from app.services.audio_quality_service import QualityReport
    return QualityReport(
        valid=(result.status == "PASS"),
        reason=result.message if result.status == "FAIL" else None,
        metrics=result.metrics
    )


# Legacy QualityReport class for backward compatibility
class QualityReport(BaseModel):
    """Legacy audio quality validation report (for backward compatibility)."""
    valid: bool = Field(..., description="Whether audio passes quality checks")
    reason: str | None = Field(None, description="Reason for rejection (if invalid)")
    metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Quality metrics (duration, silence ratio, RMS, etc.)"
    )
