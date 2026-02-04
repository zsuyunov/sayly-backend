"""Service for audio normalization and processing."""

import os
from typing import Dict, Any, Optional
from pydub import AudioSegment
from pydub.effects import normalize

# Target format for AI processing
TARGET_SAMPLE_RATE = 16000  # 16kHz for Whisper and speaker diarization
TARGET_CHANNELS = 1  # Mono
TARGET_FORMAT = "wav"


def normalize_audio(input_path: str, output_path: str) -> str:
    """Normalize audio file to AI-ready format (16kHz mono WAV).
    
    This function:
    1. Loads the audio file
    2. Converts to mono if stereo
    3. Resamples to 16kHz
    4. Normalizes volume levels
    5. Removes leading/trailing silence
    6. Saves as WAV format
    
    Args:
        input_path: Path to input audio file
        output_path: Path to save normalized audio file
        
    Returns:
        Path to normalized audio file
        
    Raises:
        Exception: If audio processing fails
    """
    try:
        print(f"[AUDIO_SERVICE] Normalizing audio: {input_path} -> {output_path}")
        
        # Load audio file
        # pydub automatically detects format from extension
        audio = AudioSegment.from_file(input_path)
        
        # Convert to mono if stereo
        if audio.channels > TARGET_CHANNELS:
            print(f"[AUDIO_SERVICE] Converting from {audio.channels} channels to mono")
            audio = audio.set_channels(TARGET_CHANNELS)
        
        # Resample to target sample rate if needed
        if audio.frame_rate != TARGET_SAMPLE_RATE:
            print(f"[AUDIO_SERVICE] Resampling from {audio.frame_rate}Hz to {TARGET_SAMPLE_RATE}Hz")
            audio = audio.set_frame_rate(TARGET_SAMPLE_RATE)
        
        # Normalize volume levels (peak normalization)
        # This ensures consistent volume across all recordings
        audio = normalize(audio)
        
        # Remove leading and trailing silence
        # Trim silence with threshold of -50dBFS (adjustable)
        audio = audio.strip_silence(silence_len=100, silence_thresh=-50)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(output_path) if os.path.dirname(output_path) else ".", exist_ok=True)
        
        # Export as WAV
        audio.export(output_path, format=TARGET_FORMAT)
        
        print(f"[AUDIO_SERVICE] Successfully normalized audio: {output_path}")
        return output_path
        
    except Exception as e:
        print(f"[AUDIO_SERVICE] Error normalizing audio: {e}")
        raise Exception(f"Failed to normalize audio: {str(e)}")


def extract_audio_metadata(file_path: str) -> Dict[str, Any]:
    """Extract detailed audio metadata using pydub.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        Dict with sample_rate, channels, duration_seconds, format, and bit_depth
    """
    metadata = {
        'sampleRate': None,
        'channels': None,
        'durationSeconds': None,
        'format': None,
        'bitDepth': None,
        'frameRate': None,
    }
    
    try:
        audio = AudioSegment.from_file(file_path)
        metadata['sampleRate'] = audio.frame_rate
        metadata['channels'] = audio.channels
        metadata['durationSeconds'] = len(audio) / 1000.0  # pydub returns duration in milliseconds
        metadata['format'] = os.path.splitext(file_path)[1][1:].upper()  # Get extension without dot
        metadata['frameRate'] = audio.frame_rate
        # Bit depth is not directly available in pydub, but WAV files are typically 16-bit
        metadata['bitDepth'] = 16 if metadata['format'] == 'WAV' else None
        
    except Exception as e:
        print(f"[AUDIO_SERVICE] Error extracting metadata: {e}")
    
    return metadata


def validate_audio_format(file_path: str) -> bool:
    """Validate that audio file is in correct format for AI processing.
    
    Args:
        file_path: Path to audio file
        
    Returns:
        True if format is valid (WAV, 16kHz, mono), False otherwise
    """
    try:
        audio = AudioSegment.from_file(file_path)
        
        is_valid = (
            audio.frame_rate == TARGET_SAMPLE_RATE and
            audio.channels == TARGET_CHANNELS and
            os.path.splitext(file_path)[1].lower() == '.wav'
        )
        
        if not is_valid:
            print(f"[AUDIO_SERVICE] Audio format validation failed:")
            print(f"  Sample rate: {audio.frame_rate}Hz (expected {TARGET_SAMPLE_RATE}Hz)")
            print(f"  Channels: {audio.channels} (expected {TARGET_CHANNELS})")
            print(f"  Format: {os.path.splitext(file_path)[1]} (expected .wav)")
        
        return is_valid
        
    except Exception as e:
        print(f"[AUDIO_SERVICE] Error validating audio format: {e}")
        return False

