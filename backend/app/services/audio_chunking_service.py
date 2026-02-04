"""
Audio Chunking Service for Session Processing

This service splits session audio into 10-15 second chunks for
chunk-level speaker verification. Preserves timing metadata.
"""
import os
import tempfile
from typing import List, NamedTuple, Optional
from pydub import AudioSegment


# Default chunk duration (configurable)
DEFAULT_CHUNK_DURATION_SECONDS = 12.0  # 12 seconds per chunk
CHUNK_OVERLAP_SECONDS = 0.5  # 0.5 second overlap to avoid cutting words


class AudioChunk(NamedTuple):
    """Represents a single audio chunk."""
    path: str  # Path to chunk file
    start_time: float  # Start time in seconds (relative to original audio)
    end_time: float  # End time in seconds (relative to original audio)
    index: int  # Chunk index (0-based)
    duration: float  # Chunk duration in seconds


def split_audio(
    audio_path: str,
    chunk_duration: float = DEFAULT_CHUNK_DURATION_SECONDS,
    output_dir: Optional[str] = None,
    overlap: float = CHUNK_OVERLAP_SECONDS
) -> List[AudioChunk]:
    """Split audio file into chunks of specified duration.
    
    CHUNKING RATIONALE:
    ===================
    Chunk-level verification enables:
    1. Detection of speaker changes mid-session
    2. More granular control (process only OWNER chunks)
    3. Better handling of long sessions
    4. Per-chunk audit trail
    
    CHUNK DURATION:
    - 10-15 seconds is optimal for speaker verification
    - Too short (< 5s): Insufficient audio for reliable embedding
    - Too long (> 20s): May miss speaker changes
    
    OVERLAP:
    - 0.5s overlap prevents cutting words at boundaries
    - Trade-off: More chunks but better quality
    
    TIMING METADATA:
    - Each chunk stores start_time and end_time relative to original audio
    - Enables reconstruction and audit trail
    
    Args:
        audio_path: Path to input audio file
        chunk_duration: Duration of each chunk in seconds (default: 12.0, range: 10-15s)
        output_dir: Directory to save chunk files (if None, uses temp directory)
        overlap: Overlap between chunks in seconds (default: 0.5)
        
    Returns:
        List of AudioChunk objects with paths and timing metadata
    """
    # Load audio
    audio = AudioSegment.from_file(audio_path)
    total_duration_ms = len(audio)
    total_duration_seconds = total_duration_ms / 1000.0
    
    # Create output directory
    if output_dir is None:
        output_dir = tempfile.mkdtemp(prefix="audio_chunks_")
    else:
        os.makedirs(output_dir, exist_ok=True)
    
    chunks = []
    chunk_index = 0
    current_start_ms = 0
    chunk_duration_ms = int(chunk_duration * 1000)
    overlap_ms = int(overlap * 1000)
    
    # Extract base filename for chunk naming
    base_name = os.path.splitext(os.path.basename(audio_path))[0]
    
    while current_start_ms < total_duration_ms:
        # Calculate chunk end (with overlap)
        chunk_end_ms = min(current_start_ms + chunk_duration_ms, total_duration_ms)
        
        # Extract chunk
        chunk_audio = audio[current_start_ms:chunk_end_ms]
        
        # Skip if chunk is too short (less than 1 second)
        if len(chunk_audio) < 1000:
            break
        
        # Save chunk to file
        chunk_filename = f"{base_name}_chunk_{chunk_index:04d}.wav"
        chunk_path = os.path.join(output_dir, chunk_filename)
        chunk_audio.export(chunk_path, format="wav")
        
        # Calculate timing
        start_time = current_start_ms / 1000.0
        end_time = chunk_end_ms / 1000.0
        duration = (chunk_end_ms - current_start_ms) / 1000.0
        
        chunks.append(AudioChunk(
            path=chunk_path,
            start_time=start_time,
            end_time=end_time,
            index=chunk_index,
            duration=duration
        ))
        
        # Move to next chunk (with overlap)
        current_start_ms = chunk_end_ms - overlap_ms
        chunk_index += 1
        
        # Safety check to avoid infinite loop
        if chunk_index > 1000:
            print(f"[CHUNKING] Warning: Too many chunks, stopping at {chunk_index}")
            break
    
    print(f"[CHUNKING] Split audio into {len(chunks)} chunks (total duration: {total_duration_seconds:.1f}s)")
    return chunks


def cleanup_chunks(chunks: List[AudioChunk]) -> None:
    """Clean up chunk files.
    
    Args:
        chunks: List of AudioChunk objects to clean up
    """
    for chunk in chunks:
        try:
            if os.path.exists(chunk.path):
                os.remove(chunk.path)
        except Exception as e:
            print(f"[CHUNKING] Warning: Could not delete chunk {chunk.path}: {e}")
    
    # Try to remove directory if it's empty
    if chunks:
        try:
            chunk_dir = os.path.dirname(chunks[0].path)
            if os.path.exists(chunk_dir) and chunk_dir.startswith(tempfile.gettempdir()):
                os.rmdir(chunk_dir)
        except Exception:
            pass  # Directory not empty or doesn't exist


def reconstruct_audio_from_chunks(
    chunks: List[AudioChunk],
    decisions: List[str],
    output_path: str,
    include_uncertain: bool = True
) -> None:
    """Reconstruct audio from chunks based on verification decisions.
    
    RECONSTRUCTION LOGIC:
    =====================
    Only OWNER (and optionally UNCERTAIN) chunks are included in reconstruction.
    OTHER chunks are discarded to prevent non-owner speech from entering AI pipeline.
    
    Decision Mapping:
    - OWNER chunks → Included (full processing)
    - UNCERTAIN chunks → Included if include_uncertain=True (flagged processing)
    - OTHER chunks → Discarded (not processed)
    - SKIPPED chunks → Discarded (verification failed)
    
    WHY RECONSTRUCT:
    - Enables processing only verified audio segments
    - Maintains audio continuity for STT/analysis
    - Preserves timing information for audit
    
    Args:
        chunks: List of AudioChunk objects (must match decisions order)
        decisions: List of decision strings (OWNER/UNCERTAIN/OTHER/SKIPPED) for each chunk
        output_path: Path to save reconstructed audio file
        include_uncertain: Whether to include UNCERTAIN chunks (default: True)
    """
    if len(chunks) != len(decisions):
        raise ValueError(f"Chunks ({len(chunks)}) and decisions ({len(decisions)}) count mismatch")
    
    # Filter chunks based on decisions
    included_chunks = []
    for chunk, decision in zip(chunks, decisions):
        if decision == "OWNER" or (decision == "UNCERTAIN" and include_uncertain):
            included_chunks.append(chunk)
    
    if not included_chunks:
        # No chunks to include - create empty audio file
        empty_audio = AudioSegment.silent(duration=100)  # 100ms silence
        empty_audio.export(output_path, format="wav")
        return
    
    # Load and concatenate included chunks
    reconstructed = AudioSegment.empty()
    for chunk in included_chunks:
        chunk_audio = AudioSegment.from_file(chunk.path)
        reconstructed += chunk_audio
    
    # Export reconstructed audio
    reconstructed.export(output_path, format="wav")
    print(f"[CHUNKING] Reconstructed audio from {len(included_chunks)}/{len(chunks)} chunks: {output_path}")

