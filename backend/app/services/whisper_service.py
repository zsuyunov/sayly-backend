import os
import httpx
import asyncio
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Constants
HF_API_URL_TEMPLATE = "https://router.huggingface.co/hf-inference/models/{model_id}"
DEFAULT_MODEL_ID = "openai/whisper-small"
DEFAULT_TIMEOUT = 30

async def transcribe_audio(audio_path: str) -> Dict[str, Any]:
    """
    Transcribe audio using Hugging Face Inference API.
    
    Args:
        audio_path: Path to the audio file to transcribe
        
    Returns:
        Dict containing text and segments:
        {
          "text": "...",
          "segments": [
            {
              "start": float,
              "end": float,
              "text": str
            }
          ]
        }
    """
    hf_api_key = os.getenv("HF_API_KEY")
    if not hf_api_key:
        logger.error("HF_API_KEY environment variable not set")
        raise ValueError("HF_API_KEY environment variable not set")
        
    model_id = os.getenv("WHISPER_MODEL_ID", DEFAULT_MODEL_ID)
    api_url = HF_API_URL_TEMPLATE.format(model_id=model_id)
    
    try:
        timeout = int(os.getenv("WHISPER_TIMEOUT", DEFAULT_TIMEOUT))
    except (ValueError, TypeError):
        timeout = DEFAULT_TIMEOUT
    
    headers = {
        "Authorization": f"Bearer {hf_api_key}",
        "Content-Type": "audio/wav"
    }
    
    # Read audio file
    try:
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        # Check file size (approximate check based on prompt "Reject audio > 15 minutes")
        # 15 mins of 16kHz mono WAV is approx: 16000 * 2 bytes * 60 * 15 = 28.8 MB
        # Let's set a safe limit, say 30MB
        file_size = os.path.getsize(audio_path)
        if file_size > 30 * 1024 * 1024:
            logger.warning(f"Audio file too large: {file_size} bytes")
            # We continue but log warning, or should we reject? 
            # Prompt says "Add maximum file limit: Reject audio > 15 minutes"
            # I'll implement this check in the analysis pipeline or here.
            # Implementing here is safer.
            raise ValueError("Audio file too large (exceeds ~15 minutes limit)")

        with open(audio_path, "rb") as f:
            audio_data = f.read()
            
    except Exception as e:
        logger.error(f"Failed to read audio file {audio_path}: {e}")
        raise
        
    async with httpx.AsyncClient() as client:
        try:
            response_data = await _make_request_with_retry(client, api_url, headers, audio_data, timeout)
            return _parse_response(response_data)
        except httpx.HTTPStatusError as e:
            logger.error(f"HF API error: {e.response.status_code} - {e.response.text}")
            raise
        except Exception as e:
            logger.error(f"Transcription failed: {e}")
            raise

async def _make_request_with_retry(client: httpx.AsyncClient, url: str, headers: Dict, data: bytes, timeout: int) -> Dict:
    """Make request with retry logic for model loading (503)."""
    try:
        response = await client.post(url, headers=headers, content=data, timeout=timeout)
        
        if response.status_code == 503:
            # Model loading
            error_data = response.json()
            estimated_time = error_data.get("estimated_time", 2.0)
            wait_time = min(estimated_time, 2.0) # Prompt says "Retry once after 2s"
            
            logger.info(f"Model loading (503), retrying in {wait_time}s...")
            await asyncio.sleep(wait_time)
            
            # Retry once
            response = await client.post(url, headers=headers, content=data, timeout=timeout)
        
        if response.status_code == 429:
             logger.error("Rate limit exceeded (429)")
             raise httpx.HTTPStatusError("Rate limit exceeded", request=response.request, response=response)

        response.raise_for_status()
        return response.json()
        
    except httpx.TimeoutException:
        logger.error("Request timed out")
        raise

def _parse_response(data: Dict) -> Dict:
    """Parse HF response into expected format."""
    text = data.get("text", "")
    chunks = data.get("chunks", [])
    
    segments = []
    
    # If chunks are present, parse them
    if chunks:
        for chunk in chunks:
            timestamp = chunk.get("timestamp", [])
            # timestamp can be [start, end] or sometimes just start? 
            # Prompt assumes [start, end]
            start = 0.0
            end = 0.0
            
            if isinstance(timestamp, list) and len(timestamp) >= 2:
                start = float(timestamp[0])
                end = float(timestamp[1])
            elif isinstance(timestamp, list) and len(timestamp) == 1:
                start = float(timestamp[0])
                # End might be missing
            
            segments.append({
                "start": start,
                "end": end,
                "text": chunk.get("text", "").strip()
            })
    
    # If no chunks/segments, we just return text with empty segments
    # The prompt says "If timestamps not present: Fallback to full text only"
    
    return {
        "text": text,
        "segments": segments
    }

