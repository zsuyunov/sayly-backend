import os
import httpx
import asyncio
import logging
from typing import Dict, Any, List, Optional

# Configure logging
logger = logging.getLogger(__name__)

# Constants
ASSEMBLYAI_UPLOAD_URL = "https://api.assemblyai.com/v2/upload"
ASSEMBLYAI_TRANSCRIPT_URL = "https://api.assemblyai.com/v2/transcript"
DEFAULT_TIMEOUT = 300  # 5 minutes timeout for transcription

class AssemblyAIService:
    def __init__(self):
        self.api_key = os.getenv("ASSEMBLYAI_API_KEY")
        if not self.api_key:
            logger.error("ASSEMBLYAI_API_KEY environment variable not set")
            raise ValueError("ASSEMBLYAI_API_KEY environment variable not set")
        
        self.headers = {
            "authorization": self.api_key
        }

    async def transcribe_audio(self, audio_path: str) -> Dict[str, Any]:
        """
        Transcribe audio using AssemblyAI API (async).
        
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
        async with httpx.AsyncClient(timeout=DEFAULT_TIMEOUT) as client:
            # 1. Upload audio file
            upload_url = await self._upload_file(client, audio_path)
            
            # 2. Request transcription
            transcript_id = await self._request_transcription(client, upload_url)
            
            # 3. Poll for completion
            result = await self._poll_for_completion(client, transcript_id)
            
            # 4. Parse result
            return self._parse_response(result)

    async def _upload_file(self, client: httpx.AsyncClient, audio_path: str) -> str:
        """Upload audio file to AssemblyAI."""
        if not os.path.exists(audio_path):
            raise FileNotFoundError(f"Audio file not found: {audio_path}")
            
        logger.info(f"Uploading audio file: {audio_path}")
        
        try:
            with open(audio_path, 'rb') as f:
                # Read entire file into memory for simple async upload
                # (httpx supports async generators but simple bytes is safer for small files < 50MB)
                file_content = f.read()
                
            response = await client.post(
                ASSEMBLYAI_UPLOAD_URL,
                headers=self.headers,
                content=file_content
            )
            response.raise_for_status()
            return response.json()['upload_url']
        except Exception as e:
            logger.error(f"Failed to upload file to AssemblyAI: {e}")
            raise

    async def _request_transcription(self, client: httpx.AsyncClient, audio_url: str) -> str:
        """Request transcription for uploaded audio."""
        logger.info("Requesting transcription...")
        
        json_data = {
            "audio_url": audio_url,
            "speaker_labels": False,  # We handle speaker ID via MFCC locally
            "punctuation": True,
            "format_text": True
        }
        
        try:
            response = await client.post(
                ASSEMBLYAI_TRANSCRIPT_URL,
                json=json_data,
                headers=self.headers
            )
            response.raise_for_status()
            return response.json()['id']
        except Exception as e:
            logger.error(f"Failed to request transcription: {e}")
            raise

    async def _poll_for_completion(self, client: httpx.AsyncClient, transcript_id: str) -> Dict[str, Any]:
        """Poll AssemblyAI API until transcription is complete."""
        polling_endpoint = f"{ASSEMBLYAI_TRANSCRIPT_URL}/{transcript_id}"
        
        start_time = asyncio.get_event_loop().time()
        while True:
            if asyncio.get_event_loop().time() - start_time > DEFAULT_TIMEOUT:
                raise TimeoutError("Transcription timed out")
                
            try:
                response = await client.get(polling_endpoint, headers=self.headers)
                response.raise_for_status()
                result = response.json()
                
                status = result['status']
                
                if status == 'completed':
                    logger.info("Transcription completed")
                    return result
                elif status == 'error':
                    error_msg = result.get('error', 'Unknown error')
                    raise Exception(f"Transcription failed: {error_msg}")
                
                # Wait before polling again
                await asyncio.sleep(3)
                
            except Exception as e:
                logger.error(f"Error polling transcription status: {e}")
                raise

    def _parse_response(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """Parse AssemblyAI response into standard format."""
        text = data.get("text", "")
        words = data.get("words", [])
        
        segments = []
        for word in words:
            segments.append({
                "start": word["start"] / 1000.0,  # AssemblyAI uses ms
                "end": word["end"] / 1000.0,
                "text": word["text"]
            })
            
        return {
            "text": text,
            "segments": segments
        }

# Singleton instance
_service = None

async def transcribe_audio(audio_path: str) -> Dict[str, Any]:
    """Wrapper function to match previous interface."""
    global _service
    if not _service:
        _service = AssemblyAIService()
    return await _service.transcribe_audio(audio_path)

