"""
Hugging Face Service Module for Speaker Embedding Extraction
Uses Hugging Face Inference API to extract speaker embeddings from audio files.
"""
import os
import time
from typing import List, Optional
import requests

# Hugging Face API configuration
# NOTE (2026): Hugging Face deprecated direct `api-inference.huggingface.co`.
# Use the Inference Router instead.
HF_ROUTER_BASE_URL = "https://router.huggingface.co/hf-inference/models"
HF_LEGACY_BASE_URL = "https://api-inference.huggingface.co/models"  # fallback for older setups
HF_MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"
HF_API_TIMEOUT = 60  # seconds (HF cold starts can be slow)
HF_RATE_LIMIT_RETRY_DELAY = 5  # seconds
HF_MAX_RETRIES = 3

# Lazy initialization - only check API key when functions are called
_hf_api_key: Optional[str] = None


def get_hf_api_key() -> str:
    """Get or initialize Hugging Face API key. Checks API key only when needed.
    
    Returns:
        Hugging Face API key string
        
    Raises:
        ValueError if HF_API_KEY environment variable is not set or invalid
    """
    global _hf_api_key
    if _hf_api_key is None:
        _hf_api_key = os.getenv("HF_API_KEY")
        if not _hf_api_key:
            raise ValueError(
                "HF_API_KEY environment variable is not set. "
                "Please set it in your environment variables or deployment configuration."
            )
        # Validate API key format (should start with "hf_")
        if not _hf_api_key.startswith("hf_"):
            print(f"[HF] Warning: API key doesn't start with 'hf_'. Key preview: {_hf_api_key[:10]}...")
            # Don't fail here - some keys might have different formats, but log a warning
    return _hf_api_key


def extract_speaker_embedding(audio_path: str) -> List[float]:
    """Extract speaker embedding from audio file using Hugging Face ECAPA model.
    
    This function:
    1. Reads the audio file
    2. Sends it to Hugging Face Inference API
    3. Extracts the speaker embedding vector
    4. Handles timeouts, rate limits, and errors gracefully
    
    Args:
        audio_path: Path to the audio file (should be 16kHz mono WAV)
        
    Returns:
        List of floats representing the speaker embedding vector
        
    Raises:
        Exception: If embedding extraction fails after retries
    """
    # Check API key first
    try:
        api_key = get_hf_api_key()
        # Log first 10 chars for debugging (don't log full key for security)
        if api_key:
            print(f"[HF] API key found (starts with: {api_key[:10]}...)")
        else:
            print("[HF] API key is None or empty")
    except ValueError as e:
        # API key not set - re-raise with clear message
        print(f"[HF] API key error: {e}")
        raise ValueError(str(e))
    
    # Prefer router endpoint; keep legacy fallback for safety
    api_urls = [
        f"{HF_ROUTER_BASE_URL}/{HF_MODEL_NAME}",
        f"{HF_LEGACY_BASE_URL}/{HF_MODEL_NAME}",
    ]
    api_url = api_urls[0]
    print(f"[HF] Using API URL: {api_url}")
    
    # Read audio file
    try:
        with open(audio_path, 'rb') as audio_file:
            audio_data = audio_file.read()
    except Exception as e:
        print(f"[HF] Error reading audio file {audio_path}: {e}")
        raise Exception(f"Failed to read audio file: {str(e)}")
    
    # Prepare request headers
    # Explicit headers reduce the chance of upstream returning HTML (proxies / auth pages).
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/octet-stream",
        "User-Agent": "sayly-backend/1.0 (+https://sayly-backend.onrender.com)",
    }
    
    # Retry logic for rate limits and temporary failures
    last_error = None
    for attempt in range(HF_MAX_RETRIES):
        try:
            print(f"[HF] Extracting embedding from {audio_path} (attempt {attempt + 1}/{HF_MAX_RETRIES})")
            
            # Make request to Hugging Face API
            # wait_for_model=true avoids 503 loops during cold start
            last_router_deprecation_error = None
            response = None
            for candidate_url in api_urls:
                response = requests.post(
                    candidate_url,
                    headers=headers,
                    data=audio_data,
                    params={"wait_for_model": "true"},
                    timeout=HF_API_TIMEOUT,
                )
                # If HF tells us this host is no longer supported, switch to router immediately.
                if (
                    response is not None
                    and response.status_code in (400, 404, 410, 500)
                    and "no longer supported" in (response.text or "").lower()
                    and "router.huggingface.co" in (response.text or "").lower()
                ):
                    last_router_deprecation_error = (response.text or "")[:300]
                    print(f"[HF] Legacy host deprecated. Switching to router. preview={last_router_deprecation_error}")
                    continue
                # Otherwise keep this response (success or a real error)
                api_url = candidate_url
                break
            
            # Handle model loading / transient throttling
            if response.status_code in (429, 503):
                # Model is loading, wait and retry
                wait_time = HF_RATE_LIMIT_RETRY_DELAY * (attempt + 1)
                print(f"[HF] Transient HF response {response.status_code}, waiting {wait_time} seconds before retry...")
                time.sleep(wait_time)
                continue
            
            # Handle other HTTP errors
            if response.status_code != 200:
                error_msg = f"HTTP {response.status_code}"
                
                # Check if response is HTML (error page)
                content_type = response.headers.get('content-type', '').lower()
                is_html = '<!doctype html>' in response.text.lower() or '<html' in response.text.lower() or 'text/html' in content_type
                
                if is_html:
                    # HTML response usually means auth/billing/gating/proxy issues.
                    error_msg = (
                        f"HTTP {response.status_code}: Received HTML error page from Hugging Face. "
                        f"This usually means one of: invalid token, billing/inference access not enabled, "
                        f"model access blocked (gated/terms not accepted), or an upstream/proxy block."
                    )
                    print(f"[HF] HTML error response detected. content-type={content_type} preview={response.text[:300]}")
                else:
                    # Try to parse JSON error
                    try:
                        error_data = response.json()
                        if "error" in error_data:
                            error_msg = error_data["error"]
                        elif "message" in error_data:
                            error_msg = error_data["message"]
                    except:
                        # Not JSON, use text
                        error_msg = response.text[:200] if response.text else error_msg
                
                print(f"[HF] API error: {error_msg}")
                
                # If it's a client error (4xx), don't retry
                if 400 <= response.status_code < 500:
                    raise Exception(f"Hugging Face API error: {error_msg}")
                
                # For server errors (5xx), retry
                if attempt < HF_MAX_RETRIES - 1:
                    wait_time = HF_RATE_LIMIT_RETRY_DELAY * (attempt + 1)
                    print(f"[HF] Server error, retrying in {wait_time} seconds...")
                    time.sleep(wait_time)
                    continue
                else:
                    raise Exception(f"Hugging Face API error after {HF_MAX_RETRIES} attempts: {error_msg}")
            
            # Check if response is HTML before parsing
            content_type = response.headers.get('content-type', '').lower()
            is_html = '<!doctype html>' in response.text.lower() or '<html' in response.text.lower() or 'text/html' in content_type
            
            if is_html:
                error_msg = "Received HTML response instead of JSON. This usually means the API key is invalid or the endpoint is incorrect."
                print(f"[HF] HTML response detected. Response preview: {response.text[:500]}")
                raise Exception(f"Hugging Face API error: {error_msg}")
            
            # Parse response
            try:
                result = response.json()
            except Exception as e:
                print(f"[HF] Error parsing JSON response: {e}")
                print(f"[HF] Response content-type: {content_type}")
                print(f"[HF] Response text (first 500 chars): {response.text[:500]}")
                raise Exception(f"Failed to parse API response as JSON: {str(e)}")
            
            # Extract embedding from response
            # The response format may vary, try common field names
            embedding = None
            if isinstance(result, dict):
                # Try different possible field names
                embedding = result.get("embedding") or result.get("embeddings") or result.get("features")
                
                # If it's a list of lists, take the first one
                if isinstance(embedding, list) and len(embedding) > 0 and isinstance(embedding[0], list):
                    embedding = embedding[0]
            elif isinstance(result, list):
                # Response might be directly a list
                embedding = result
                # If it's a list of lists, take the first one
                if len(embedding) > 0 and isinstance(embedding[0], list):
                    embedding = embedding[0]
            
            if embedding is None:
                print(f"[HF] Unexpected response format: {result}")
                raise Exception("Could not extract embedding from API response")
            
            # Ensure embedding is a list of floats
            if not isinstance(embedding, list):
                raise Exception(f"Embedding is not a list: {type(embedding)}")
            
            # Convert to list of floats
            embedding_floats = [float(x) for x in embedding]
            
            print(f"[HF] Successfully extracted embedding (dimension: {len(embedding_floats)})")
            return embedding_floats
            
        except requests.exceptions.Timeout:
            last_error = f"Request timeout after {HF_API_TIMEOUT} seconds"
            print(f"[HF] {last_error}")
            if attempt < HF_MAX_RETRIES - 1:
                wait_time = HF_RATE_LIMIT_RETRY_DELAY * (attempt + 1)
                print(f"[HF] Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                raise Exception(f"{last_error} (after {HF_MAX_RETRIES} attempts)")
                
        except requests.exceptions.RequestException as e:
            last_error = f"Network error: {str(e)}"
            print(f"[HF] {last_error}")
            if attempt < HF_MAX_RETRIES - 1:
                wait_time = HF_RATE_LIMIT_RETRY_DELAY * (attempt + 1)
                print(f"[HF] Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            else:
                raise Exception(f"{last_error} (after {HF_MAX_RETRIES} attempts)")
                
        except Exception as e:
            # For other exceptions, don't retry
            print(f"[HF] Error extracting embedding: {e}")
            raise
    
    # If we get here, all retries failed
    raise Exception(f"Failed to extract embedding after {HF_MAX_RETRIES} attempts: {last_error}")

