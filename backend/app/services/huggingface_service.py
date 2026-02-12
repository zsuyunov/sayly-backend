"""
Hugging Face Service Module for Speaker Embedding Extraction
Uses huggingface_hub InferenceClient – handles endpoint routing automatically,
so we never need to hard-code api-inference / router URLs again.
"""
import os
import json
import time
from typing import List, Optional

from huggingface_hub import InferenceClient
from huggingface_hub.utils import HfHubHTTPError

# Model configuration
HF_MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"
HF_API_TIMEOUT = 120  # seconds (cold starts can be slow)
HF_MAX_RETRIES = 3
HF_RETRY_DELAY = 5  # seconds between retries

# Lazy-initialised singleton client
_client: Optional[InferenceClient] = None


def get_hf_api_key() -> str:
    """Return the HF API key from the environment.

    Raises:
        ValueError: if HF_API_KEY is missing or looks wrong.
    """
    api_key = os.getenv("HF_API_KEY")
    if not api_key:
        raise ValueError(
            "HF_API_KEY environment variable is not set. "
            "Please set it in your environment variables or deployment configuration."
        )
    if not api_key.startswith("hf_"):
        print(f"[HF] Warning: API key doesn't start with 'hf_'. Preview: {api_key[:10]}...")
    return api_key


def _get_client() -> InferenceClient:
    """Return (or create) a cached InferenceClient."""
    global _client
    if _client is None:
        api_key = get_hf_api_key()
        print(f"[HF] Creating InferenceClient (key starts with: {api_key[:10]}...)")
        _client = InferenceClient(
            model=HF_MODEL_NAME,
            token=api_key,
            timeout=HF_API_TIMEOUT,
        )
    return _client


def extract_speaker_embedding(audio_path: str) -> List[float]:
    """Extract speaker embedding from an audio file via Hugging Face Inference.

    Uses ``huggingface_hub.InferenceClient`` which automatically resolves the
    correct Inference API endpoint (router / serverless / dedicated), so we
    don't have to guess URLs.

    Args:
        audio_path: Path to a 16 kHz mono WAV file.

    Returns:
        List of floats – the speaker embedding vector.

    Raises:
        ValueError: if the API key is not configured.
        Exception: on unrecoverable API / network errors after retries.
    """
    # Validate API key early (raises ValueError if missing)
    try:
        client = _get_client()
    except ValueError:
        raise  # re-raise so caller can distinguish "not configured" vs "API error"

    # Read audio bytes
    try:
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        print(f"[HF] Read {len(audio_data)} bytes from {audio_path}")
    except Exception as e:
        raise Exception(f"Failed to read audio file: {e}")

    last_error: Optional[str] = None

    for attempt in range(1, HF_MAX_RETRIES + 1):
        try:
            print(f"[HF] Extracting embedding from {audio_path} (attempt {attempt}/{HF_MAX_RETRIES})")

            # InferenceClient.post() sends raw bytes to the model endpoint.
            # The library figures out the correct URL (router / serverless).
            raw_response = client.post(
                data=audio_data,
                model=HF_MODEL_NAME,
            )

            # raw_response is bytes – decode to JSON
            result = json.loads(raw_response)
            print(f"[HF] Got response type: {type(result).__name__}")

            # ---- Extract embedding from various response shapes ----
            embedding = None

            if isinstance(result, dict):
                embedding = (
                    result.get("embedding")
                    or result.get("embeddings")
                    or result.get("features")
                )
                # list-of-lists → take first
                if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
                    embedding = embedding[0]

            elif isinstance(result, list):
                embedding = result
                if embedding and isinstance(embedding[0], list):
                    embedding = embedding[0]

            if embedding is None:
                print(f"[HF] Unexpected response shape: {str(result)[:300]}")
                raise Exception("Could not extract embedding from API response")

            embedding_floats = [float(x) for x in embedding]
            print(f"[HF] Successfully extracted embedding (dimension: {len(embedding_floats)})")
            return embedding_floats

        except HfHubHTTPError as e:
            status = getattr(e.response, "status_code", None) if hasattr(e, "response") else None
            last_error = f"HfHub HTTP error (status={status}): {e}"
            print(f"[HF] {last_error}")

            # 503 → model loading; 429 → rate limit → retry
            if status in (429, 503) and attempt < HF_MAX_RETRIES:
                wait = HF_RETRY_DELAY * attempt
                print(f"[HF] Retrying in {wait}s …")
                time.sleep(wait)
                continue

            # 4xx client errors → don't retry
            if status and 400 <= status < 500:
                raise Exception(f"Hugging Face API error: {e}")

            # 5xx or unknown → retry
            if attempt < HF_MAX_RETRIES:
                wait = HF_RETRY_DELAY * attempt
                print(f"[HF] Server error, retrying in {wait}s …")
                time.sleep(wait)
                continue

            raise Exception(f"Hugging Face API error after {HF_MAX_RETRIES} attempts: {e}")

        except json.JSONDecodeError as e:
            last_error = f"JSON decode error: {e}"
            print(f"[HF] {last_error}")
            raise Exception(f"Failed to parse Hugging Face response as JSON: {e}")

        except ValueError:
            raise  # API key not set – bubble up

        except Exception as e:
            last_error = str(e)
            print(f"[HF] Error extracting embedding: {e}")

            if attempt < HF_MAX_RETRIES:
                wait = HF_RETRY_DELAY * attempt
                print(f"[HF] Retrying in {wait}s …")
                time.sleep(wait)
                continue

            raise

    raise Exception(f"Failed to extract embedding after {HF_MAX_RETRIES} attempts: {last_error}")
