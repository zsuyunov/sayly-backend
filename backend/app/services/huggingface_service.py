"""
Hugging Face Service Module for Speaker Embedding Extraction
Uses huggingface_hub InferenceClient for endpoint routing,
with fallbacks for different library versions.
"""
import os
import json
import time
from typing import List, Optional

import requests as http_requests   # alias to avoid shadowing

# ── Configuration ────────────────────────────────────────────────────────────
HF_MODEL_NAME = "speechbrain/spkrec-ecapa-voxceleb"
HF_API_TIMEOUT = 120   # seconds (cold starts can be slow)
HF_MAX_RETRIES = 3
HF_RETRY_DELAY = 5     # seconds between retries


def get_hf_api_key() -> str:
    """Return the HF API key from the environment.

    Raises:
        ValueError: if HF_API_KEY is missing.
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


# ── Internal helpers ─────────────────────────────────────────────────────────

def _call_via_inference_client(audio_data: bytes, api_key: str):
    """Try using huggingface_hub InferenceClient (handles endpoint routing).

    Returns parsed JSON result, or raises if the library call fails.
    """
    try:
        import huggingface_hub
        from huggingface_hub import InferenceClient
        print(f"[HF] huggingface_hub version: {huggingface_hub.__version__}")
    except ImportError:
        raise RuntimeError("huggingface_hub is not installed")

    client = InferenceClient(
        model=HF_MODEL_NAME,
        token=api_key,
        timeout=HF_API_TIMEOUT,
    )

    # The public API changed across versions.  Try in order of preference:
    #   1. client.post()          – public since ~0.22
    #   2. client._post()         – private, exists in almost every version
    raw: Optional[bytes] = None

    # --- attempt 1: public post() ---
    post_fn = getattr(client, "post", None)
    if callable(post_fn):
        print("[HF] Using InferenceClient.post()")
        raw = post_fn(data=audio_data, model=HF_MODEL_NAME)

    # --- attempt 2: private _post() ---
    if raw is None:
        _post_fn = getattr(client, "_post", None)
        if callable(_post_fn):
            print("[HF] Using InferenceClient._post()")
            raw = _post_fn(data=audio_data, model=HF_MODEL_NAME)

    if raw is None:
        raise RuntimeError(
            "InferenceClient has neither post() nor _post(). "
            f"Installed version: {huggingface_hub.__version__}"
        )

    return json.loads(raw)


def _call_via_requests(audio_data: bytes, api_key: str):
    """Fallback: plain HTTP requests to known HF inference URLs."""
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Accept": "application/json",
        "Content-Type": "application/octet-stream",
    }

    # Try several known URL patterns
    candidate_urls = [
        f"https://router.huggingface.co/hf-inference/models/{HF_MODEL_NAME}",
        f"https://router.huggingface.co/models/{HF_MODEL_NAME}",
        f"https://api-inference.huggingface.co/models/{HF_MODEL_NAME}",
    ]

    last_error = None
    for url in candidate_urls:
        try:
            print(f"[HF] Trying URL: {url}")
            resp = http_requests.post(
                url,
                headers=headers,
                data=audio_data,
                params={"wait_for_model": "true"},
                timeout=HF_API_TIMEOUT,
            )
            if resp.status_code == 200:
                return resp.json()
            preview = (resp.text or "")[:200]
            print(f"[HF] {url} → HTTP {resp.status_code}: {preview}")
            last_error = f"HTTP {resp.status_code}: {preview}"
        except Exception as e:
            last_error = str(e)
            print(f"[HF] {url} → error: {e}")

    raise RuntimeError(f"All HF inference URLs failed. Last error: {last_error}")


# ── Public API ───────────────────────────────────────────────────────────────

def extract_speaker_embedding(audio_path: str) -> List[float]:
    """Extract speaker embedding from an audio file via Hugging Face Inference.

    Args:
        audio_path: Path to a 16 kHz mono WAV file.

    Returns:
        List of floats – the speaker embedding vector.

    Raises:
        ValueError: if the API key is not configured.
        Exception: on unrecoverable API / network errors after retries.
    """
    api_key = get_hf_api_key()
    print(f"[HF] API key found (starts with: {api_key[:10]}...)")

    # Read audio bytes
    try:
        with open(audio_path, "rb") as f:
            audio_data = f.read()
        print(f"[HF] Read {len(audio_data)} bytes from {audio_path}")
    except Exception as e:
        raise Exception(f"Failed to read audio file: {e}")

    last_error: Optional[str] = None

    for attempt in range(1, HF_MAX_RETRIES + 1):
        print(f"[HF] Extracting embedding from {audio_path} (attempt {attempt}/{HF_MAX_RETRIES})")
        try:
            # --- Primary path: InferenceClient (auto-routes) ---
            result = _call_via_inference_client(audio_data, api_key)
            print(f"[HF] InferenceClient succeeded")

        except Exception as client_err:
            print(f"[HF] InferenceClient failed: {client_err}")
            try:
                # --- Fallback: raw requests ---
                result = _call_via_requests(audio_data, api_key)
                print(f"[HF] Raw requests fallback succeeded")
            except Exception as req_err:
                last_error = f"InferenceClient: {client_err} | requests: {req_err}"
                print(f"[HF] Both methods failed: {last_error}")
                if attempt < HF_MAX_RETRIES:
                    wait = HF_RETRY_DELAY * attempt
                    print(f"[HF] Retrying in {wait}s …")
                    time.sleep(wait)
                    continue
                raise Exception(
                    f"Hugging Face API error after {HF_MAX_RETRIES} attempts: {last_error}"
                )

        # ── Parse embedding from response ────────────────────────────────
        embedding = _extract_embedding_from_result(result)
        embedding_floats = [float(x) for x in embedding]
        print(f"[HF] Successfully extracted embedding (dimension: {len(embedding_floats)})")
        return embedding_floats

    raise Exception(f"Failed to extract embedding after {HF_MAX_RETRIES} attempts: {last_error}")


def _extract_embedding_from_result(result) -> list:
    """Pull the embedding list out of whatever shape HF returned."""
    embedding = None

    if isinstance(result, dict):
        embedding = (
            result.get("embedding")
            or result.get("embeddings")
            or result.get("features")
        )
        if isinstance(embedding, list) and embedding and isinstance(embedding[0], list):
            embedding = embedding[0]

    elif isinstance(result, list):
        embedding = result
        if embedding and isinstance(embedding[0], list):
            embedding = embedding[0]

    if embedding is None:
        print(f"[HF] Unexpected response shape: {str(result)[:300]}")
        raise Exception("Could not extract embedding from API response")

    if not isinstance(embedding, list):
        raise Exception(f"Embedding is not a list: {type(embedding)}")

    return embedding
