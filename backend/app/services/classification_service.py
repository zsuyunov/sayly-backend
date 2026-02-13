import os
import requests
import logging
import time
from typing import Dict, Any, List

# Configure logging
logger = logging.getLogger(__name__)

# Constants
HF_CLASSIFICATION_URL = "https://router.huggingface.co/hf-inference/models/facebook/bart-large-mnli"
CLASSIFICATION_LABELS = [
    "gossip",
    "insult or unethical speech",
    "wasteful talk",
    "productive or meaningful speech"
]
MAX_RETRIES = 3
RETRY_DELAY = 2  # seconds

class HuggingFaceClassificationService:
    def __init__(self):
        # Use HF_API_KEY as per user instructions
        self.api_key = os.getenv("HF_API_KEY")
        if not self.api_key:
            logger.error("HF_API_KEY environment variable not set")
            raise ValueError("HF_API_KEY environment variable not set")
        
        self.headers = {
            "Authorization": f"Bearer {self.api_key}",
        }

    def classify_text(self, text: str) -> Dict[str, Any]:
        """
        Classify text using Hugging Face Zero-Shot Classification.
        
        Args:
            text: The text to classify
            
        Returns:
            Dict containing classification scores and labels
        """
        if not text or not text.strip():
            print("[CLASSIFICATION] Empty text provided, skipping classification")
            return {"labels": [], "scores": []}
        
        text_length = len(text)
        text_preview = text[:100] + "..." if len(text) > 100 else text
        print(f"[CLASSIFICATION] Starting classification (text length: {text_length} chars)")
        print(f"[CLASSIFICATION] Text preview: {text_preview}")
            
        payload = {
            "inputs": text,
            "parameters": {"candidate_labels": CLASSIFICATION_LABELS}
        }
        
        # Retry logic for transient errors (504, 503, 429)
        last_error = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                if attempt > 1:
                    wait_time = RETRY_DELAY * (attempt - 1)
                    print(f"[CLASSIFICATION] Retry attempt {attempt}/{MAX_RETRIES} after {wait_time}s delay")
                    time.sleep(wait_time)
                
                print(f"[CLASSIFICATION] Sending request to Hugging Face API (attempt {attempt}/{MAX_RETRIES})")
                logger.info(f"Sending text to Hugging Face for classification (attempt {attempt})...")
                response = requests.post(HF_CLASSIFICATION_URL, headers=self.headers, json=payload, timeout=30)
                
                if response.status_code != 200:
                    # Truncate HTML error messages for cleaner logs
                    error_text = response.text
                    if len(error_text) > 500 or '<!DOCTYPE html>' in error_text:
                        # Extract just the status code and message from HTML
                        if '504' in error_text:
                            error_text = "504 Gateway Timeout"
                        elif '503' in error_text:
                            error_text = "503 Service Unavailable"
                        elif '502' in error_text:
                            error_text = "502 Bad Gateway"
                        else:
                            error_text = error_text[:200] + "..."
                    
                    print(f"[CLASSIFICATION] API error {response.status_code}: {error_text}")
                    logger.error(f"HF API error {response.status_code}: {error_text}")
                    
                    # Retry on transient errors (5xx, 429)
                    if response.status_code in [502, 503, 504, 429] and attempt < MAX_RETRIES:
                        last_error = Exception(f"HF API error {response.status_code}: {error_text}")
                        continue
                    
                    raise Exception(f"HF API error {response.status_code}: {error_text}")
                
                response.raise_for_status()
                result = response.json()
                
                # Handle both list and dict responses from Hugging Face API
                # The API can return:
                # 1. A dict with "labels" and "scores" arrays
                # 2. A dict with "label" and "score" (singular)
                # 3. A list of dicts, each with "label" and "score"
                # 4. A list with one dict element
                if isinstance(result, list):
                    if len(result) == 0:
                        print(f"[CLASSIFICATION] API returned empty list")
                        raise Exception("Empty list response from Hugging Face API")
                    
                    # Check if list contains dicts with "label"/"score" (need to aggregate)
                    if isinstance(result[0], dict) and "label" in result[0] and "score" in result[0]:
                        # Aggregate all label/score pairs into arrays
                        labels = [item["label"] for item in result if "label" in item]
                        scores = [item["score"] for item in result if "score" in item]
                        result = {"labels": labels, "scores": scores}
                        print(f"[CLASSIFICATION] Aggregated {len(labels)} label/score pairs from list response")
                    else:
                        # List with one dict element (standard format)
                        result = result[0]
                        print(f"[CLASSIFICATION] API returned list, using first element")
                
                # Ensure result is a dict
                if not isinstance(result, dict):
                    print(f"[CLASSIFICATION] Unexpected response type: {type(result)}")
                    raise Exception(f"Unexpected response type from API: {type(result)}")
                
                # Normalize response format - handle both "label"/"score" (singular) and "labels"/"scores" (plural)
                if "label" in result and "score" in result and "labels" not in result:
                    # Single label/score format - convert to plural arrays
                    result = {
                        "labels": [result["label"]],
                        "scores": [result["score"]]
                    }
                    print(f"[CLASSIFICATION] Converted singular format to plural arrays")
                elif "labels" not in result or "scores" not in result:
                    print(f"[CLASSIFICATION] Unexpected response structure: {list(result.keys())}")
                    raise Exception(f"Unexpected response structure: missing 'labels'/'scores' or 'label'/'score'")
                
                # Log successful classification results
                labels = result.get("labels", [])
                scores = result.get("scores", [])
                if labels and scores:
                    top_label = labels[0]
                    top_score = scores[0]
                    print(f"[CLASSIFICATION] Classification successful!")
                    print(f"[CLASSIFICATION] Top category: {top_label} (confidence: {top_score:.3f})")
                    print(f"[CLASSIFICATION] All scores: {dict(zip(labels, [f'{s:.3f}' for s in scores]))}")
                else:
                    print(f"[CLASSIFICATION] Classification successful but no labels/scores in response")
                    print(f"[CLASSIFICATION] Response structure: {list(result.keys()) if isinstance(result, dict) else 'not a dict'}")
                
                return result
                
            except requests.exceptions.Timeout as e:
                error_msg = "Request timeout after 30s"
                print(f"[CLASSIFICATION] {error_msg}")
                if attempt < MAX_RETRIES:
                    last_error = Exception(error_msg)
                    continue
                raise Exception(error_msg)
            except requests.exceptions.RequestException as e:
                error_msg = str(e)
                if hasattr(e, 'response') and e.response:
                    error_text = e.response.text
                    # Truncate HTML errors
                    if len(error_text) > 200:
                        error_text = error_text[:200] + "..."
                    status_code = e.response.status_code
                    print(f"[CLASSIFICATION] Request failed: {status_code} - {error_text}")
                    logger.error(f"Failed to classify text: {error_msg}")
                    
                    # Retry on transient errors
                    if status_code in [502, 503, 504, 429] and attempt < MAX_RETRIES:
                        last_error = Exception(f"Classification failed: {error_msg}")
                        continue
                else:
                    print(f"[CLASSIFICATION] Request failed: {error_msg}")
                    logger.error(f"Failed to classify text: {error_msg}")
                raise Exception(f"Classification failed: {error_msg}")
            except Exception as e:
                error_msg = str(e)
                # If it's a retriable error and we have retries left, continue
                if 'HF API error' in error_msg and any(code in error_msg for code in ['502', '503', '504', '429']) and attempt < MAX_RETRIES:
                    last_error = e
                    continue
                    
                print(f"[CLASSIFICATION] Classification error: {error_msg[:200]}")
                logger.error(f"Failed to classify text: {error_msg[:200]}")
                raise
        
        # If we exhausted all retries, raise the last error
        if last_error:
            print(f"[CLASSIFICATION] All {MAX_RETRIES} retry attempts failed")
            raise last_error

# Singleton instance
_service = None

def classify_speech_text(text: str) -> Dict[str, Any]:
    """Wrapper function for classification."""
    global _service
    if not _service:
        print("[CLASSIFICATION] Initializing Hugging Face Classification Service")
        _service = HuggingFaceClassificationService()
    return _service.classify_text(text)

