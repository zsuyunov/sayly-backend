import os
import requests
import logging
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
        
        try:
            print(f"[CLASSIFICATION] Sending request to Hugging Face API: {HF_CLASSIFICATION_URL}")
            logger.info("Sending text to Hugging Face for classification...")
            response = requests.post(HF_CLASSIFICATION_URL, headers=self.headers, json=payload)
            
            if response.status_code != 200:
                error_text = response.text
                print(f"[CLASSIFICATION] API error {response.status_code}: {error_text}")
                logger.error(f"HF API error {response.status_code}: {error_text}")
                raise Exception(f"HF API error {response.status_code}: {error_text}")
            
            response.raise_for_status()
            result = response.json()
            
            # Handle both list and dict responses from Hugging Face API
            # Sometimes the API returns a list with one element (the result dict)
            if isinstance(result, list):
                if len(result) > 0:
                    result = result[0]
                    print(f"[CLASSIFICATION] API returned list, using first element")
                else:
                    print(f"[CLASSIFICATION] API returned empty list")
                    raise Exception("Empty list response from Hugging Face API")
            
            # Ensure result is a dict
            if not isinstance(result, dict):
                print(f"[CLASSIFICATION] Unexpected response type: {type(result)}")
                raise Exception(f"Unexpected response type from API: {type(result)}")
            
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
        except requests.exceptions.RequestException as e:
            error_msg = str(e)
            if hasattr(e, 'response') and e.response:
                error_text = e.response.text
                status_code = e.response.status_code
                print(f"[CLASSIFICATION] Request failed: {status_code} - {error_text}")
                logger.error(f"Failed to classify text: {error_msg} - Response: {error_text}")
            else:
                print(f"[CLASSIFICATION] Request failed: {error_msg}")
                logger.error(f"Failed to classify text: {error_msg}")
            raise Exception(f"Classification failed: {error_msg}")
        except Exception as e:
            error_msg = str(e)
            print(f"[CLASSIFICATION] Classification error: {error_msg}")
            logger.error(f"Failed to classify text: {error_msg}")
            if hasattr(e, 'response') and e.response:
                print(f"[CLASSIFICATION] Response details: {e.response.text}")
                logger.error(f"Response: {e.response.text}")
            raise

# Singleton instance
_service = None

def classify_speech_text(text: str) -> Dict[str, Any]:
    """Wrapper function for classification."""
    global _service
    if not _service:
        print("[CLASSIFICATION] Initializing Hugging Face Classification Service")
        _service = HuggingFaceClassificationService()
    return _service.classify_text(text)

