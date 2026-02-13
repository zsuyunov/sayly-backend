import os
import json
from typing import Dict, Any, Optional
from app.services.classification_service import classify_speech_text

def transcribe_audio(audio_path: str) -> str:
    """Legacy function, now unused as we use AssemblyAI directly.
    Kept for interface compatibility if needed, but raises error.
    """
    raise NotImplementedError("Use AssemblyAI service for transcription")


def analyze_speech(text: str) -> Dict[str, Any]:
    """Analyze speech text for gossip and unethical speech patterns using Hugging Face Zero-Shot Classification.
    
    Categories:
    - gossip
    - insult or unethical speech
    - wasteful talk
    - productive or meaningful speech
    
    Args:
        text: The transcribed speech text to analyze
        
    Returns:
        Dict containing analysis metrics
    """
    if not text or not text.strip():
        return {
            "flaggedCount": 0,
            "positiveCount": 0,
            "score": 50,  # Neutral score for empty text
            "flaggedExamples": [],
            "positiveExamples": [],
            "classification": {}
        }
    
    try:
        # Use Hugging Face Zero-Shot Classification
        print(f"[AI] Analyzing speech text using Hugging Face classification (text length: {len(text)} chars)")
        result = classify_speech_text(text)
        
        labels = result.get("labels", [])
        scores = result.get("scores", [])
        
        # Extract clean category names (remove descriptions after colon)
        clean_labels = []
        for label in labels:
            # Extract category name before colon
            category = label.split(":")[0].strip()
            # Map to our standard category names
            if "gossip" in category.lower():
                clean_labels.append("gossip")
            elif "insult" in category.lower() or "unethical" in category.lower():
                clean_labels.append("insult or unethical speech")
            elif "wasteful" in category.lower() or "waste" in category.lower():
                clean_labels.append("wasteful talk")
            elif "productive" in category.lower() or "meaningful" in category.lower():
                clean_labels.append("productive or meaningful speech")
            else:
                clean_labels.append(category)
        
        # Create a mapping of clean label to score
        classification = dict(zip(clean_labels, scores))
        
        # Determine top category (using clean label)
        top_category = clean_labels[0] if clean_labels else "unknown"
        top_score = scores[0] if scores else 0.0
        
        # Determine if speech is flagged (negative)
        flagged_categories = ["gossip", "insult or unethical speech"]
        positive_categories = ["productive or meaningful speech"]
        # neutral_categories = ["wasteful talk"]
        
        flagged_count = 0
        positive_count = 0
        
        if top_category in flagged_categories and top_score > 0.4:
            flagged_count = 1
        elif top_category in positive_categories and top_score > 0.4:
            positive_count = 1
            
        # Calculate overall score (0-100)
        # Higher score is better/more positive
        score = 50 # Start neutral
        
        if top_category == "productive or meaningful speech":
            score = 50 + int(top_score * 50)
        elif top_category == "wasteful talk":
            score = 50 - int(top_score * 20)
        elif top_category == "gossip":
            score = 50 - int(top_score * 40)
        elif top_category == "insult or unethical speech":
            score = 50 - int(top_score * 50)
            
        # Ensure score is in valid range
        score = max(0, min(100, score))
        
        # Construct reasoning string
        reasoning = f"Primary classification: {top_category} ({top_score:.2f})."
        
        return {
            "flaggedCount": flagged_count,
            "positiveCount": positive_count,
            "score": score,
            "flaggedExamples": [text[:100] + "..."] if flagged_count > 0 else [],
            "positiveExamples": [text[:100] + "..."] if positive_count > 0 else [],
            "reasoning": reasoning,
            "classification": classification
        }
        
    except Exception as e:
        print(f"[AI] Error analyzing speech with Hugging Face: {e}")
        # Fallback to neutral if classification fails
        return {
            "flaggedCount": 0,
            "positiveCount": 0,
            "score": 50,
            "flaggedExamples": [],
            "positiveExamples": [],
            "reasoning": f"Analysis failed: {str(e)}",
            "classification": {}
        }


def generate_session_summary(analysis: Dict[str, Any], transcript: str) -> str:
    """Generate a template-based summary using classification results.
    
    Args:
        analysis: The analysis results from analyze_speech
        transcript: The full transcript text (unused in template but kept for signature)
        
    Returns:
        A neutral, non-judgmental summary string
    """
    flagged = analysis.get("flaggedCount", 0)
    positive = analysis.get("positiveCount", 0)
    score = analysis.get("score", 50)
    reasoning = analysis.get("reasoning", "")
    
    # Extract top category from reasoning or classification
    classification = analysis.get("classification", {})
    if classification:
        top_category = max(classification, key=classification.get)
    else:
        # Fallback parsing from reasoning string if dict not available
        if "Primary classification:" in reasoning:
            try:
                top_category = reasoning.split(":")[1].split("(")[0].strip()
            except:
                top_category = "unknown"
        else:
            top_category = "unknown"

    # Template-based summary generation
    if top_category == "productive or meaningful speech":
        return "Your speech in this session was primarily productive and meaningful. Keep up the constructive communication!"
    elif top_category == "wasteful talk":
        return "This session involved some casual or unstructured conversation. Reflect on whether this aligns with your goals."
    elif top_category == "gossip":
        return "Patterns of gossip were detected in this session. Consider focusing on more direct and constructive topics."
    elif top_category == "insult or unethical speech":
        return "Some speech in this session was flagged as potentially harmful. Please reflect on using more positive language."
    else:
        # Fallback for mixed or low-confidence results
        if score >= 60:
            return "Your speech patterns were generally positive. Continue to be mindful of your communication style."
        elif score <= 40:
            return "Some areas for improvement were noted in your speech. Review the session details for more insight."
        else:
            return "This session has been recorded. Review your speech patterns to identify areas for improvement."
