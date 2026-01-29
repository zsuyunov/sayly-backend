import os
import json
from typing import Dict, Any, Optional
from openai import OpenAI

# Initialize OpenAI client
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY environment variable is not set")

client = OpenAI(api_key=OPENAI_API_KEY)


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file to text using OpenAI Whisper.
    
    Args:
        audio_path: Path to the audio file
        
    Returns:
        Transcribed text as a string
        
    Raises:
        Exception if transcription fails
    """
    try:
        with open(audio_path, "rb") as audio_file:
            transcript = client.audio.transcriptions.create(
                model="whisper-1",
                file=audio_file,
                language="en"  # Can be made configurable
            )
        return transcript.text
    except Exception as e:
        print(f"[AI] Error transcribing audio: {e}")
        raise Exception(f"Failed to transcribe audio: {str(e)}")


def analyze_speech(text: str) -> Dict[str, Any]:
    """Analyze speech text for gossip and unethical speech patterns.
    
    Uses GPT to detect:
    - Third-person focus (talking about others)
    - Negative talk
    - Backbiting patterns
    - Positive vs harmful speech
    
    Args:
        text: The transcribed speech text to analyze
        
    Returns:
        Dict containing:
            - flaggedCount: Number of flagged interactions
            - positiveCount: Number of positive interactions
            - score: Gossip score (0-100, lower is better)
            - flaggedExamples: List of flagged phrases/examples
            - positiveExamples: List of positive phrases/examples
    """
    if not text or not text.strip():
        return {
            "flaggedCount": 0,
            "positiveCount": 0,
            "score": 50,  # Neutral score for empty text
            "flaggedExamples": [],
            "positiveExamples": [],
        }
    
    prompt = f"""Analyze the following speech transcript for gossip, backbiting, and unethical speech patterns.

Gossip indicators include:
- Talking about others in their absence (third-person focus)
- Negative comments about others
- Spreading rumors or unverified information
- Backbiting or speaking ill of others
- Excessive criticism or judgment of others

Positive speech indicators include:
- Constructive conversations
- Speaking directly to people (first/second person)
- Problem-solving discussions
- Encouraging or supportive language
- Neutral or factual statements

Transcript:
{text}

Analyze this transcript and return a JSON object with the following structure:
{{
    "flaggedCount": <number of flagged interactions/phrases>,
    "positiveCount": <number of positive interactions/phrases>,
    "flaggedExamples": [<list of up to 3 example phrases that were flagged>],
    "positiveExamples": [<list of up to 3 example phrases that were positive>],
    "reasoning": "<brief explanation of the analysis>"
}}

Be strict but fair. Only flag clear instances of gossip or backbiting. Count each distinct instance separately.
Return ONLY valid JSON, no additional text."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",  # Using mini for cost efficiency, can upgrade to gpt-4 if needed
            messages=[
                {
                    "role": "system",
                    "content": "You are an ethical speech analysis assistant. Analyze speech for gossip and backbiting patterns. Return only valid JSON."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.3,  # Lower temperature for more consistent analysis
            max_tokens=1000,
        )
        
        content = response.choices[0].message.content.strip()
        
        # Parse JSON response
        # Sometimes GPT wraps JSON in markdown code blocks
        if content.startswith("```"):
            # Extract JSON from code block
            lines = content.split("\n")
            json_start = None
            json_end = None
            for i, line in enumerate(lines):
                if line.strip().startswith("{"):
                    json_start = i
                    break
            if json_start is not None:
                json_lines = lines[json_start:]
                # Find closing brace
                for i in range(len(json_lines) - 1, -1, -1):
                    if json_lines[i].strip().endswith("}"):
                        json_end = i + 1
                        break
                if json_end:
                    content = "\n".join(json_lines[:json_end])
        
        analysis = json.loads(content)
        
        # Calculate score (0-100, lower is worse/more gossip)
        flagged = analysis.get("flaggedCount", 0)
        positive = analysis.get("positiveCount", 0)
        total = flagged + positive
        
        if total == 0:
            score = 50  # Neutral
        else:
            # Score based on ratio: more positive = higher score
            positive_ratio = positive / total
            score = int(positive_ratio * 100)
        
        # Ensure score is in valid range
        score = max(0, min(100, score))
        
        return {
            "flaggedCount": flagged,
            "positiveCount": positive,
            "score": score,
            "flaggedExamples": analysis.get("flaggedExamples", [])[:3],
            "positiveExamples": analysis.get("positiveExamples", [])[:3],
            "reasoning": analysis.get("reasoning", ""),
        }
        
    except json.JSONDecodeError as e:
        print(f"[AI] Error parsing JSON response: {e}")
        print(f"[AI] Response content: {content}")
        # Fallback: basic analysis
        return {
            "flaggedCount": 0,
            "positiveCount": 0,
            "score": 50,
            "flaggedExamples": [],
            "positiveExamples": [],
            "reasoning": "Analysis failed - could not parse response",
        }
    except Exception as e:
        print(f"[AI] Error analyzing speech: {e}")
        raise Exception(f"Failed to analyze speech: {str(e)}")


def generate_session_summary(analysis: Dict[str, Any], transcript: str) -> str:
    """Generate a neutral, reflective summary of the session.
    
    Args:
        analysis: The analysis results from analyze_speech
        transcript: The full transcript text
        
    Returns:
        A neutral, non-judgmental summary string
    """
    flagged = analysis.get("flaggedCount", 0)
    positive = analysis.get("positiveCount", 0)
    score = analysis.get("score", 50)
    reasoning = analysis.get("reasoning", "")
    
    prompt = f"""Generate a brief, neutral, and reflective summary of this speech session.

Analysis results:
- Flagged interactions: {flagged}
- Positive interactions: {positive}
- Overall score: {score}/100
- Reasoning: {reasoning}

Guidelines:
- Be neutral and non-judgmental
- Focus on patterns, not individual statements
- Use encouraging language
- Keep it brief (2-3 sentences)
- Frame it as self-reflection, not criticism

Generate a summary that helps the user reflect on their speech patterns."""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": "You are a helpful assistant that provides neutral, reflective summaries of speech patterns. Be encouraging and non-judgmental."
                },
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=200,
        )
        
        summary = response.choices[0].message.content.strip()
        return summary
        
    except Exception as e:
        print(f"[AI] Error generating summary: {e}")
        # Fallback summary
        if flagged == 0 and positive > 0:
            return "Your speech in this session was primarily positive and constructive."
        elif flagged > 0:
            return f"You had {flagged} flagged interaction(s) in this session. Consider focusing on more direct, positive conversations."
        else:
            return "This session has been recorded. Review your speech patterns to identify areas for improvement."

