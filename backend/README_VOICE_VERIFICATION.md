# Voice Verification System Documentation (V1)

## Overview

The voice verification system enables speaker identification using speaker embeddings extracted via Hugging Face's ECAPA-VoxCeleb model. **V1 implements a simplified binary decision policy** (OWNER vs OTHER) for improved user experience, while maintaining comprehensive internal logging for research and debugging.

## Architecture

### Components

1. **Enrollment**: User records 3 voice samples → Quality validation → Extract embeddings → Store individually in Firestore
2. **Verification**: Session audio → Split into chunks → Verify each chunk → Label chunks (OWNER/OTHER) → Filter text segments during analysis
3. **Decision Policy (V1)**: Binary decision - OWNER (included in analysis) or OTHER (excluded from analysis). Internal states (UNCERTAIN, SKIPPED) are logged but mapped to OWNER/OTHER for user-facing results.

### Data Flow (V1)

```
Enrollment:
User Records 3 Samples → Quality Check → Extract Embeddings → Store All 3 Individually

Verification (V1 - Filtering Mode):
IF Voice Registered:
  Session Audio → Split into 12s Chunks → For Each Chunk:
    → Extract Embedding
    → Compare with All 3 Enrollment Embeddings
    → Compute: max(similarities)
    → Apply V1 Binary Decision Policy:
       - maxSimilarity >= ownerThreshold → OWNER
       - maxSimilarity < ownerThreshold → OTHER
    → Store Decision + Similarities (internal states logged)
ELSE (No Voice Registration):
  → Verification DISABLED
  → All speech processed transparently
  → Disclaimer added to session

Processing (V1 - No Blocking):
Full Audio → Transcribe (STT) → Filter Text Segments (if verification enabled):
  → OWNER chunks → Included in analysis
  → OTHER chunks → Excluded from analysis
  → Errors → Mapped to OWNER (never block session)
  → No Registration → All speech included with disclaimer

Key V1 Principles:
1. Verification filters, doesn't block. Sessions are never discarded.
2. Without voice registration, verification is DISABLED. All speech is analyzed transparently.
3. Default thresholds only apply when voice is registered. They define confidence boundaries, not ownership determination.
```

## Key Features

### 1. Individual Embedding Storage

**Why**: Averaging embeddings loses information. Individual embeddings preserve variability and enable better verification.

**Implementation**:
- Store all 3 enrollment embeddings separately in `enrollmentEmbeddings: List[List[float]]`
- Compute similarity against each embedding
- Use max similarity and top-K mean (K=2) for decision

**Firestore Schema**:
```python
{
  "enrollmentEmbeddings": [[float, ...], [float, ...], [float, ...]],  # 3 embeddings
  "enrollmentMetadata": [
    {"index": 0, "extractedAt": timestamp, "similarityToOthers": [float, float]},
    ...
  ]
}
```

### 2. Dynamic Threshold System

**Why**: Hardcoded thresholds don't adapt to different environments or user populations.

**Implementation**:
- Thresholds stored in Firestore `verification_thresholds` collection (per environment)
- Fallback to environment variables: `VERIFICATION_OWNER_THRESHOLD`, `VERIFICATION_UNCERTAIN_THRESHOLD`
- Similarity scores logged for distribution analysis and calibration
- No hardcoded values in code

**Default Thresholds (V1)**:
- OWNER: ≥ 0.75 (max similarity only - simplified policy)
- OTHER: < 0.75 (max similarity)
- Internal: UNCERTAIN state (0.6-0.75) is logged but mapped to OTHER for user-facing decision

**IMPORTANT**: Thresholds only apply when voice is registered. Without registration:
- Verification is DISABLED
- All speech is analyzed transparently
- A disclaimer is added to the session indicating verification was not performed
- Thresholds do NOT determine ownership - they only define confidence boundaries once a registered voice reference exists

### 3. Chunk-Level Verification

**Why**: Full-session verification can miss speaker changes. Chunk-level processing enables:
- Detection of speaker changes mid-session
- More granular control over what gets processed
- Better handling of long sessions

**Implementation (V1)**:
- Split audio into 12-second chunks (configurable, 10-15s range)
- Verify each chunk independently
- Label chunks as OWNER or OTHER (binary decision)
- Store per-chunk decisions with timing metadata
- **No audio reconstruction** - full audio is transcribed, text segments are filtered based on chunk labels

**Chunk Processing**:
```python
chunks = split_audio(audio_path, chunk_duration=12.0)
for chunk in chunks:
    decision = verify_chunk(chunk, enrollment_embeddings)
    # decision.decision: OWNER/UNCERTAIN/OTHER/SKIPPED
```

### 4. Error Handling (V1 - Never Block)

**V1 Principle**: Never silently discard a full user session. This avoids user frustration and trust loss.

**Behavior (V1)**:
- API timeout → Status: SKIPPED (internal), Decision: OWNER (user-facing), shouldProcess: True
- Rate limit (429) → Status: SKIPPED (internal), Decision: OWNER (user-facing), shouldProcess: True
- Cold start (503) → Status: SKIPPED (internal), Decision: OWNER (user-facing), shouldProcess: True
- Network error → Status: SKIPPED (internal), Decision: OWNER (user-facing), shouldProcess: True

**Rationale**: Errors are logged internally for debugging, but sessions are always processed to avoid user frustration. Verification improves analysis quality but doesn't block it.
**Result (V1)**: Errors are logged internally, but sessions are always processed. User may see subtle notification if verification was unavailable, but session is never blocked.

### 5. Audio Quality Validation

**Why**: Poor quality enrollment samples lead to poor verification accuracy.

**Validation Rules**:
- Duration: ≥ 8 seconds
- Silence ratio: ≤ 30%
- RMS/loudness: Above threshold
- Sample rate: 16kHz (with tolerance)
- Channels: Mono

**Rejection**: Clear error messages guide user to re-record.

### 6. Model Versioning

**Why**: Model updates can change embedding space. Need to track compatibility.

**Tracking**:
- Model ID: `speechbrain/spkrec-ecapa-voxceleb`
- Model revision: Git commit hash or branch
- Internal version: Our versioning scheme
- Extraction timestamp

**Compatibility**:
- Check on verification
- Warn if revision changed
- Block if version incompatible

## Decision Policy (V1 - Simplified)

### V1 Binary Decision Rules

**User-Facing Decision (Binary)**:
```
IF Voice Registered:
  OWNER:
    - maxSimilarity >= ownerThreshold (default: 0.75)
    → Speech is treated as user's own
    → Included in transcription, analysis, summaries, and reports
  
  OTHER:
    - maxSimilarity < ownerThreshold
    → Speech is ignored for analysis
    → Not transcribed, not counted in productivity, gossip, or reports

ELSE (No Voice Registration):
  → Verification DISABLED
  → All speech processed transparently
  → Disclaimer: "Voice recognition not enabled. All speech analyzed transparently."
  → Thresholds do NOT apply - they only define confidence boundaries when registration exists
```

**Internal States (Logged, Not Exposed)**:
```
UNCERTAIN (Internal):
  - uncertainThreshold <= maxSimilarity < ownerThreshold (default: 0.6-0.75)
  → Logged internally for research/debugging
  → Mapped to OTHER for user-facing decision

SKIPPED (Internal):
  - Verification failed (API error, timeout, etc.)
  → Logged internally with error details
  → Mapped to OWNER for user-facing decision (never block session)
```

### Why V1 Simplified Policy?

✅ **Deterministic**: Simple threshold check, easy to reason about  
✅ **Explainable**: Clear rule: "If similarity >= threshold, it's you"  
✅ **Easier to Test**: Binary decision simplifies test cases  
✅ **User-Friendly**: No confusing intermediate states  
✅ **Sufficient**: Adequate for reflective self-improvement app  

**Academic Justification**:  
"In the first version of the system, speaker verification is simplified to a binary decision (owner vs. other) to reduce system complexity and improve user experience. Verification is used to label speech segments rather than block session processing, ensuring that users are not penalized for transient technical issues or environmental noise. More granular confidence states are logged internally for future analysis but are not exposed at the user interface level."

### Similarity Computation

1. **Max Similarity**: `max(similarity(session_emb, emb) for emb in enrollment_embeddings)`
   - Best match across all enrollment samples
   - Robust to one poor enrollment sample

2. **Top-K Mean**: `mean(sorted(similarities)[-2:])` where K=2
   - Average of top 2 similarities
   - More stable than max alone
   - Reduces impact of outliers

3. **All Similarities**: Stored for audit/debugging
   - Enables analysis of enrollment quality
   - Helps with threshold calibration

## Security & Privacy

### User Isolation

- All endpoints require authentication (`get_current_user` dependency)
- Embeddings stored with `uid` as document ID
- No cross-user access possible (Firestore security rules)
- Rate limiting on enrollment endpoint

### Data Protection

- Embeddings encrypted at rest (Firestore managed encryption)
- Audio files deleted immediately after embedding extraction
- No raw audio stored long-term
- User can delete voice profile at any time

### Audit Trail

- All verification attempts logged
- Similarity scores stored for calibration
- Model metadata tracked per verification
- Chunk-level decisions stored in session

## Configuration

### Environment Variables

```bash
# Hugging Face API
HF_API_KEY=your_api_key_here

# Thresholds (optional, defaults provided)
VERIFICATION_OWNER_THRESHOLD=0.75
VERIFICATION_UNCERTAIN_THRESHOLD=0.6
VERIFICATION_ENVIRONMENT=dev  # dev/test/prod

# Model Versioning
HF_MODEL_REVISION=main  # or commit hash
INTERNAL_MODEL_VERSION=1.0.0
```

### Firestore Collections

- `voice_profiles`: User voice profiles with embeddings
- `verification_thresholds`: Threshold configurations per environment
- `verification_logs`: Similarity score logs for calibration

## API Endpoints

### POST `/api/voice/enroll`
- Accepts 3 WAV files (16kHz mono)
- Validates quality before processing
- Stores all 3 embeddings individually
- Returns success/error with clear messages

### POST `/api/voice/verify`
- Accepts session audio embedding
- Compares against all enrollment embeddings
- Returns decision + similarity scores
- Uses dynamic thresholds

### GET `/api/voice/status`
- Returns registration status
- Includes model version info

### DELETE `/api/voice/delete`
- Removes voice profile
- Deletes all embeddings

## Limitations & Future Improvements

### Current Limitations

1. **Chunk Overlap**: 0.5s overlap may cut words. Consider VAD (Voice Activity Detection) for smarter splitting.

2. **Threshold Calibration**: Manual process. Future: Automatic calibration based on similarity distributions.

3. **Model Updates**: Manual re-enrollment required. Future: Embedding space alignment/translation.

4. **Multi-Speaker Sessions**: Current system assumes single speaker. Future: Speaker diarization integration.

5. **Noise Handling**: Quality validation is basic. Future: SNR (Signal-to-Noise Ratio) checks.

### Academic Defensibility

- **Similarity Metrics**: Max similarity is standard in speaker verification literature
- **Decision Policy (V1)**: Simplified binary decision improves UX while maintaining internal logging for research
- **Audit Trail**: Complete logging enables reproducibility (internal states preserved)
- **Model Versioning**: Tracks which model version produced which results
- **V1 Rationale**: "Verification is used to label speech segments rather than block session processing, ensuring that users are not penalized for transient technical issues or environmental noise."

### Known Edge Cases

1. **Very Short Sessions**: < 12 seconds may produce only 1 chunk
2. **Silent Chunks**: Chunks with only silence may have low similarity (handled by quality validation)
3. **Background Noise**: High noise may affect embedding quality (partially handled by RMS check)
4. **Voice Changes**: User's voice may change (illness, age) - requires re-enrollment

## Code Flow

### Enrollment Flow

```
1. Receive 3 audio files
2. For each file:
   a. Validate quality (duration, silence, RMS)
   b. Extract embedding via HF API
   c. Store embedding
3. Compute inter-enrollment similarities
4. Store all 3 embeddings + metadata in Firestore
5. Cleanup temporary files
```

### Verification Flow (V1 - Chunk-Level Filtering)

```
1. Check if user has enrolled voice
2. Split session audio into 12s chunks
3. For each chunk:
   a. Extract embedding via HF API
   b. Compare with all enrollment embeddings
   c. Compute max similarity
   d. Apply v1 binary decision (OWNER/OTHER)
   e. Store decision + internal state for logging
4. Transcribe full audio (no reconstruction)
5. Filter text segments based on chunk labels:
   - OWNER chunks → Include in analysis
   - OTHER chunks → Exclude from analysis
6. Generate session summary from filtered text
3. For each chunk:
   a. Extract embedding
   b. Compare with all 3 enrollment embeddings
   c. Compute max similarity and top-K mean
   d. Apply decision policy (dynamic thresholds)
   e. Store decision + similarities
4. Reconstruct audio from OWNER + UNCERTAIN chunks
5. Process reconstructed audio (STT, analysis, summary)
6. Store chunk-level results in session
7. Cleanup chunk files
```

### Error Handling Flow

```
1. Try to extract embedding
2. If error:
   a. Detect error type (timeout, rate_limit, etc.)
   b. Return VerificationResult with:
      - status: "SKIPPED"
      - decision: "BLOCKED"
      - shouldProcess: False
      - errorReason: specific reason
3. Analysis flow checks shouldProcess
4. If False → Block processing, return user-safe message
```

## Testing Recommendations

1. **Threshold Calibration**: Collect similarity distributions, adjust thresholds
2. **Chunk Boundary Testing**: Test with audio that has speaker changes at chunk boundaries
3. **Error Scenarios**: Test timeout, rate limit, network failure
4. **Quality Validation**: Test with various audio qualities (quiet, noisy, short)
5. **Model Compatibility**: Test with different model versions

## Maintenance

### Threshold Calibration

1. Query similarity distributions: `compute_similarity_distribution(uid=None)`
2. Analyze percentiles (p50, p75, p95)
3. Adjust thresholds based on false positive/negative rates
4. Update Firestore configuration

### Model Updates

1. Update `HF_MODEL_REVISION` or `INTERNAL_MODEL_VERSION`
2. System will detect incompatibility on next verification
3. Users will be prompted to re-enroll
4. Old embeddings remain for backward compatibility (with warning)

## References

- ECAPA-TDNN: [SpeechBrain Documentation](https://speechbrain.github.io/)
- Hugging Face Inference API: [HF API Docs](https://huggingface.co/docs/api-inference)
- Speaker Verification: Standard cosine similarity approach

