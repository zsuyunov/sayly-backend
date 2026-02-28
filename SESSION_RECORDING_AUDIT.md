# Session Recording Feature Audit Report

**Date:** 2025-01-27  
**Auditor:** Senior Mobile + Backend Engineer  
**Scope:** React Native (Expo) + FastAPI Session Recording Implementation

---

## Executive Summary

The session recording feature is **functionally complete** with solid authentication and privacy controls, but has **critical gaps** for AI readiness and production reliability. The system uses expo-av with HIGH_QUALITY preset (format/sample rate unspecified), lacks interruption handling, and has no retry/cleanup mechanisms.

**Overall Status:** ⚠️ **Needs Work Before AI Integration**

---

## ✅ PASSED Items (Correctly Implemented)

### 1. Mobile Recording Layer
- ✅ **Audio permissions handled correctly** - `checkMicrophonePermission()` and `requestMicrophonePermission()` properly implemented
- ✅ **Start/stop logic is reliable** - Proper error handling with cleanup on failure
- ✅ **User feedback is clear** - Visual indicators (breathing animations, status text) show recording state
- ✅ **Recording stops on logout** - Session cleared from AsyncStorage, recording state managed
- ✅ **Permission fallback UI** - Error messages displayed when permission denied

### 2. Session Lifecycle Logic
- ✅ **Session created before recording** - `startSession()` called after `startRecording()` succeeds
- ✅ **Session ID persists** - Stored in AsyncStorage until upload completes
- ✅ **Session ownership enforced** - Backend validates `uid` on all operations
- ✅ **Status tracking** - `ACTIVE`/`STOPPED` status properly managed

### 3. Audio Upload & Storage
- ✅ **File size limits enforced** - 50MB maximum on backend
- ✅ **Secure file naming** - Uses `{session_id}_{uuid}.ext` (no PII)
- ✅ **MIME type validation** - Backend validates against allowed types
- ✅ **Private storage** - Files stored in local `./audio_storage` directory (not public URLs)

### 4. Backend Session Handling
- ✅ **Auth token validation** - All endpoints use `get_current_user` dependency
- ✅ **Session ownership enforcement** - UID checked on upload, stop, and detail endpoints
- ✅ **MIME type validation** - Comprehensive allowlist of audio types
- ✅ **Error handling** - Proper HTTP status codes and error messages

### 5. Database Session Schema
- ✅ **Core fields present** - `id`, `uid`, `startedAt`, `endedAt`, `status`, `totals`
- ✅ **Audio tracking** - `audioUrl`, `audioProcessed`, `analysisStatus` fields
- ✅ **Status enum** - `ACTIVE`/`STOPPED` properly typed
- ✅ **Analysis status enum** - `PENDING`/`PROCESSING`/`COMPLETED`/`FAILED`

### 6. Privacy & Ethics Readiness
- ✅ **User consent enforced** - `listeningEnabled` privacy setting checked before recording
- ✅ **Recording indicator visible** - UI shows "Listening to your words…" when active
- ✅ **No background recording** - `staysActiveInBackground: false` configured
- ✅ **Privacy settings respected** - `dataAnalysisEnabled` enforced server-side

---

## 🚨 MUST FIX Before AI Integration

### 1. Mobile Recording Layer

#### **CRITICAL: Audio Format Not AI-Ready**
- **Issue:** Using `Audio.RecordingOptionsPresets.HIGH_QUALITY` without explicit format configuration
  - Unknown sample rate (may be 44.1kHz or 48kHz, not 16kHz recommended for AI)
  - Unknown channel configuration (may be stereo, not mono)
  - Format varies by platform (m4a on iOS, mp3 on Android) - not WAV
- **Impact:** AI models (Whisper, speaker diarization) expect 16kHz mono WAV
- **Fix Required:**
  ```typescript
  // Replace HIGH_QUALITY preset with explicit configuration
  const recordingOptions = {
    android: {
      extension: '.wav',
      outputFormat: Audio.AndroidOutputFormat.DEFAULT,
      audioEncoder: Audio.AndroidAudioEncoder.DEFAULT,
      sampleRate: 16000,
      numberOfChannels: 1, // Mono
      bitRate: 128000,
    },
    ios: {
      extension: '.wav',
      outputFormat: Audio.IOSOutputFormat.LINEARPCM,
      audioQuality: Audio.IOSAudioQuality.HIGH,
      sampleRate: 16000,
      numberOfChannels: 1, // Mono
      bitRate: 128000,
      linearPCMBitDepth: 16,
      linearPCMIsBigEndian: false,
      linearPCMIsFloat: false,
    },
  };
  ```

#### **CRITICAL: No Interruption Handling**
- **Issue:** No listeners for:
  - Phone calls interrupting recording
  - App backgrounding (recording stops but no recovery)
  - Audio session interruptions
- **Impact:** Lost recordings, poor user experience
- **Fix Required:**
  - Add `AppState` listener to detect backgrounding
  - Add audio interruption callbacks
  - Implement pause/resume or graceful stop on interruption

#### **HIGH: No Recording Duration Limits**
- **Issue:** No maximum recording duration enforced
- **Impact:** Unbounded file sizes, storage issues, upload failures
- **Fix Required:** Add configurable max duration (e.g., 2 hours) with warning at 90%

#### **HIGH: No Microphone Availability Check**
- **Issue:** Only checks permission, not if microphone is actually available
- **Impact:** Recording may fail silently if mic is in use by another app
- **Fix Required:** Check microphone availability before starting

### 2. Session Lifecycle Logic

#### **CRITICAL: Race Condition in Start Flow**
- **Issue:** If `startSession()` fails after `startRecording()` succeeds, recording continues without session
- **Location:** `mobile/app/(tabs)/session.tsx:217-229`
- **Impact:** Orphan recordings, inconsistent state
- **Fix Required:** Ensure atomicity - if session creation fails, cancel recording

#### **HIGH: No Retry Logic for Failed Uploads**
- **Issue:** Upload failure only shows error message, no retry mechanism
- **Location:** `mobile/app/(tabs)/session.tsx:283-285`
- **Impact:** Lost audio data, poor UX
- **Fix Required:**
  - Implement exponential backoff retry (3 attempts)
  - Store failed uploads in AsyncStorage for retry on app restart
  - Add "Retry Upload" button in UI
- **403/404 and the retry queue:** When the upload queue retries and gets **403** (forbidden) or **404** (session not found), the backend will not accept the upload on retry. The client should **remove that session from the failed-upload queue** (do not retry again). The API sends `X-Upload-Retry: false` on 403 and uses 404 for invalid/missing sessions so the app can treat these as permanent failures and clear them from the queue.

#### **HIGH: No Orphan Audio File Cleanup**
- **Issue:** If upload fails, local audio file remains on device indefinitely
- **Impact:** Storage bloat, privacy risk
- **Fix Required:**
  - Delete local file after successful upload
  - Cleanup old failed uploads after 7 days
  - Add periodic cleanup job

### 3. Audio Upload & Storage

#### **CRITICAL: No Chunked/Resumable Upload**
- **Issue:** Entire file uploaded in one request (up to 50MB)
- **Impact:** 
  - Network failures lose entire upload
  - Poor experience on slow/unstable connections
  - Timeout issues
- **Fix Required:** Implement chunked upload with resume capability

#### **HIGH: Audio Format Not WAV**
- **Issue:** Backend accepts multiple formats, but AI needs WAV
- **Impact:** Requires conversion before AI processing (extra step, quality loss)
- **Fix Required:** 
  - Enforce WAV format on mobile (see fix #1)
  - Add backend validation to reject non-WAV files
  - Or: Add audio normalization pipeline to convert to WAV

#### **MEDIUM: No Audio Normalization Pipeline**
- **Issue:** Backend stores audio as-is, no normalization
- **Impact:** Variable quality, sample rates, channels - harder for AI
- **Fix Required:** Add normalization step after upload:
  - Convert to 16kHz mono WAV
  - Normalize volume levels
  - Remove silence at start/end

### 4. Backend Session Handling

#### **HIGH: No Temporary Storage Cleanup**
- **Issue:** Audio files stored permanently, no cleanup after processing
- **Location:** `backend/app/api/audio.py:159` - files stored in `./audio_storage`
- **Impact:** Unbounded storage growth
- **Fix Required:**
  - Delete audio files after analysis completes (or after 30 days)
  - Or: Move to cloud storage (S3/GCS) with lifecycle policies
  - Add cleanup job for orphaned files

#### **MEDIUM: Path Traversal Risk (Low Priority)**
- **Issue:** Filename generated from `session_id` + `uuid`, but no explicit sanitization
- **Status:** Likely safe (UUIDs don't contain path chars), but should validate
- **Fix Required:** Add filename sanitization before `os.path.join()`

### 5. Database Session Schema

#### **MEDIUM: Missing `error_reason` Field**
- **Issue:** No field to store why upload/analysis failed
- **Impact:** Hard to debug failures, no user feedback on specific errors
- **Fix Required:** Add `error_reason: Optional[str]` to session model

#### **MEDIUM: No Database Indices**
- **Issue:** Firestore queries may be slow without composite indices
- **Impact:** Performance degradation as data grows
- **Fix Required:** Add composite indices:
  - `(uid, status, startedAt)` for active session queries
  - `(uid, startedAt)` for session history
  - `(uid, analysisStatus)` for pending analyses

### 6. AI-Readiness Check

#### **CRITICAL: Audio Format Incompatible**
- **Issue:** Current format (m4a/mp3, unknown sample rate, possibly stereo) not optimal for AI
- **Impact:** 
  - Whisper STT may have lower accuracy
  - Speaker diarization requires mono
  - Speaker embeddings need consistent format
- **Fix Required:** See fix #1 (explicit 16kHz mono WAV configuration)

#### **HIGH: Missing Timestamps in Audio Metadata**
- **Issue:** No timestamps stored for audio segments
- **Impact:** Cannot segment sessions for diarization, harder to correlate with events
- **Fix Required:** Store `recordingStartedAt` and `recordingEndedAt` timestamps

#### **MEDIUM: No Session Segmentation Support**
- **Issue:** Entire session is one audio file
- **Impact:** Cannot analyze segments separately, harder for diarization
- **Fix Required:** Consider chunking long sessions (e.g., 5-minute segments)

---

## 💡 OPTIONAL Enhancements (Not Blocking)

### Mobile Recording Layer
- ⚪ **Pause/Resume functionality** - Currently only start/stop
- ⚪ **Recording quality selector** - Let users choose quality vs. file size
- ⚪ **Background recording indicator** - System notification when app backgrounded
- ⚪ **Audio level visualization** - Show waveform/levels during recording

### Session Lifecycle
- ⚪ **Session templates** - Pre-configured session types
- ⚪ **Session notes before recording** - Add context before starting
- ⚪ **Session sharing** - Export/share session summaries

### Audio Upload & Storage
- ⚪ **Cloud storage integration** - Move from local filesystem to S3/GCS
- ⚪ **Audio compression** - Compress before upload (with quality trade-off)
- ⚪ **Upload progress indicator** - Show progress bar during upload
- ⚪ **Offline queue** - Queue uploads when offline, retry when online

### Backend
- ⚪ **Audio streaming** - Stream audio chunks instead of full file upload
- ⚪ **Audio deduplication** - Detect and deduplicate identical audio
- ⚪ **Audio quality metrics** - Analyze and report audio quality scores
- ⚪ **Batch processing** - Process multiple sessions in parallel

### Database
- ⚪ **Soft delete** - Add `deletedAt` field instead of hard delete
- ⚪ **Session tags** - Allow users to tag sessions for organization
- ⚪ **Session search** - Full-text search on notes and summaries

### Privacy & Ethics
- ⚪ **Recording consent dialog** - Explicit consent before first recording
- ⚪ **Session deletion confirmation** - Confirm before deleting sessions
- ⚪ **Data export** - Allow users to export all their session data
- ⚪ **Recording history** - Show list of all recordings with metadata

---

## 📋 AI Integration Migration Plan

### Phase 1: Audio Format Standardization (Week 1)
**Priority: CRITICAL**

1. **Update Mobile Recording Configuration**
   - Replace `HIGH_QUALITY` preset with explicit 16kHz mono WAV config
   - Test on both iOS and Android
   - Verify file format and properties

2. **Backend Validation**
   - Add validation to reject non-WAV files (or convert on upload)
   - Add audio metadata extraction (sample rate, channels, duration)

3. **Testing**
   - Record test sessions on both platforms
   - Verify WAV format, 16kHz, mono
   - Test upload and processing

### Phase 2: Interruption & Reliability (Week 2)
**Priority: HIGH**

1. **Add Interruption Handling**
   - Implement `AppState` listener
   - Add audio interruption callbacks
   - Test phone call interruption
   - Test app backgrounding

2. **Add Retry Logic**
   - Implement exponential backoff retry for uploads
   - Add failed upload queue in AsyncStorage
   - Add retry UI button

3. **Add Cleanup Logic**
   - Delete local files after successful upload
   - Add periodic cleanup job for old files

### Phase 3: Audio Normalization Pipeline (Week 3)
**Priority: HIGH**

1. **Backend Audio Processing**
   - Add audio normalization service (use `pydub` or `ffmpeg`)
   - Convert all uploads to 16kHz mono WAV
   - Normalize volume levels
   - Remove leading/trailing silence

2. **Storage Optimization**
   - Store normalized WAV files
   - Delete original files after normalization
   - Add cleanup job for processed files

### Phase 4: AI Integration (Week 4)
**Priority: MEDIUM**

1. **Whisper STT Integration**
   - Update `transcribe_audio()` to use normalized WAV
   - Verify transcription quality
   - Add language detection

2. **Speaker Diarization Preparation**
   - Ensure mono audio format
   - Add session segmentation support
   - Store speaker turn timestamps

3. **Speaker Embeddings Preparation**
   - Ensure consistent audio format
   - Add voice registration integration
   - Prepare embedding extraction pipeline

### Phase 5: Monitoring & Optimization (Ongoing)
**Priority: LOW**

1. **Add Metrics**
   - Track upload success/failure rates
   - Monitor audio file sizes
   - Track processing times

2. **Performance Optimization**
   - Add database indices
   - Optimize audio processing
   - Add caching where appropriate

---

## 🔍 Code References

### Critical Files Reviewed
- `mobile/utils/audioRecording.ts` - Recording implementation
- `mobile/app/(tabs)/session.tsx` - Session UI and lifecycle
- `mobile/utils/api.ts` - API client (upload, session management)
- `backend/app/api/sessions.py` - Session endpoints
- `backend/app/api/audio.py` - Audio upload endpoint
- `backend/app/api/analysis.py` - AI analysis processing
- `backend/app/models/session.py` - Session data models

### Key Issues by File

**`mobile/utils/audioRecording.ts:63`**
- Using `HIGH_QUALITY` preset without explicit format
- No interruption handling

**`mobile/app/(tabs)/session.tsx:217-229`**
- Race condition: recording can start without session
- No retry logic for failed uploads

**`backend/app/api/audio.py:159`**
- Audio stored in local filesystem (no cleanup)
- No normalization pipeline

**`backend/app/api/analysis.py:97`**
- Checks file existence but no cleanup after processing

---

## 📊 Risk Assessment

| Risk | Severity | Likelihood | Impact | Priority |
|------|----------|------------|--------|----------|
| Audio format incompatible with AI | HIGH | CERTAIN | Blocks AI integration | P0 |
| No interruption handling | HIGH | LIKELY | Lost recordings | P0 |
| No retry logic | MEDIUM | LIKELY | Lost uploads | P1 |
| No cleanup | MEDIUM | CERTAIN | Storage bloat | P1 |
| Race condition in start | MEDIUM | POSSIBLE | Orphan recordings | P1 |
| No duration limits | LOW | POSSIBLE | Large files | P2 |

---

## ✅ Conclusion

The session recording feature is **production-ready for basic use** but **not ready for AI integration** without the critical fixes outlined above. The most urgent items are:

1. **Audio format standardization** (16kHz mono WAV)
2. **Interruption handling**
3. **Retry logic for uploads**
4. **Storage cleanup**

With these fixes, the system will be ready for AI features (Whisper STT, speaker diarization, embeddings) while maintaining reliability and user experience.

**Estimated Effort:** 3-4 weeks for critical fixes, 1-2 weeks for AI integration.

---

**End of Audit Report**

