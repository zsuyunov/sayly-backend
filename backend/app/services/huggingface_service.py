"""
Speaker Embedding Service – Local MFCC-based extraction.

Computes speaker embeddings entirely on the server using MFCC features.
No external API calls, no billing, no quota issues.

The embedding is a 120-dimensional L2-normalised vector built from:
  20 MFCCs  +  20 deltas  +  20 delta-deltas   =  60 features per frame
  mean + std across all frames                  = 120 dimensions
"""
import os
import wave
import numpy as np
from typing import List
from scipy.fftpack import dct

# ── MFCC hyper-parameters ────────────────────────────────────────────────────
SAMPLE_RATE = 16000
N_MFCC = 20           # number of cepstral coefficients
N_MELS = 40           # mel filterbank bins
N_FFT = 512           # FFT window
WIN_LENGTH = 400      # 25 ms at 16 kHz
HOP_LENGTH = 160      # 10 ms at 16 kHz
PRE_EMPHASIS = 0.97
FMIN = 80             # mel filterbank lower edge (Hz)
FMAX = 7600           # mel filterbank upper edge (Hz)


# ── Kept for backward compatibility (debug endpoints read HF_API_KEY) ────────
def get_hf_api_key() -> str:
    """Return HF_API_KEY from env (kept for backward-compat with debug endpoints)."""
    return os.getenv("HF_API_KEY", "")


# ── Internal helpers ─────────────────────────────────────────────────────────

def _hz_to_mel(hz):
    return 2595.0 * np.log10(1.0 + hz / 700.0)


def _mel_to_hz(mel):
    return 700.0 * (10.0 ** (mel / 2595.0) - 1.0)


def _create_mel_filterbank(n_fft, n_mels, sample_rate, fmin, fmax):
    """Create a triangular mel filterbank matrix  (n_mels × n_fft//2+1)."""
    mel_min = _hz_to_mel(fmin)
    mel_max = _hz_to_mel(fmax)
    mel_points = np.linspace(mel_min, mel_max, n_mels + 2)
    hz_points = _mel_to_hz(mel_points)
    bins = np.floor((n_fft + 1) * hz_points / sample_rate).astype(int)

    n_bins = n_fft // 2 + 1
    fb = np.zeros((n_mels, n_bins))

    for i in range(n_mels):
        left, centre, right = bins[i], bins[i + 1], bins[i + 2]
        for j in range(left, centre):
            if centre != left:
                fb[i, j] = (j - left) / (centre - left)
        for j in range(centre, right):
            if right != centre:
                fb[i, j] = (right - j) / (right - centre)
    return fb


def _compute_mfcc(signal: np.ndarray, sample_rate: int) -> np.ndarray:
    """Return (num_frames, N_MFCC) MFCC matrix from a 1-D float signal."""
    # Pre-emphasis
    emph = np.append(signal[0], signal[1:] - PRE_EMPHASIS * signal[:-1])

    # Framing
    num_frames = 1 + (len(emph) - WIN_LENGTH) // HOP_LENGTH
    if num_frames < 1:
        raise ValueError(f"Audio too short for MFCC extraction ({len(signal)} samples)")

    indices = (
        np.arange(WIN_LENGTH)[None, :] +
        (np.arange(num_frames) * HOP_LENGTH)[:, None]
    )
    frames = emph[indices]

    # Hamming window
    frames *= np.hamming(WIN_LENGTH)

    # Power spectrum
    mag = np.abs(np.fft.rfft(frames, N_FFT))
    power = (mag ** 2) / N_FFT

    # Mel filterbank → log → DCT
    mel_fb = _create_mel_filterbank(N_FFT, N_MELS, sample_rate, FMIN, FMAX)
    mel_spec = np.dot(power, mel_fb.T)
    mel_spec = np.maximum(mel_spec, np.finfo(float).eps)
    log_mel = np.log(mel_spec)

    mfcc = dct(log_mel, type=2, axis=1, norm="ortho")[:, :N_MFCC]
    return mfcc


def _compute_deltas(features: np.ndarray, width: int = 2) -> np.ndarray:
    """Compute delta (first-derivative) features."""
    n_frames, n_feat = features.shape
    denom = 2 * sum(i * i for i in range(1, width + 1))
    deltas = np.zeros_like(features)
    for t in range(n_frames):
        for n in range(1, width + 1):
            t_prev = max(0, t - n)
            t_next = min(n_frames - 1, t + n)
            deltas[t] += n * (features[t_next] - features[t_prev])
        deltas[t] /= denom
    return deltas


# ── Public API (same signature as before) ────────────────────────────────────

def extract_speaker_embedding(audio_path: str) -> List[float]:
    """Extract a 120-dim MFCC-based speaker embedding from a 16 kHz mono WAV.

    Steps:
      1.  Read WAV (already normalised to 16 kHz mono by our pipeline)
      2.  Compute 20 MFCCs per frame
      3.  Append deltas & delta-deltas  →  60 features / frame
      4.  Aggregate with mean + std     → 120 dimensions
      5.  L2-normalise for cosine similarity

    Args:
        audio_path: Path to a 16 kHz mono WAV file.

    Returns:
        List of 120 floats (L2-normalised).
    """
    print(f"[EMBEDDING] Extracting local MFCC embedding from {audio_path}")

    # ---- Read WAV ----
    try:
        with wave.open(audio_path, "rb") as wf:
            n_channels = wf.getnchannels()
            sample_width = wf.getsampwidth()
            frame_rate = wf.getframerate()
            n_frames = wf.getnframes()
            raw = wf.readframes(n_frames)
    except Exception as e:
        raise Exception(f"Failed to read audio file {audio_path}: {e}")

    # ---- Convert to float64 in [-1, 1] ----
    dtype_map = {1: np.uint8, 2: np.int16, 4: np.int32}
    if sample_width not in dtype_map:
        raise ValueError(f"Unsupported sample width: {sample_width}")

    signal = np.frombuffer(raw, dtype=dtype_map[sample_width]).astype(np.float64)
    if sample_width == 1:
        signal = (signal - 128.0) / 128.0
    else:
        signal = signal / (2 ** (8 * sample_width - 1))

    # Mono mix if stereo
    if n_channels == 2:
        signal = signal.reshape(-1, 2).mean(axis=1)
    elif n_channels > 2:
        signal = signal.reshape(-1, n_channels).mean(axis=1)

    print(
        f"[EMBEDDING] Audio: {len(signal)} samples, {frame_rate} Hz, "
        f"{n_channels}ch, {sample_width}B, duration={len(signal)/frame_rate:.2f}s"
    )

    if len(signal) < WIN_LENGTH:
        raise Exception(
            f"Audio too short ({len(signal)} samples, "
            f"need at least {WIN_LENGTH} = {WIN_LENGTH/SAMPLE_RATE*1000:.0f} ms)"
        )

    # ---- Feature extraction ----
    mfcc = _compute_mfcc(signal, frame_rate)
    deltas = _compute_deltas(mfcc)
    delta2 = _compute_deltas(deltas)

    features = np.concatenate([mfcc, deltas, delta2], axis=1)  # (frames, 60)

    # ---- Aggregate → fixed-length vector ----
    mean_f = np.mean(features, axis=0)
    std_f = np.std(features, axis=0)
    embedding = np.concatenate([mean_f, std_f])  # 120-dim

    # ---- L2 normalise ----
    norm = np.linalg.norm(embedding)
    if norm > 0:
        embedding = embedding / norm

    emb_list = embedding.tolist()
    print(f"[EMBEDDING] Done – {len(emb_list)}-dim vector (L2-normalised)")
    return emb_list
