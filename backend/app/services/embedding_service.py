"""
Local MFCC Speaker Embedding Service

Extracts speaker embeddings using MFCC (Mel-frequency cepstral coefficients) features.
This replaces the previous HuggingFace-based embedding extraction.
"""
import os
import numpy as np
import scipy.io.wavfile as wavfile
from scipy.fft import dct
from typing import List
import logging

logger = logging.getLogger(__name__)

def extract_speaker_embedding(audio_path: str) -> List[float]:
    """
    Extract speaker embedding from audio file using local MFCC features.
    
    This function:
    1. Loads audio file (16kHz mono WAV expected)
    2. Extracts MFCC features
    3. Computes statistics (mean, std) across time frames
    4. Returns a 120-dimensional embedding vector
    
    Args:
        audio_path: Path to audio file (WAV format, 16kHz mono)
        
    Returns:
        List[float]: 120-dimensional speaker embedding vector
        
    Raises:
        FileNotFoundError: If audio file doesn't exist
        ValueError: If audio processing fails
    """
    if not os.path.exists(audio_path):
        raise FileNotFoundError(f"Audio file not found: {audio_path}")
    
    try:
        # Load audio file
        sample_rate, audio = wavfile.read(audio_path)
        
        # Ensure mono (single channel)
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # Convert to float32 and normalize
        audio = audio.astype(np.float32)
        if audio.max() > 0:
            audio = audio / np.abs(audio).max()
        
        # Ensure 16kHz sample rate (resample if needed)
        if sample_rate != 16000:
            from scipy import signal
            num_samples = int(len(audio) * 16000 / sample_rate)
            audio = signal.resample(audio, num_samples)
            sample_rate = 16000
        
        # Extract MFCC features
        mfcc_features = extract_mfcc(audio, sample_rate)
        
        # Compute statistics across time frames to create fixed-size embedding
        # Use mean and std of MFCC coefficients across time
        embedding = []
        
        # Mean of each MFCC coefficient across time
        embedding.extend(np.mean(mfcc_features, axis=0).tolist())
        
        # Std of each MFCC coefficient across time
        embedding.extend(np.std(mfcc_features, axis=0).tolist())
        
        # Additional statistics: min, max for robustness
        embedding.extend(np.min(mfcc_features, axis=0).tolist())
        embedding.extend(np.max(mfcc_features, axis=0).tolist())
        
        # Pad or truncate to exactly 120 dimensions
        target_dim = 120
        if len(embedding) < target_dim:
            # Pad with zeros
            embedding.extend([0.0] * (target_dim - len(embedding)))
        elif len(embedding) > target_dim:
            # Truncate
            embedding = embedding[:target_dim]
        
        # L2 normalize
        embedding_array = np.array(embedding)
        norm = np.linalg.norm(embedding_array)
        if norm > 0:
            embedding_array = embedding_array / norm
        
        embedding = embedding_array.tolist()
        
        logger.info(f"Extracted {len(embedding)}-dim MFCC embedding from {audio_path}")
        
        return embedding
        
    except Exception as e:
        logger.error(f"Error extracting MFCC embedding from {audio_path}: {e}")
        raise ValueError(f"Failed to extract speaker embedding: {str(e)}")


def extract_mfcc(audio: np.ndarray, sample_rate: int, n_mfcc: int = 13, n_fft: int = 512, hop_length: int = 160) -> np.ndarray:
    """
    Extract MFCC features from audio signal.
    
    Args:
        audio: Audio signal as numpy array
        sample_rate: Sample rate in Hz
        n_mfcc: Number of MFCC coefficients to extract
        n_fft: FFT window size
        hop_length: Hop length for STFT
        
    Returns:
        np.ndarray: MFCC features (time_frames x n_mfcc)
    """
    # Pre-emphasis filter
    pre_emphasis = 0.97
    emphasized = np.append(audio[0], audio[1:] - pre_emphasis * audio[:-1])
    
    # Framing
    frame_length = n_fft
    frame_step = hop_length
    signal_length = len(emphasized)
    num_frames = int(np.ceil(float(np.abs(signal_length - frame_length)) / frame_step))
    
    pad_signal_length = num_frames * frame_step + frame_length
    z = np.zeros((pad_signal_length - signal_length))
    pad_signal = np.append(emphasized, z)
    
    indices = np.tile(np.arange(0, frame_length), (num_frames, 1)) + \
              np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_length, 1)).T
    frames = pad_signal[indices.astype(np.int32, copy=False)]
    
    # Apply Hamming window
    frames *= np.hamming(frame_length)
    
    # FFT and Power Spectrum
    mag_frames = np.absolute(np.fft.rfft(frames, n_fft))
    pow_frames = ((1.0 / n_fft) * ((mag_frames) ** 2))
    
    # Mel Filter Bank
    nfilt = 40
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (sample_rate / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, nfilt + 2)
    hz_points = (700 * (10**(mel_points / 2595) - 1))
    bin = np.floor((n_fft + 1) * hz_points / sample_rate)
    
    fbank = np.zeros((nfilt, int(np.floor(n_fft / 2 + 1))))
    for m in range(1, nfilt + 1):
        f_m_minus = int(bin[m - 1])
        f_m = int(bin[m])
        f_m_plus = int(bin[m + 1])
        
        for k in range(f_m_minus, f_m_plus):
            if k < f_m:
                fbank[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
            else:
                fbank[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])
    
    # Apply Mel filterbank to power spectrum
    filter_banks = np.dot(pow_frames, fbank.T)
    filter_banks = np.where(filter_banks == 0, np.finfo(float).eps, filter_banks)
    filter_banks = 20 * np.log10(filter_banks)
    
    # DCT to get MFCC coefficients
    mfcc = dct(filter_banks, type=2, axis=1, norm='ortho')[:, 1 : (n_mfcc + 1)]
    
    return mfcc

