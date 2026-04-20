"""Feature extraction helpers (FFT, MFCC, and spectrogram)."""

from typing import Dict

import librosa
import numpy as np


def extract_fft(signal: np.ndarray) -> np.ndarray:
    """Compute FFT magnitudes for a signal."""
    return np.abs(np.fft.rfft(signal))


def extract_mfcc(signal: np.ndarray, sample_rate: int = 16000, n_mfcc: int = 13) -> np.ndarray:
    """Compute MFCC features."""
    return librosa.feature.mfcc(y=signal, sr=sample_rate, n_mfcc=n_mfcc)


def extract_log_spectrogram(signal: np.ndarray, sample_rate: int = 16000) -> np.ndarray:
    """Compute log-scaled spectrogram features."""
    stft = np.abs(librosa.stft(signal))
    return librosa.amplitude_to_db(stft, ref=np.max)


def extract_all(signal: np.ndarray, sample_rate: int = 16000) -> Dict[str, np.ndarray]:
    """Return a common set of audio features."""
    return {
        "fft": extract_fft(signal),
        "mfcc": extract_mfcc(signal, sample_rate=sample_rate),
        "log_spectrogram": extract_log_spectrogram(signal, sample_rate=sample_rate),
    }
