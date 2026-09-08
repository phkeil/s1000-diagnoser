"""Utilities for loading and preparing audio signals."""

from pathlib import Path
from typing import List, Tuple

import librosa
import numpy as np


def load_audio(file_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """Load an audio file with librosa."""
    audio, sr = librosa.load(Path(file_path), sr=sample_rate, mono=True)
    return audio, sr


def trim_audio(signal: np.ndarray, start_sample: int, end_sample: int) -> np.ndarray:
    """Return a sliced audio segment."""
    start = max(start_sample, 0)
    end = min(max(end_sample, start), len(signal))
    return signal[start:end]


def chunk_audio(
    signal: np.ndarray,
    sr: int,
    segment_duration: float,
    step_duration: float,
) -> List[Tuple[float, np.ndarray]]:
    """Split a signal into overlapping fixed-length segments.

    Sliding-window logic consolidated from the offline preprocessing pipeline
    in notebooks/02_preprocessing.ipynb, which both dataset-generation and live
    inference must use identically (notebook 04's ad-hoc live-dashboard loop
    used a slightly different `range()` bound that drops the final boundary
    window — this version keeps notebook 02's inclusive formula since that's
    the one the training data was actually built with).

    Returns a list of (start_time_seconds, segment_samples) tuples. Segments
    shorter than `segment_duration` are dropped rather than padded.
    """
    segment_samples = int(segment_duration * sr)
    step_samples = int(step_duration * sr)

    if len(signal) < segment_samples or step_samples <= 0:
        return []

    num_windows = int((len(signal) - segment_samples) / step_samples) + 1

    segments = []
    for i in range(num_windows):
        start_sample = i * step_samples
        end_sample = start_sample + segment_samples
        segments.append((start_sample / sr, signal[start_sample:end_sample]))

    return segments
