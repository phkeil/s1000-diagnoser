"""Utilities for loading and preparing audio signals."""

from pathlib import Path
from typing import Tuple

import librosa
import numpy as np


def load_audio(file_path: str, sample_rate: int = 16000) -> Tuple[np.ndarray, int]:
    """Load an audio file with librosa."""
    audio, sr = librosa.load(Path(file_path), sr=sample_rate, mono=True)
    return audio, sr


def trim_audio(signal: np.ndarray, start_sample: int, end_sample: int) -> np.ndarray:
    """Return a sliced audio segment."""
    start = max(start_sample, 0)
    end = max(end_sample, start)
    return signal[start:end]
