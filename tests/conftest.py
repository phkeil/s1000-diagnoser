"""Shared pytest fixtures.

Everything here is synthetic (a generated sine wave, a randomly-initialized
MotorAutoencoder checkpoint) — no test may depend on files under data/ or
models/, since those are gitignored and won't exist in a fresh CI checkout.
"""

import numpy as np
import pytest
import torch
from scipy.io import wavfile

from src.config import load_config
from src.inference import load_model
from src.model import MotorAutoencoder

SINE_FREQUENCY_HZ = 440.0
SINE_DURATION_SECONDS = 3.0


@pytest.fixture(scope="session")
def cfg():
    """The real config/config.yaml — tracked in git, not "real data"."""
    return load_config()


@pytest.fixture
def device():
    # Force CPU: CI has no GPU/MPS, and results must be deterministic across machines.
    return torch.device("cpu")


@pytest.fixture
def sine_wave_signal(cfg):
    """A synthetic mono sine wave, long enough to yield several chunk_audio segments."""
    sr = cfg.audio.sample_rate
    t = np.linspace(0, SINE_DURATION_SECONDS, int(sr * SINE_DURATION_SECONDS), endpoint=False)
    signal = (0.5 * np.sin(2 * np.pi * SINE_FREQUENCY_HZ * t)).astype(np.float32)
    return signal, sr


@pytest.fixture
def sine_wave_audio_file(tmp_path, sine_wave_signal):
    """The sine wave above, written out as a real 16-bit PCM .wav file on disk."""
    signal, sr = sine_wave_signal
    path = tmp_path / "sine.wav"
    pcm16 = (signal * np.iinfo(np.int16).max).astype(np.int16)
    wavfile.write(path, sr, pcm16)
    return path


@pytest.fixture
def synthetic_checkpoint_path(tmp_path, cfg):
    """A randomly-initialized (untrained) MotorAutoencoder, saved like a real
    checkpoint. Tests must never load models/s1000_CAE_MEL_annomaly.pth."""
    torch.manual_seed(0)
    model = MotorAutoencoder(bottleneck_size=cfg.model.bottleneck_size)
    path = tmp_path / "synthetic_cae.pth"
    torch.save(model.state_dict(), path)
    return path


@pytest.fixture
def synthetic_model(synthetic_checkpoint_path, cfg, device):
    return load_model(cfg, checkpoint_path=synthetic_checkpoint_path, device=device)
