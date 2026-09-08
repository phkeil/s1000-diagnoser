"""Anomaly scoring for the CAE, consolidated from notebooks/04_CAE_Annomalydetector.ipynb
(`check_anomaly`, `generate_score_table`, `get_domain_relative_results`,
`evaluate_smoothed_metrics`, and the scoring half of `plot_anomaly_dashboard` —
everything except the matplotlib visualization).

Pipeline: reconstruction MSE per segment -> normalize by the segment's
per-domain healthy baseline -> rolling-mean smoothing -> threshold.
"""

from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import pandas as pd
import torch
import torch.nn.functional as F

from src.audio_utils import chunk_audio, load_audio
from src.config import Config
from src.data import build_transform, preprocess_audio_segment
from src.model import MotorAutoencoder


def get_device() -> torch.device:
    """Same cuda -> mps -> cpu fallback used throughout the notebooks."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def load_model(cfg: Config, checkpoint_path: Optional[Path] = None, device: Optional[torch.device] = None) -> MotorAutoencoder:
    """Instantiate MotorAutoencoder and load trained weights for inference."""
    device = device or get_device()
    checkpoint_path = checkpoint_path or cfg.resolve_path(cfg.model.checkpoint_path)

    model = MotorAutoencoder(bottleneck_size=cfg.model.bottleneck_size).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()  # important for BatchNorm
    return model


@torch.no_grad()
def reconstruction_error(model: MotorAutoencoder, segment_tensor: torch.Tensor, device: torch.device) -> float:
    """MSE between a (C, H, W) input tensor and its reconstruction."""
    input_tensor = segment_tensor.unsqueeze(0).to(device)
    reconstruction = model(input_tensor)
    return F.mse_loss(reconstruction, input_tensor).item()


def infer_domain(name_or_path: str, cfg: Config) -> str:
    """'Garage' if the filename matches one of the known own-recording hints,
    else the configured default domain (matches notebook 04's domain split)."""
    if any(hint in name_or_path for hint in cfg.anomaly.garage_name_hints):
        return "Garage"
    return cfg.anomaly.default_domain


@dataclass
class SegmentScore:
    start_time: float
    raw_score: float
    relative_score: float
    smoothed_score: float
    is_anomalous: bool


@dataclass
class AnomalyResult:
    domain: str
    baseline: float
    threshold: float
    segments: List[SegmentScore]

    @property
    def is_anomalous(self) -> bool:
        return any(s.is_anomalous for s in self.segments)


def score_segments(raw_scores: List[float], start_times: List[float], domain: str, cfg: Config) -> AnomalyResult:
    """Normalize raw reconstruction-MSE scores against the domain baseline, smooth,
    and threshold — the non-plotting half of `plot_anomaly_dashboard`."""
    baseline = cfg.anomaly.domain_baselines[domain]
    threshold = cfg.anomaly.rel_threshold

    relative_scores = [score / baseline for score in raw_scores]
    smoothed_scores = (
        pd.Series(relative_scores)
        .rolling(window=cfg.anomaly.window_size, center=True, min_periods=1)
        .mean()
        .tolist()
    )

    segments = [
        SegmentScore(
            start_time=start_times[i],
            raw_score=raw_scores[i],
            relative_score=relative_scores[i],
            smoothed_score=smoothed_scores[i],
            is_anomalous=smoothed_scores[i] > threshold,
        )
        for i in range(len(raw_scores))
    ]

    return AnomalyResult(domain=domain, baseline=baseline, threshold=threshold, segments=segments)


def score_audio_file(
    audio_path: Path,
    model: MotorAutoencoder,
    cfg: Config,
    device: Optional[torch.device] = None,
) -> AnomalyResult:
    """End-to-end: load audio -> chunk -> preprocess -> reconstruction MSE per
    segment -> domain-normalized, smoothed anomaly result. This is what the
    FastAPI /predict endpoint calls."""
    device = device or get_device()
    transform = build_transform(cfg)

    signal, sr = load_audio(str(audio_path), sample_rate=cfg.audio.sample_rate)
    chunks = chunk_audio(signal, sr, cfg.audio.segment_duration, cfg.audio.step_duration)

    start_times, raw_scores = [], []
    for start_time, segment in chunks:
        segment_tensor = preprocess_audio_segment(segment, sr, cfg, transform=transform)
        raw_scores.append(reconstruction_error(model, segment_tensor, device))
        start_times.append(start_time)

    domain = infer_domain(str(audio_path), cfg)
    return score_segments(raw_scores, start_times, domain, cfg)
