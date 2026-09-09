"""Loader for config/config.yaml — the single source of truth for preprocessing,
model, and anomaly-scoring parameters shared by training and serving code."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Tuple

import yaml

DEFAULT_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"


@dataclass
class AudioConfig:
    sample_rate: int
    segment_duration: float
    overlap: float

    @property
    def step_duration(self) -> float:
        return self.segment_duration * (1 - self.overlap)


@dataclass
class MelSpectrogramConfig:
    n_fft: int
    hop_length: int
    n_mels: int
    fmin: int
    fmax: int
    db_vmin: float
    db_vmax: float


@dataclass
class ImageConfig:
    size: Tuple[int, int]
    resample: str
    colormap: str


@dataclass
class NormalizeConfig:
    mean: List[float]
    std: List[float]


@dataclass
class ModelConfig:
    bottleneck_size: int
    checkpoint_path: str


@dataclass
class AnomalyConfig:
    domain_baselines: Dict[str, float]
    default_domain: str
    garage_name_hints: List[str]
    rel_threshold: float
    window_size: int


@dataclass
class Config:
    audio: AudioConfig
    mel_spectrogram: MelSpectrogramConfig
    image: ImageConfig
    normalize: NormalizeConfig
    model: ModelConfig
    anomaly: AnomalyConfig
    root_dir: Path = field(default=DEFAULT_CONFIG_PATH.parent.parent)

    def resolve_path(self, relative_path: str) -> Path:
        """Resolve a config-relative path (e.g. model.checkpoint_path) against the repo root."""
        return self.root_dir / relative_path


def load_config(path: Path | str = DEFAULT_CONFIG_PATH) -> Config:
    path = Path(path)
    with open(path, "r") as f:
        raw = yaml.safe_load(f)

    return Config(
        audio=AudioConfig(**raw["audio"]),
        mel_spectrogram=MelSpectrogramConfig(**raw["mel_spectrogram"]),
        image=ImageConfig(size=tuple(raw["image"]["size"]), resample=raw["image"]["resample"], colormap=raw["image"]["colormap"]),
        normalize=NormalizeConfig(**raw["normalize"]),
        model=ModelConfig(**raw["model"]),
        anomaly=AnomalyConfig(**raw["anomaly"]),
        root_dir=path.resolve().parent.parent,
    )
