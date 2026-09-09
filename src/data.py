"""Mel-spectrogram preprocessing shared by offline dataset generation, training,
and live inference.

`render_mel_spectrogram_image` / `preprocess_audio_segment` reproduce, parameter
for parameter, the rendering path used to build data/processed/dataset_idle_mel*
in notebooks/02_preprocessing.ipynb (and re-confirmed for live audio in the
"Parameter-Synchronisation" cell of notebooks/04_CAE_Annomalydetector.ipynb):
librosa mel spectrogram -> dB -> matplotlib magma imshow (fixed -80..0dB range)
-> PNG buffer -> PIL RGB -> LANCZOS resize. Skipping the PNG round-trip would
silently change what the model sees, since training data went through it.
"""

import io
import os
from pathlib import Path
from typing import Callable, List, Optional, Sequence, Tuple, Union

import librosa
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

from src.config import Config

PathLike = Union[str, Path]


def build_transform(cfg: Config) -> transforms.Compose:
    """The torch transform applied after image rendering: identical across all
    of notebooks 03/04/07 (Resize -> ToTensor -> ImageNet Normalize)."""
    return transforms.Compose(
        [
            transforms.Resize(tuple(cfg.image.size)),
            transforms.ToTensor(),
            transforms.Normalize(cfg.normalize.mean, cfg.normalize.std),
        ]
    )


def render_mel_spectrogram_image(y_segment: np.ndarray, sr: int, cfg: Config) -> Image.Image:
    """Audio segment -> RGB mel-spectrogram image, rendered in-memory (no file I/O).

    Uses matplotlib's Figure/FigureCanvasAgg object API directly rather than
    the pyplot global interface: pyplot auto-selects a GUI backend (e.g. macOS's),
    which raises if a figure is created off the main thread - something any
    multi-worker/multi-threaded server deployment (and TestClient-based tests)
    can trigger. FigureCanvasAgg is a fixed, non-interactive, thread-safe
    renderer, and produces pixel-identical output to the pyplot path (verified
    against notebooks/02_preprocessing.ipynb's rendering, which trained data
    went through) - so this doesn't reopen the train/serve parity risk.
    """
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    mel_cfg = cfg.mel_spectrogram
    S = librosa.feature.melspectrogram(
        y=y_segment,
        sr=sr,
        n_fft=mel_cfg.n_fft,
        hop_length=mel_cfg.hop_length,
        n_mels=mel_cfg.n_mels,
        fmin=mel_cfg.fmin,
        fmax=mel_cfg.fmax,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    fig = Figure(figsize=(2.24, 2.24), dpi=100)
    FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.axis("off")
    ax.imshow(
        S_db,
        aspect="auto",
        origin="lower",
        cmap=cfg.image.colormap,
        vmin=mel_cfg.db_vmin,
        vmax=mel_cfg.db_vmax,
    )

    buf = io.BytesIO()
    fig.savefig(buf, format="png", pad_inches=0)
    buf.seek(0)

    resample = getattr(Image.Resampling, cfg.image.resample)
    return Image.open(buf).convert("RGB").resize(tuple(cfg.image.size), resample)


def preprocess_audio_segment(
    y_segment: np.ndarray,
    sr: int,
    cfg: Config,
    transform: Optional[transforms.Compose] = None,
) -> torch.Tensor:
    """Audio segment -> normalized (C, H, W) tensor, ready to batch and feed to a model."""
    transform = transform or build_transform(cfg)
    image = render_mel_spectrogram_image(y_segment, sr, cfg)
    return transform(image)


def infer_label_from_filename(filename: str, positive_hints: Sequence[str]) -> int:
    """1 (defective) if any hint substring appears in the filename, else 0 (healthy).

    Matches the labeling rule in notebooks/07_EfficiencyNet.ipynb's MotorMelDataset
    (hints there: 'CamChain', 'Defekt', 'Rattle').
    """
    return 1 if any(hint in filename for hint in positive_hints) else 0


class MelSpectrogramDataset(Dataset):
    """Loads pre-rendered mel-spectrogram PNGs from one or more directories.

    Consolidates the three near-duplicate inline dataset classes from
    notebooks 03 (`MotorDataset`), 04 (`SimpleDataset`), and 07
    (`MotorMelDataset`): a directory (or list of directories) of PNGs, an
    optional `label_fn(filename) -> int` for supervised use (defaults to
    dummy label 0 for the CAE's unsupervised healthy-only training), and the
    same Resize/ToTensor/Normalize transform.
    """

    def __init__(
        self,
        root_dirs: Union[PathLike, Sequence[PathLike]],
        transform: transforms.Compose,
        label_fn: Optional[Callable[[str], int]] = None,
    ):
        self.transform = transform
        self.label_fn = label_fn or (lambda _filename: 0)

        if isinstance(root_dirs, (str, Path)):
            root_dirs = [root_dirs]
        self.root_dirs = [Path(d) for d in root_dirs]

        self.samples: List[Tuple[Path, int]] = []
        for root in self.root_dirs:
            for filename in sorted(os.listdir(root)):
                if filename.lower().endswith((".png", ".jpg", ".jpeg")):
                    self.samples.append((root / filename, self.label_fn(filename)))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        path, label = self.samples[idx]
        image = Image.open(path).convert("RGB")
        return self.transform(image), label
