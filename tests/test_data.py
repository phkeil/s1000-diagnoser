import numpy as np
import pytest
import torch
from PIL import Image

from src.data import (
    MelSpectrogramDataset,
    build_transform,
    infer_label_from_filename,
    preprocess_audio_segment,
    render_mel_spectrogram_image,
)


def _first_segment(sine_wave_signal, cfg):
    signal, sr = sine_wave_signal
    return signal[: int(cfg.audio.segment_duration * sr)], sr


def test_render_mel_spectrogram_image_matches_configured_size(sine_wave_signal, cfg):
    segment, sr = _first_segment(sine_wave_signal, cfg)

    image = render_mel_spectrogram_image(segment, sr, cfg)

    assert image.mode == "RGB"
    assert image.size == tuple(cfg.image.size)


def test_preprocess_audio_segment_returns_normalized_chw_tensor(sine_wave_signal, cfg):
    segment, sr = _first_segment(sine_wave_signal, cfg)

    tensor = preprocess_audio_segment(segment, sr, cfg)

    assert tensor.shape == (3, cfg.image.size[0], cfg.image.size[1])
    assert tensor.dtype == torch.float32
    # ImageNet-normalized values aren't clipped to [0, 1]; a generous bounded
    # range still catches a missing/duplicated Normalize step.
    assert tensor.min() > -10 and tensor.max() < 10


def test_preprocess_audio_segment_is_deterministic_for_a_given_transform(sine_wave_signal, cfg):
    segment, sr = _first_segment(sine_wave_signal, cfg)
    transform = build_transform(cfg)

    t1 = preprocess_audio_segment(segment, sr, cfg, transform=transform)
    t2 = preprocess_audio_segment(segment, sr, cfg, transform=transform)

    torch.testing.assert_close(t1, t2)


@pytest.mark.parametrize(
    "filename,expected",
    [
        ("bike_CamChain_1.00s.png", 1),
        ("bike_Defekt_1.00s.png", 1),
        ("bike_Rattle_1.00s.png", 1),
        ("bike_idle_1.00s.png", 0),
    ],
)
def test_infer_label_from_filename(filename, expected):
    assert infer_label_from_filename(filename, ["CamChain", "Defekt", "Rattle"]) == expected


def _write_dummy_png(path, size=(224, 224)):
    Image.fromarray(np.random.randint(0, 255, (*size, 3), dtype=np.uint8)).save(path)


def test_mel_spectrogram_dataset_loads_images_from_a_single_dir(tmp_path, cfg):
    for i in range(3):
        _write_dummy_png(tmp_path / f"img_{i}.png")

    dataset = MelSpectrogramDataset(tmp_path, transform=build_transform(cfg))

    assert len(dataset) == 3
    tensor, label = dataset[0]
    assert tensor.shape == (3, cfg.image.size[0], cfg.image.size[1])
    assert label == 0  # default dummy label, for unsupervised CAE training


def test_mel_spectrogram_dataset_applies_label_fn_across_multiple_dirs(tmp_path, cfg):
    healthy_dir = tmp_path / "healthy"
    defective_dir = tmp_path / "defective"
    healthy_dir.mkdir()
    defective_dir.mkdir()
    _write_dummy_png(healthy_dir / "idle_0.png")
    _write_dummy_png(defective_dir / "CamChain_0.png")

    dataset = MelSpectrogramDataset(
        [healthy_dir, defective_dir],
        transform=build_transform(cfg),
        label_fn=lambda filename: infer_label_from_filename(filename, ["CamChain"]),
    )

    labels = sorted(label for _, label in dataset)
    assert labels == [0, 1]


def test_mel_spectrogram_dataset_ignores_non_image_files(tmp_path, cfg):
    _write_dummy_png(tmp_path / "img.png")
    (tmp_path / "notes.txt").write_text("not an image")

    dataset = MelSpectrogramDataset(tmp_path, transform=build_transform(cfg))

    assert len(dataset) == 1
