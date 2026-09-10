import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, Dataset

from src.manifest import ManifestSegment
from src.thresholds import (
    ThresholdSet,
    apply_thresholds,
    compute_thresholds,
    load_thresholds_from_local_file,
    load_thresholds_from_mlflow_run,
    save_thresholds_locally,
)


class _ZeroReconstructionModel(torch.nn.Module):
    """Always reconstructs zeros, so MSE(input, 0) == mean(input**2) - fully
    deterministic and controllable via each sample's constant fill value."""

    def forward(self, x):
        return torch.zeros_like(x)


class _ConstantValueDataset(Dataset):
    """A minimal stand-in for ManifestDataset: exposes the `.rows` list
    compute_thresholds() reads domains from, without touching real audio/PNGs."""

    def __init__(self, rows, values):
        self.rows = rows
        self.values = values

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        tensor = torch.full((3, 4, 4), self.values[idx], dtype=torch.float32)
        return tensor, 0


def _make_row(domain, segment_id=1):
    return ManifestSegment(
        id=segment_id,
        source_file_id=1,
        source_file_path="data/raw/example.m4a",
        start_time_seconds=0.0,
        duration_seconds=1.0,
        label="healthy",
        domain=domain,
        rendered_png_path=None,
    )


def _make_threshold_set(**overrides):
    defaults = dict(
        rel_threshold=2.5,
        domain_baselines={"Garage": 0.05, "YouTube": 0.04},
        run_id="run-abc",
        computed_at="2024-01-01T00:00:00+00:00",
        val_segment_counts={"Garage": 10, "YouTube": 12},
    )
    defaults.update(overrides)
    return ThresholdSet(**defaults)


def test_compute_thresholds_returns_correct_domains_and_percentile():
    # Garage segments all reconstruct to MSE 1.0 -> baseline 1.0.
    # YouTube segments all reconstruct to MSE 4.0 -> baseline 4.0.
    # Every normalized score is therefore exactly 1.0, at any percentile.
    rows = [_make_row("Garage", i) for i in range(3)] + [_make_row("YouTube", i) for i in range(3, 6)]
    values = [1.0, 1.0, 1.0, 2.0, 2.0, 2.0]
    val_loader = DataLoader(_ConstantValueDataset(rows, values), batch_size=2, shuffle=False)

    thresholds = compute_thresholds(
        _ZeroReconstructionModel(), val_loader, cfg=None, device=torch.device("cpu"), percentile=99.0, run_id="run-1"
    )

    assert set(thresholds.domain_baselines.keys()) == {"Garage", "YouTube"}
    assert thresholds.domain_baselines["Garage"] == pytest.approx(1.0)
    assert thresholds.domain_baselines["YouTube"] == pytest.approx(4.0)
    assert thresholds.rel_threshold == pytest.approx(1.0)
    assert thresholds.val_segment_counts == {"Garage": 3, "YouTube": 3}
    assert thresholds.run_id == "run-1"


def test_compute_thresholds_percentile_reflects_score_spread():
    # Single domain, baseline = mean(1,1,1,1,9) = 2.6 -> normalized scores
    # [1/2.6, 1/2.6, 1/2.6, 1/2.6, 9/2.6]. rel_threshold must match numpy's
    # own percentile of that exact distribution, not a hand-picked value.
    rows = [_make_row("Garage", i) for i in range(5)]
    values = [1.0, 1.0, 1.0, 1.0, 3.0]  # MSE = value**2 -> 1,1,1,1,9
    val_loader = DataLoader(_ConstantValueDataset(rows, values), batch_size=5, shuffle=False)

    thresholds = compute_thresholds(
        _ZeroReconstructionModel(), val_loader, cfg=None, device=torch.device("cpu"), percentile=90.0
    )

    baseline = sum(v**2 for v in values) / len(values)
    normalized_scores = [v**2 / baseline for v in values]
    expected = np.percentile(normalized_scores, 90.0)
    assert thresholds.rel_threshold == pytest.approx(expected)


def test_threshold_set_round_trips_through_dict_exactly():
    thresholds = _make_threshold_set()

    restored = ThresholdSet.from_dict(thresholds.to_dict())

    assert restored == thresholds


def test_save_and_load_thresholds_locally_round_trips(tmp_path):
    thresholds = _make_threshold_set()
    path = tmp_path / "nested" / "thresholds.json"

    save_thresholds_locally(thresholds, str(path))
    loaded = load_thresholds_from_local_file(str(path))

    assert path.exists()
    assert loaded == thresholds


def test_load_thresholds_from_local_file_returns_none_for_missing_file(tmp_path):
    assert load_thresholds_from_local_file(str(tmp_path / "does-not-exist.json")) is None


def test_load_thresholds_from_local_file_returns_none_for_malformed_json(tmp_path):
    path = tmp_path / "malformed.json"
    path.write_text("{not valid json")

    assert load_thresholds_from_local_file(str(path)) is None


def test_load_thresholds_from_mlflow_run_returns_none_on_any_mlflow_error(monkeypatch):
    import mlflow.artifacts

    def _raise(*args, **kwargs):
        raise RuntimeError("tracking server unreachable")

    monkeypatch.setattr(mlflow.artifacts, "load_dict", _raise)

    assert load_thresholds_from_mlflow_run("some-run-id") is None


def test_apply_thresholds_returns_new_config_without_mutating_the_original(cfg):
    original_rel_threshold = cfg.anomaly.rel_threshold
    original_baselines = dict(cfg.anomaly.domain_baselines)

    thresholds = _make_threshold_set(
        rel_threshold=original_rel_threshold + 10,
        domain_baselines={"Garage": 999.0},
    )

    result = apply_thresholds(cfg, thresholds)

    assert id(result) != id(cfg)
    assert cfg.anomaly.rel_threshold == original_rel_threshold
    assert cfg.anomaly.domain_baselines == original_baselines

    assert result.anomaly.rel_threshold == pytest.approx(original_rel_threshold + 10)
    assert result.anomaly.domain_baselines["Garage"] == 999.0
    # YouTube wasn't in the applied ThresholdSet - it must fall back to cfg's own value.
    assert result.anomaly.domain_baselines["YouTube"] == original_baselines["YouTube"]
