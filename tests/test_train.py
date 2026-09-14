import argparse
from pathlib import Path

import mlflow
import numpy as np
import pytest
import yaml
from PIL import Image

from src.config import load_config
from src.manifest import add_segment, get_connection, get_or_create_source_file, init_db, set_split
from src.train import NO_TRAINING_DATA_MESSAGE, train

REAL_CONFIG_PATH = Path(__file__).resolve().parent.parent / "config" / "config.yaml"
REAL_MODELS_DIR = Path(__file__).resolve().parent.parent / "models"


def _write_temp_config(tmp_path):
    """A config.yaml rooted entirely under tmp_path (same field values as the
    real config/config.yaml), so cfg.resolve_path(...) - and therefore
    --write-local-checkpoint - can never touch the real repo's models/ dir."""
    raw = yaml.safe_load(REAL_CONFIG_PATH.read_text())
    raw["model"]["checkpoint_path"] = "models/test_checkpoint.pth"

    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    return config_path


def _build_manifest(manifest_path, tmp_path, cfg, train_count=4):
    """train_count healthy YouTube segments (split='train') plus one Garage
    and one YouTube healthy val segment (split='val') - both domains present
    in val guarantees compute_thresholds() produces both baselines without
    relying on train.py's config-default fallback."""
    conn = get_connection(str(manifest_path))
    init_db(conn)
    source_file_id = get_or_create_source_file(conn, "data/raw/train_smoke.m4a", "YouTube")

    def _add(start, domain, split):
        png_path = tmp_path / f"seg_{domain}_{start:.2f}.png"
        Image.fromarray(np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)).save(png_path)
        segment_id = add_segment(
            conn,
            source_file_id,
            start_time_seconds=start,
            duration_seconds=cfg.audio.segment_duration,
            label="healthy",
            domain=domain,
            rendered_png_path=str(png_path),
            approved=True,
        )
        set_split(conn, segment_id, split)

    for i in range(train_count):
        _add(float(i), "YouTube", "train")
    _add(100.0, "Garage", "val")
    _add(101.0, "YouTube", "val")

    conn.close()


def _make_args(**overrides):
    defaults = dict(
        manifest_db="data/manifest.db",
        val_fraction=0.15,
        split_seed=42,
        rel_threshold_percentile=99.0,
        epochs=1,
        batch_size=2,
        lr=1e-4,
        scheduler_patience=5,
        scheduler_factor=0.5,
        tracking_uri=None,
        experiment_name="s1000-cae-anomaly-test",
        run_name=None,
        config=None,
        write_local_checkpoint=False,
    )
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


def test_train_smoke_logs_expected_mlflow_metrics_and_artifact(tmp_path):
    config_path = _write_temp_config(tmp_path)
    cfg = load_config(config_path)

    manifest_path = tmp_path / "manifest.db"
    _build_manifest(manifest_path, tmp_path, cfg)

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    args = _make_args(
        manifest_db=str(manifest_path),
        config=str(config_path),
        tracking_uri=tracking_uri,
    )

    run_id = train(args)

    client = mlflow.tracking.MlflowClient(tracking_uri=tracking_uri)
    run = client.get_run(run_id)

    metrics = run.data.metrics
    for key in (
        "train_mse_loss",
        "val_mse_loss",
        "rel_threshold",
        "healthy_median_score",
        "baseline_garage",
        "baseline_youtube",
        "final_val_mse_loss",
    ):
        assert key in metrics, f"missing metric: {key}"

    artifact_paths = {f.path for f in client.list_artifacts(run_id)}
    assert "thresholds.json" in artifact_paths

    # No local checkpoint requested - the real repo's models/ dir must stay untouched.
    checkpoint_path = cfg.resolve_path(cfg.model.checkpoint_path)
    assert not checkpoint_path.exists()


def test_train_write_local_checkpoint_writes_only_under_tmp_path(tmp_path):
    config_path = _write_temp_config(tmp_path)
    cfg = load_config(config_path)

    manifest_path = tmp_path / "manifest.db"
    _build_manifest(manifest_path, tmp_path, cfg)

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    args = _make_args(
        manifest_db=str(manifest_path),
        config=str(config_path),
        tracking_uri=tracking_uri,
        write_local_checkpoint=True,
    )

    train(args)

    checkpoint_path = cfg.resolve_path(cfg.model.checkpoint_path)
    thresholds_path = checkpoint_path.parent / "thresholds.json"

    assert checkpoint_path.exists()
    assert thresholds_path.exists()
    assert str(checkpoint_path).startswith(str(tmp_path))
    assert str(thresholds_path).startswith(str(tmp_path))
    assert not str(checkpoint_path).startswith(str(REAL_MODELS_DIR))


def test_train_raises_a_clear_error_when_the_manifest_has_no_training_segments(tmp_path):
    config_path = _write_temp_config(tmp_path)

    manifest_path = tmp_path / "empty_manifest.db"
    conn = get_connection(str(manifest_path))
    init_db(conn)
    conn.close()

    tracking_uri = f"sqlite:///{tmp_path / 'mlflow.db'}"
    args = _make_args(manifest_db=str(manifest_path), config=str(config_path), tracking_uri=tracking_uri)

    with pytest.raises((SystemExit, ValueError)) as exc_info:
        train(args)

    assert "backfill_manifest" in str(exc_info.value)
    assert NO_TRAINING_DATA_MESSAGE in str(exc_info.value)
