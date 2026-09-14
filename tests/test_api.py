"""Tests for the FastAPI inference service (UPGRADE_PLAN.md §8).

The `get_model_state` dependency is overridden with a synthetic ModelState
(built from the synthetic_checkpoint_path fixture in conftest.py), so these
tests have zero dependency on the real models/s1000_CAE_MEL_annomaly.pth
checkpoint - gitignored, 115MB, and not present in a fresh CI checkout.

The TestClient is intentionally NOT used as a context manager: entering it
(`with TestClient(app) as client`) runs the app's lifespan, which would call
resolve_model() and try to load that real checkpoint since MODEL_URI is unset
in tests. Skipping lifespan is safe here because get_model_state is fully
replaced by the override below, so the route never touches app.state at all.
"""

import dataclasses

import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import MAX_UPLOAD_BYTES, ModelState, app, get_model_state, resolve_model
from src.inference import load_model
from src.thresholds import ThresholdSet, save_thresholds_locally

FAKE_MODEL_VERSION = "synthetic-test-checkpoint"
FAKE_MODEL_RUN_ID = "test-run-id"


@pytest.fixture
def client(cfg, synthetic_checkpoint_path, device):
    model = load_model(cfg, checkpoint_path=synthetic_checkpoint_path, device=device)
    fake_state = ModelState(
        cfg=cfg,
        device=device,
        model=model,
        model_version=FAKE_MODEL_VERSION,
        model_run_id=FAKE_MODEL_RUN_ID,
    )

    app.dependency_overrides[get_model_state] = lambda: fake_state
    test_client = TestClient(app)
    try:
        yield test_client
    finally:
        test_client.close()
        app.dependency_overrides.clear()


def test_health_returns_200_and_expected_schema(client):
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "model_version": FAKE_MODEL_VERSION,
        "model_run_id": FAKE_MODEL_RUN_ID,
    }


def test_predict_with_valid_wav_returns_200_and_prediction_schema(client, sine_wave_audio_file):
    with open(sine_wave_audio_file, "rb") as f:
        response = client.post("/predict", files={"file": ("sine.wav", f, "audio/wav")})

    assert response.status_code == 200
    body = response.json()

    assert set(body.keys()) == {
        "segment_scores",
        "relative_scores",
        "confidence_scores",
        "overall_confidence",
        "is_anomalous",
        "model_version",
    }
    assert len(body["segment_scores"]) > 0
    assert len(body["segment_scores"]) == len(body["relative_scores"]) == len(body["confidence_scores"])
    assert all(isinstance(s, (int, float)) for s in body["segment_scores"])
    assert all(isinstance(s, (int, float)) for s in body["relative_scores"])
    assert all(isinstance(s, (int, float)) and 0.0 <= s <= 1.0 for s in body["confidence_scores"])
    assert isinstance(body["overall_confidence"], (int, float))
    assert 0.0 <= body["overall_confidence"] <= 1.0
    assert body["overall_confidence"] == max(body["confidence_scores"])
    assert isinstance(body["is_anomalous"], bool)
    assert body["model_version"] == FAKE_MODEL_VERSION


def test_predict_with_unsupported_file_type_returns_400(client):
    response = client.post("/predict", files={"file": ("notes.txt", b"not audio", "text/plain")})

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]


def test_predict_with_explicit_domain_overrides_filename_heuristic(client, sine_wave_audio_file, cfg):
    # "sine.wav" contains none of cfg.anomaly.garage_name_hints, so the
    # filename heuristic alone would infer "YouTube" (the default domain).
    with open(sine_wave_audio_file, "rb") as f:
        default_response = client.post("/predict", files={"file": ("sine.wav", f, "audio/wav")})
    with open(sine_wave_audio_file, "rb") as f:
        garage_response = client.post(
            "/predict",
            files={"file": ("sine.wav", f, "audio/wav")},
            data={"domain": "Garage"},
        )

    assert default_response.status_code == 200
    assert garage_response.status_code == 200

    youtube_baseline = cfg.anomaly.domain_baselines["YouTube"]
    garage_baseline = cfg.anomaly.domain_baselines["Garage"]
    expected_ratio = youtube_baseline / garage_baseline

    default_scores = default_response.json()["relative_scores"]
    garage_scores = garage_response.json()["relative_scores"]
    for default_score, garage_score in zip(default_scores, garage_scores):
        assert garage_score == pytest.approx(default_score * expected_ratio, rel=1e-6)


def test_predict_with_oversized_file_returns_413(client):
    oversized = b"0" * (MAX_UPLOAD_BYTES + 1)

    response = client.post("/predict", files={"file": ("big.wav", oversized, "audio/wav")})

    assert response.status_code == 413
    assert "Maximum upload size" in response.json()["detail"]


def _cfg_with_checkpoint(cfg, checkpoint_path):
    """A cfg whose model.checkpoint_path is an absolute tmp_path location -
    cfg.resolve_path() leaves an absolute path unchanged, so resolve_model()
    never touches the real models/ checkpoint in these tests."""
    model_cfg = dataclasses.replace(cfg.model, checkpoint_path=str(checkpoint_path))
    return dataclasses.replace(cfg, model=model_cfg)


def test_resolve_model_local_checkpoint_path_applies_sibling_thresholds(
    monkeypatch, cfg, synthetic_checkpoint_path, device
):
    monkeypatch.delenv("MODEL_URI", raising=False)

    thresholds = ThresholdSet(
        rel_threshold=cfg.anomaly.rel_threshold + 5,
        domain_baselines={"Garage": 123.0, "YouTube": 456.0},
        run_id=None,
        computed_at="2024-01-01T00:00:00+00:00",
        val_segment_counts={"Garage": 10, "YouTube": 12},
        healthy_median_score=cfg.anomaly.healthy_median_score + 1,
    )
    save_thresholds_locally(thresholds, str(synthetic_checkpoint_path.parent / "thresholds.json"))

    test_cfg = _cfg_with_checkpoint(cfg, synthetic_checkpoint_path)

    state = resolve_model(test_cfg, device)

    assert state.cfg.anomaly.rel_threshold == thresholds.rel_threshold
    assert state.cfg.anomaly.domain_baselines["Garage"] == 123.0
    assert state.cfg.anomaly.healthy_median_score == thresholds.healthy_median_score


def test_resolve_model_local_checkpoint_path_falls_back_to_config_default_without_thresholds_file(
    monkeypatch, cfg, synthetic_checkpoint_path, device
):
    monkeypatch.delenv("MODEL_URI", raising=False)
    # No thresholds.json written next to synthetic_checkpoint_path.

    test_cfg = _cfg_with_checkpoint(cfg, synthetic_checkpoint_path)

    state = resolve_model(test_cfg, device)

    assert state.cfg.anomaly.rel_threshold == cfg.anomaly.rel_threshold
    assert state.cfg.anomaly.domain_baselines == cfg.anomaly.domain_baselines
    assert state.cfg.anomaly.healthy_median_score == cfg.anomaly.healthy_median_score


def test_resolve_model_mlflow_path_applies_thresholds_when_available(monkeypatch, cfg, synthetic_model, device):
    monkeypatch.setenv("MODEL_URI", "models:/s1000-cae-anomaly/Production")
    monkeypatch.setattr(api_main, "_load_from_mlflow", lambda model_uri, device: (synthetic_model, "run-123"))

    thresholds = ThresholdSet(
        rel_threshold=cfg.anomaly.rel_threshold + 1,
        domain_baselines={"Garage": 1.0, "YouTube": 2.0},
        run_id="run-123",
        computed_at="2024-01-01T00:00:00+00:00",
        val_segment_counts={"Garage": 1, "YouTube": 1},
        healthy_median_score=cfg.anomaly.healthy_median_score + 1,
    )
    monkeypatch.setattr(api_main, "load_thresholds_from_mlflow_run", lambda run_id: thresholds)

    state = resolve_model(cfg, device)

    assert state.cfg.anomaly.rel_threshold == thresholds.rel_threshold
    assert state.cfg.anomaly.domain_baselines["Garage"] == 1.0
    assert state.cfg.anomaly.healthy_median_score == thresholds.healthy_median_score


def test_resolve_model_mlflow_path_falls_back_to_config_default_when_thresholds_unavailable(
    monkeypatch, cfg, synthetic_model, device
):
    monkeypatch.setenv("MODEL_URI", "models:/s1000-cae-anomaly/Production")
    monkeypatch.setattr(api_main, "_load_from_mlflow", lambda model_uri, device: (synthetic_model, "run-123"))
    monkeypatch.setattr(api_main, "load_thresholds_from_mlflow_run", lambda run_id: None)

    state = resolve_model(cfg, device)

    assert state.cfg.anomaly.rel_threshold == cfg.anomaly.rel_threshold
    assert state.cfg.anomaly.domain_baselines == cfg.anomaly.domain_baselines
    assert state.cfg.anomaly.healthy_median_score == cfg.anomaly.healthy_median_score
