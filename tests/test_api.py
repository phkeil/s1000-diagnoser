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

import pytest
from fastapi.testclient import TestClient

from api.main import ModelState, app, get_model_state
from src.inference import load_model

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

    assert set(body.keys()) == {"segment_scores", "relative_scores", "is_anomalous", "model_version"}
    assert len(body["segment_scores"]) > 0
    assert len(body["segment_scores"]) == len(body["relative_scores"])
    assert all(isinstance(s, (int, float)) for s in body["segment_scores"])
    assert all(isinstance(s, (int, float)) for s in body["relative_scores"])
    assert isinstance(body["is_anomalous"], bool)
    assert body["model_version"] == FAKE_MODEL_VERSION


def test_predict_with_unsupported_file_type_returns_400(client):
    response = client.post("/predict", files={"file": ("notes.txt", b"not audio", "text/plain")})

    assert response.status_code == 400
    assert "Unsupported file type" in response.json()["detail"]
