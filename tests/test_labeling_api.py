"""Tests for the labeling tool's FastAPI router (api/labeling.py, mounted on
api/tool_app.py).

Fully decoupled from tests/test_api.py: no dependency_overrides, since these
routes have zero dependency on ModelState/get_model_state. Every test
monkeypatches api.labeling._cfg to a copy of the real config with root_dir
pointed at tmp_path, so a confirmed upload's persisted WAV/PNG files never
touch the real repo's data/ directory - and monkeypatches src.manifest's
functions directly (a plain contract test) rather than depending on a real
SQLite database.
"""

import dataclasses

import pytest
from fastapi.testclient import TestClient

import api.labeling as labeling
from api.tool_app import app as tool_app
from src.audio_utils import chunk_audio


@pytest.fixture(autouse=True)
def _isolated_labeling_state(monkeypatch, cfg, tmp_path):
    monkeypatch.setattr(labeling, "_cfg", dataclasses.replace(cfg, root_dir=tmp_path))
    yield
    labeling._upload_sessions.clear()
    labeling._train_jobs.clear()


@pytest.fixture
def client():
    test_client = TestClient(tool_app)
    try:
        yield test_client
    finally:
        test_client.close()


class _FakeConnection:
    def close(self):
        pass


def _upload_sine_wave(client, sine_wave_audio_file, domain=None, filename="sine.wav"):
    data = {"domain": domain} if domain else {}
    with open(sine_wave_audio_file, "rb") as f:
        return client.post("/uploads", files={"file": (filename, f, "audio/wav")}, data=data)


def test_upload_segment_count_matches_chunk_audio(client, sine_wave_audio_file, sine_wave_signal, cfg):
    response = _upload_sine_wave(client, sine_wave_audio_file)

    assert response.status_code == 201
    body = response.json()

    signal, sr = sine_wave_signal
    expected_chunks = chunk_audio(signal, sr, cfg.audio.segment_duration, cfg.audio.step_duration)
    assert len(body["segments"]) == len(expected_chunks)
    assert len(body["segments"]) > 1


def test_upload_oversized_file_returns_413(client):
    oversized = b"0" * (labeling.MAX_UPLOAD_BYTES + 1)

    response = client.post("/uploads", files={"file": ("big.wav", oversized, "audio/wav")})

    assert response.status_code == 413


def test_upload_wrong_suffix_returns_400(client):
    response = client.post("/uploads", files={"file": ("clip.mp3", b"not audio", "audio/mpeg")})

    assert response.status_code == 400


def test_upload_uses_explicit_domain_when_provided(client, sine_wave_audio_file):
    response = _upload_sine_wave(client, sine_wave_audio_file, domain="Garage")

    assert response.json()["domain"] == "Garage"


def test_upload_infers_domain_from_filename_when_domain_absent(client, sine_wave_audio_file, cfg):
    hint = cfg.anomaly.garage_name_hints[0]

    response = _upload_sine_wave(client, sine_wave_audio_file, filename=f"{hint}_recording.wav")

    assert response.json()["domain"] == "Garage"


def test_upload_falls_back_to_default_domain_without_a_hint_in_filename(client, sine_wave_audio_file, cfg):
    response = _upload_sine_wave(client, sine_wave_audio_file, filename="unrelated_clip.wav")

    assert response.json()["domain"] == cfg.anomaly.default_domain


def test_get_upload_returns_same_shape_as_post(client, sine_wave_audio_file):
    post_response = _upload_sine_wave(client, sine_wave_audio_file)
    upload_id = post_response.json()["upload_id"]

    get_response = client.get(f"/uploads/{upload_id}")

    assert get_response.status_code == 200
    assert get_response.json() == post_response.json()


def test_get_upload_unknown_id_returns_404(client):
    response = client.get("/uploads/does-not-exist")

    assert response.status_code == 404


def test_get_spectrogram_returns_valid_png(client, sine_wave_audio_file):
    upload = _upload_sine_wave(client, sine_wave_audio_file).json()
    segment_id = upload["segments"][0]["segment_id"]

    response = client.get(f"/spectrogram/{segment_id}")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_get_spectrogram_unknown_id_returns_404(client):
    response = client.get("/spectrogram/does-not-exist:0")

    assert response.status_code == 404


def test_confirm_inserts_labeled_segments_and_skips_unlabeled(client, sine_wave_audio_file, monkeypatch):
    upload = _upload_sine_wave(client, sine_wave_audio_file).json()
    segment_ids = [s["segment_id"] for s in upload["segments"]]
    assert len(segment_ids) >= 2

    monkeypatch.setattr(labeling.manifest, "get_connection", lambda path: _FakeConnection())
    monkeypatch.setattr(labeling.manifest, "init_db", lambda conn: None)
    monkeypatch.setattr(labeling.manifest, "get_or_create_source_file", lambda *a, **kw: 1)
    next_ids = iter(range(100, 200))
    monkeypatch.setattr(labeling.manifest, "add_segment", lambda *a, **kw: next(next_ids))

    labels = [
        {"segment_id": segment_ids[0], "label": "healthy"},
        {"segment_id": segment_ids[1], "label": "defective"},
        # every other segment left out of the request entirely = skipped
    ]

    response = client.post(f"/uploads/{upload['upload_id']}/confirm", json={"labels": labels})

    assert response.status_code == 200
    body = response.json()
    assert body["inserted"] == 2
    assert body["failed"] == []
    assert len(body["manifest_ids"]) == 2


def test_confirm_one_manifest_failure_does_not_abort_the_others(client, sine_wave_audio_file, monkeypatch):
    upload = _upload_sine_wave(client, sine_wave_audio_file).json()
    segment_ids = [s["segment_id"] for s in upload["segments"]]
    assert len(segment_ids) >= 2

    monkeypatch.setattr(labeling.manifest, "get_connection", lambda path: _FakeConnection())
    monkeypatch.setattr(labeling.manifest, "init_db", lambda conn: None)
    monkeypatch.setattr(labeling.manifest, "get_or_create_source_file", lambda *a, **kw: 1)

    def flaky_add_segment(conn, source_file_id, **kwargs):
        if kwargs["start_time_seconds"] == 0.0:
            raise RuntimeError("duplicate segment")
        return 42

    monkeypatch.setattr(labeling.manifest, "add_segment", flaky_add_segment)

    labels = [
        {"segment_id": segment_ids[0], "label": "healthy"},  # start_time 0.0 -> fails
        {"segment_id": segment_ids[1], "label": "defective"},  # succeeds
    ]

    response = client.post(f"/uploads/{upload['upload_id']}/confirm", json={"labels": labels})

    assert response.status_code == 200
    body = response.json()
    assert body["inserted"] == 1
    assert len(body["failed"]) == 1
    assert body["failed"][0]["segment_id"] == segment_ids[0]
    assert body["manifest_ids"] == [42]


def test_confirm_unknown_upload_id_returns_404(client):
    response = client.post("/uploads/does-not-exist/confirm", json={"labels": []})

    assert response.status_code == 404


def test_start_training_returns_202_with_job_id(client, monkeypatch):
    monkeypatch.setattr(labeling, "_launch_training_job", lambda job, argv: None)

    response = client.post("/train", json={"epochs": 2, "run_name": "smoke"})

    assert response.status_code == 202
    body = response.json()
    assert body["status"] == "running"
    assert body["job_id"]


def test_start_training_while_job_already_running_returns_409(client, monkeypatch):
    monkeypatch.setattr(labeling, "_launch_training_job", lambda job, argv: None)

    first = client.post("/train", json={"epochs": 2})
    assert first.status_code == 202

    second = client.post("/train", json={"epochs": 2})

    assert second.status_code == 409


def test_get_training_job_returns_current_status(client, monkeypatch):
    monkeypatch.setattr(labeling, "_launch_training_job", lambda job, argv: None)

    started = client.post("/train", json={"epochs": 1}).json()
    job_id = started["job_id"]

    running_response = client.get(f"/train/{job_id}")
    assert running_response.status_code == 200
    assert running_response.json()["status"] == "running"

    labeling._train_jobs[job_id].status = "completed"
    labeling._train_jobs[job_id].run_id = "run-123"
    labeling._train_jobs[job_id].metrics = {"final_val_mse_loss": 0.01}

    completed_response = client.get(f"/train/{job_id}")
    body = completed_response.json()
    assert body["status"] == "completed"
    assert body["run_id"] == "run-123"
    assert body["metrics"] == {"final_val_mse_loss": 0.01}


def test_get_training_job_unknown_id_returns_404(client):
    response = client.get("/train/does-not-exist")

    assert response.status_code == 404
