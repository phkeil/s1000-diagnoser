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
import io

import librosa
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

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


def test_upload_captures_recording_metadata_and_auto_derives_codec(client, sine_wave_audio_file):
    with open(sine_wave_audio_file, "rb") as f:
        response = client.post(
            "/uploads",
            files={"file": ("sine.wav", f, "audio/wav")},
            data={
                "contributor": "Philip",
                "exhaust_system": "Akrapovic",
                "model_year": "2015",
                "kilometers_on_bike": "42000.5",
                "oil_type": "10W-40 full synthetic",
            },
        )

    assert response.status_code == 201
    metadata = response.json()["source_metadata"]
    assert metadata["contributor"] == "Philip"
    assert metadata["exhaust_system"] == "Akrapovic"
    assert metadata["model_year"] == 2015
    assert metadata["kilometers_on_bike"] == 42000.5
    assert metadata["oil_type"] == "10W-40 full synthetic"
    assert metadata["original_codec"] == "wav"  # derived server-side, never client-supplied
    assert metadata["recording_device"] is None
    assert metadata["known_issues"] is None
    assert metadata["notes"] is None


def test_upload_without_recording_metadata_defaults_to_all_none_except_codec(client, sine_wave_audio_file):
    response = _upload_sine_wave(client, sine_wave_audio_file)

    metadata = response.json()["source_metadata"]
    assert metadata["original_codec"] == "wav"
    for field in ("contributor", "recording_device", "exhaust_system", "model_year", "known_issues", "notes"):
        assert metadata[field] is None


def test_confirm_passes_recording_metadata_through_to_get_or_create_source_file(
    client, sine_wave_audio_file, monkeypatch
):
    with open(sine_wave_audio_file, "rb") as f:
        upload = client.post(
            "/uploads",
            files={"file": ("sine.wav", f, "audio/wav")},
            data={"contributor": "Philip", "known_issues": "faint rattle at idle"},
        ).json()

    monkeypatch.setattr(labeling.manifest, "get_connection", lambda path: _FakeConnection())
    monkeypatch.setattr(labeling.manifest, "init_db", lambda conn: None)
    monkeypatch.setattr(labeling.manifest, "add_segment", lambda *a, **kw: 1)

    captured = {}

    def fake_get_or_create_source_file(conn, file_path, domain, **kwargs):
        captured.update(kwargs)
        return 1

    monkeypatch.setattr(labeling.manifest, "get_or_create_source_file", fake_get_or_create_source_file)

    regions = [{"start_time": 0.0, "end_time": 1.0, "label": "healthy"}]
    client.post(f"/uploads/{upload['upload_id']}/confirm", json={"regions": regions})

    metadata = captured["metadata"]
    assert metadata.contributor == "Philip"
    assert metadata.known_issues == "faint rattle at idle"
    assert metadata.original_codec == "wav"


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


def test_get_full_spectrogram_returns_valid_png(client, sine_wave_audio_file):
    upload = _upload_sine_wave(client, sine_wave_audio_file).json()

    response = client.get(f"/uploads/{upload['upload_id']}/spectrogram")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_get_full_spectrogram_width_is_proportional_to_duration(client, sine_wave_audio_file, sine_wave_signal, cfg):
    """1 pixel ~= 1 hop for a file well under the 4096px cap - the 3s sine
    fixture is nowhere near that, so no downsampling should have kicked in."""
    upload = _upload_sine_wave(client, sine_wave_audio_file).json()

    response = client.get(f"/uploads/{upload['upload_id']}/spectrogram")

    image = Image.open(io.BytesIO(response.content))
    signal, sr = sine_wave_signal
    mel_cfg = cfg.mel_spectrogram
    expected_hops = librosa.feature.melspectrogram(
        y=signal, sr=sr, n_fft=mel_cfg.n_fft, hop_length=mel_cfg.hop_length, n_mels=mel_cfg.n_mels
    ).shape[1]
    assert image.width == expected_hops
    assert image.height == cfg.mel_spectrogram.n_mels


def test_get_full_spectrogram_unknown_id_returns_404(client):
    response = client.get("/uploads/does-not-exist/spectrogram")

    assert response.status_code == 404


def test_confirm_inserts_chunked_segments_and_skips_skipped_regions(client, sine_wave_audio_file, monkeypatch, cfg):
    upload = _upload_sine_wave(client, sine_wave_audio_file).json()

    monkeypatch.setattr(labeling.manifest, "get_connection", lambda path: _FakeConnection())
    monkeypatch.setattr(labeling.manifest, "init_db", lambda conn: None)
    monkeypatch.setattr(labeling.manifest, "get_or_create_source_file", lambda *a, **kw: 1)
    next_ids = iter(range(100, 200))
    monkeypatch.setattr(labeling.manifest, "add_segment", lambda *a, **kw: next(next_ids))

    regions = [
        {"start_time": 0.0, "end_time": 1.5, "label": "healthy"},
        {"start_time": 1.5, "end_time": 3.0, "label": "defective"},
        {"start_time": 2.0, "end_time": 2.5, "label": "skip"},  # dropped, never chunked
    ]

    response = client.post(f"/uploads/{upload['upload_id']}/confirm", json={"regions": regions})

    assert response.status_code == 200
    body = response.json()

    expected_chunks = len(
        chunk_audio(
            np.zeros(int(1.5 * cfg.audio.sample_rate)), cfg.audio.sample_rate, cfg.audio.segment_duration, cfg.audio.step_duration
        )
    )
    assert body["inserted"] == expected_chunks * 2  # two labeled 1.5s regions, same length
    assert body["failed"] == []
    assert len(body["manifest_ids"]) == body["inserted"]
    assert body["skipped_regions"] == 0


def test_confirm_region_shorter_than_one_window_yields_zero_chunks(client, sine_wave_audio_file, monkeypatch):
    upload = _upload_sine_wave(client, sine_wave_audio_file).json()

    monkeypatch.setattr(labeling.manifest, "get_connection", lambda path: _FakeConnection())
    monkeypatch.setattr(labeling.manifest, "init_db", lambda conn: None)
    monkeypatch.setattr(labeling.manifest, "get_or_create_source_file", lambda *a, **kw: 1)
    monkeypatch.setattr(labeling.manifest, "add_segment", lambda *a, **kw: 1)

    regions = [{"start_time": 0.0, "end_time": 0.5, "label": "healthy"}]  # under the 1.0s window

    response = client.post(f"/uploads/{upload['upload_id']}/confirm", json={"regions": regions})

    assert response.status_code == 200
    body = response.json()
    assert body["inserted"] == 0
    assert body["failed"] == []
    assert body["manifest_ids"] == []
    assert body["skipped_regions"] == 1


def test_confirm_one_manifest_failure_does_not_abort_the_others(client, sine_wave_audio_file, monkeypatch):
    upload = _upload_sine_wave(client, sine_wave_audio_file).json()

    monkeypatch.setattr(labeling.manifest, "get_connection", lambda path: _FakeConnection())
    monkeypatch.setattr(labeling.manifest, "init_db", lambda conn: None)
    monkeypatch.setattr(labeling.manifest, "get_or_create_source_file", lambda *a, **kw: 1)

    def flaky_add_segment(conn, source_file_id, **kwargs):
        if kwargs["start_time_seconds"] == 0.0:
            raise RuntimeError("duplicate segment")
        return 42

    monkeypatch.setattr(labeling.manifest, "add_segment", flaky_add_segment)

    # A single region spanning two chunk windows (0.0s and 0.25s starts): the
    # first chunk's insert fails, the second still succeeds.
    regions = [{"start_time": 0.0, "end_time": 1.25, "label": "healthy"}]

    response = client.post(f"/uploads/{upload['upload_id']}/confirm", json={"regions": regions})

    assert response.status_code == 200
    body = response.json()
    assert body["inserted"] == 1
    assert len(body["failed"]) == 1
    assert body["failed"][0]["start_time"] == 0.0
    assert body["manifest_ids"] == [42]


def test_confirm_unknown_upload_id_returns_404(client):
    response = client.post("/uploads/does-not-exist/confirm", json={"regions": []})

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
