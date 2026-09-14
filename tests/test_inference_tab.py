"""Tests for the Diagnose tab's FastAPI router (api/inference_tab.py, mounted
on api/tool_app.py).

No real model is ever loaded: src.inference.score_audio_file is monkeypatched
(the router calls it as `inference.score_audio_file`, so patching the source
module is what takes effect), and the get_model_state dependency is overridden
with a ModelState whose `.model` is an inert sentinel - the fake scorer never
dereferences it. That keeps these tests independent of the gitignored
models/s1000_CAE_MEL_annomaly.pth, exactly like tests/test_api.py.

api.inference_tab._cfg is pointed at tmp_path for the same reason
tests/test_labeling_api.py does it, even though this router writes nothing
outside a TemporaryDirectory - it keeps a future change from silently gaining
the ability to touch the real repo's data/ directory.
"""

import dataclasses
import io

import pytest
from fastapi.testclient import TestClient
from PIL import Image

import api.inference_tab as inference_tab
from api.inference_tab import ModelState, get_model_state
from api.tool_app import app as tool_app
from src import inference
from src.inference import AnomalyResult, SegmentScore

# Stands in for a MotorAutoencoder: the patched scorer below is handed this
# object and never looks at it, so nothing has to be instantiated or loaded.
UNUSED_MODEL = object()


@pytest.fixture(autouse=True)
def _isolated_diagnose_state(monkeypatch, cfg, tmp_path):
    monkeypatch.setattr(inference_tab, "_cfg", dataclasses.replace(cfg, root_dir=tmp_path))
    yield
    inference_tab._diag_sessions.clear()


@pytest.fixture
def client(cfg, device):
    fake_state = ModelState(
        cfg=cfg,
        device=device,
        model=UNUSED_MODEL,
        model_version="synthetic-test-checkpoint",
        model_run_id=None,
    )
    tool_app.dependency_overrides[get_model_state] = lambda: fake_state
    test_client = TestClient(tool_app)
    try:
        yield test_client
    finally:
        test_client.close()
        tool_app.dependency_overrides.clear()


def _fake_result(cfg, smoothed_scores):
    """An AnomalyResult shaped exactly like score_segments' own output, so the
    route is exercised against the real dataclass contract rather than a stub
    that happens to satisfy today's attribute accesses."""
    threshold = cfg.anomaly.rel_threshold
    return AnomalyResult(
        domain="YouTube",
        baseline=cfg.anomaly.domain_baselines["YouTube"],
        threshold=threshold,
        segments=[
            SegmentScore(
                start_time=index * cfg.audio.step_duration,
                raw_score=score * cfg.anomaly.domain_baselines["YouTube"],
                relative_score=score,
                smoothed_score=score,
                confidence=inference.anomaly_confidence(score, threshold, cfg.anomaly.healthy_median_score),
                is_anomalous=score > threshold,
            )
            for index, score in enumerate(smoothed_scores)
        ],
    )


@pytest.fixture
def patch_score_audio_file(monkeypatch, cfg):
    """Returns a `configure(smoothed_scores)` helper, which installs the patch
    and hands back the (initially empty) list the route's calls land in."""

    def configure(smoothed_scores):
        calls = []

        def _fake_score_audio_file(audio_path, model, cfg_arg, device=None, domain=None):
            calls.append({"audio_path": audio_path, "model": model, "domain": domain})
            result = _fake_result(cfg, smoothed_scores)
            if domain is not None:
                result.domain = domain
            return result

        monkeypatch.setattr(inference, "score_audio_file", _fake_score_audio_file)
        return calls

    return configure


def _upload(client, sine_wave_audio_file, filename="sine.wav", domain=None):
    data = {"domain": domain} if domain else {}
    with open(sine_wave_audio_file, "rb") as f:
        return client.post("/diagnose/uploads", files={"file": (filename, f, "audio/wav")}, data=data)


# ---------------------------------------------------------------------------
# POST /diagnose/uploads
# ---------------------------------------------------------------------------


def test_upload_m4a_returns_upload_id_and_duration(client, sine_wave_audio_file, cfg):
    response = _upload(client, sine_wave_audio_file, filename="idle_recording.m4a")

    assert response.status_code == 201
    body = response.json()
    assert body["upload_id"]
    assert body["filename"] == "idle_recording.m4a"
    assert body["sample_rate"] == cfg.audio.sample_rate
    assert body["duration_seconds"] == pytest.approx(3.0, abs=0.01)
    assert body["domain"] == cfg.anomaly.default_domain


def test_upload_oversized_file_returns_413(client):
    oversized = b"0" * (inference_tab.MAX_UPLOAD_BYTES + 1)

    response = client.post("/diagnose/uploads", files={"file": ("big.wav", oversized, "audio/wav")})

    assert response.status_code == 413


def test_upload_wrong_suffix_returns_400(client):
    response = client.post("/diagnose/uploads", files={"file": ("clip.mp3", b"not audio", "audio/mpeg")})

    assert response.status_code == 400


def test_upload_uses_explicit_domain_when_provided(client, sine_wave_audio_file):
    response = _upload(client, sine_wave_audio_file, domain="Garage")

    assert response.json()["domain"] == "Garage"


def test_upload_infers_domain_from_filename_when_domain_absent(client, sine_wave_audio_file, cfg):
    hint = cfg.anomaly.garage_name_hints[0]

    response = _upload(client, sine_wave_audio_file, filename=f"{hint}_recording.wav")

    assert response.json()["domain"] == "Garage"


def test_upload_does_not_touch_the_labeling_session_store(client, sine_wave_audio_file):
    """The two routers' stores are separate by design - a diagnose upload must
    never become confirmable into the manifest."""
    import api.labeling as labeling

    upload_id = _upload(client, sine_wave_audio_file).json()["upload_id"]

    assert upload_id in inference_tab._diag_sessions
    assert upload_id not in labeling._upload_sessions
    assert client.get(f"/uploads/{upload_id}").status_code == 404


# ---------------------------------------------------------------------------
# GET /diagnose/{upload_id}/results
# ---------------------------------------------------------------------------


def test_results_shape_and_segment_count_match_score_audio_file(
    client, sine_wave_audio_file, patch_score_audio_file, cfg
):
    scores = [1.1, 1.4, 0.9, 1.2]
    patch_score_audio_file(scores)
    upload_id = _upload(client, sine_wave_audio_file).json()["upload_id"]

    response = client.get(f"/diagnose/{upload_id}/results")

    assert response.status_code == 200
    body = response.json()
    assert set(body.keys()) == {
        "upload_id",
        "domain",
        "duration_seconds",
        "overall_verdict",
        "overall_confidence",
        "segments",
        "threshold",
    }
    assert body["upload_id"] == upload_id
    assert body["threshold"] == cfg.anomaly.rel_threshold
    assert len(body["segments"]) == len(scores)

    for index, segment in enumerate(body["segments"]):
        assert segment["index"] == index
        assert segment["anomaly_score"] == pytest.approx(scores[index])
        assert segment["end_time"] == pytest.approx(segment["start_time"] + cfg.audio.segment_duration)
        assert segment["is_anomalous"] is False
        assert 0.0 <= segment["confidence"] <= 1.0


def test_results_verdict_is_healthy_when_no_segment_crosses_the_threshold(
    client, sine_wave_audio_file, patch_score_audio_file
):
    patch_score_audio_file([1.0, 1.2, 0.8])
    upload_id = _upload(client, sine_wave_audio_file).json()["upload_id"]

    body = client.get(f"/diagnose/{upload_id}/results").json()

    assert body["overall_verdict"] == "healthy"
    assert body["overall_confidence"] < 0.5


def test_results_verdict_is_anomaly_when_any_segment_crosses_the_threshold(
    client, sine_wave_audio_file, patch_score_audio_file, cfg
):
    above = cfg.anomaly.rel_threshold + 1.0
    patch_score_audio_file([1.0, above, 1.1])
    upload_id = _upload(client, sine_wave_audio_file).json()["upload_id"]

    body = client.get(f"/diagnose/{upload_id}/results").json()

    assert body["overall_verdict"] == "anomaly"
    assert body["overall_confidence"] > 0.5
    assert [segment["is_anomalous"] for segment in body["segments"]] == [False, True, False]
    # The graph draws the threshold line straight from `threshold`, so an
    # is_anomalous flag must never contradict anomaly_score > threshold.
    for segment in body["segments"]:
        assert segment["is_anomalous"] == (segment["anomaly_score"] > body["threshold"])


def test_results_passes_the_sessions_resolved_domain_to_the_scorer(
    client, sine_wave_audio_file, patch_score_audio_file
):
    """The audio reaches score_audio_file as a temp file with no domain hint in
    its name, so the domain resolved at upload time has to be passed through
    explicitly."""
    calls = patch_score_audio_file([1.0])
    upload_id = _upload(client, sine_wave_audio_file, domain="Garage").json()["upload_id"]

    body = client.get(f"/diagnose/{upload_id}/results").json()

    assert len(calls) == 1
    assert calls[0]["domain"] == "Garage"
    assert body["domain"] == "Garage"


def test_results_unknown_upload_id_returns_404(client, patch_score_audio_file):
    patch_score_audio_file([1.0])

    response = client.get("/diagnose/does-not-exist/results")

    assert response.status_code == 404


def test_results_with_no_scoreable_segments_returns_422(client, sine_wave_audio_file, patch_score_audio_file):
    patch_score_audio_file([])
    upload_id = _upload(client, sine_wave_audio_file).json()["upload_id"]

    response = client.get(f"/diagnose/{upload_id}/results")

    assert response.status_code == 422


# ---------------------------------------------------------------------------
# GET /diagnose/{upload_id}/spectrogram
# ---------------------------------------------------------------------------


def test_spectrogram_returns_valid_png(client, sine_wave_audio_file):
    upload_id = _upload(client, sine_wave_audio_file).json()["upload_id"]

    response = client.get(f"/diagnose/{upload_id}/spectrogram")

    assert response.status_code == 200
    assert response.headers["content-type"] == "image/png"
    assert response.content[:8] == b"\x89PNG\r\n\x1a\n"


def test_spectrogram_matches_the_labeling_tabs_rendering(client, sine_wave_audio_file, cfg):
    """The frontend maps time to x through this image's own pixel width, and
    both tabs must render a given recording identically - so the duplicated
    renderer has to stay pixel-for-pixel in step with api/labeling.py's.

    Both are fed the session's own decoded signal, so the only thing that can
    differ here is the rendering itself.
    """
    import api.labeling as labeling

    upload_id = _upload(client, sine_wave_audio_file).json()["upload_id"]
    session = inference_tab._diag_sessions[upload_id]

    response = client.get(f"/diagnose/{upload_id}/spectrogram")
    expected = labeling._render_full_file_spectrogram_image(session["audio"], session["sample_rate"], cfg)

    rendered = Image.open(io.BytesIO(response.content))
    assert rendered.size == expected.size
    assert rendered.tobytes() == expected.tobytes()


def test_spectrogram_unknown_upload_id_returns_404(client):
    response = client.get("/diagnose/does-not-exist/spectrogram")

    assert response.status_code == 404
