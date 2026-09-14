"""Read-only "Diagnose" tab for the local tool app (api/tool_app.py): upload a
recording, get back the same full-file mel spectrogram the labeling tab draws
plus a time-aligned per-segment anomaly score series for it.

Nothing here writes to the manifest, to MLflow, or to the filesystem (the one
temp WAV it creates lives inside a TemporaryDirectory) - it is purely a
diagnostic view over an already-trained model.

Independent of BOTH of its neighbours, on purpose:

* api/main.py (the deployed inference service) is never imported, and this
  router holds its own separately-loaded model. The MODEL_URI -> local
  checkpoint resolution in load_model_state() below is a deliberate copy of
  that module's resolve_model(), not a shared import: the tool must never be
  able to reach into - or force a reload of - whatever the serving process
  has loaded, and api/main.py's import graph must stay free of anything this
  local-only tool drags in.
* api/labeling.py is never imported either (and never imports this). The two
  routers share a mount point and nothing else: separate session stores
  (_diag_sessions vs. its _upload_sessions), separate copies of the upload
  constants, separate spectrogram rendering. See that module's docstring for
  the same rationale in the other direction.

The consequence, accepted knowingly, is that _render_full_file_spectrogram_image
and the upload guards below duplicate their labeling.py counterparts nearly
line for line. The spectrogram copy in particular MUST stay pixel-compatible
with the labeling tab's, because the frontend maps time to x-position through
the rendered image's own width - see web/js/diagnose.js.
"""

import io
import logging
import os
import re
import tempfile
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Literal, Optional

import librosa
import numpy as np
import soundfile as sf
import torch
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile
from fastapi.responses import Response
from PIL import Image
from pydantic import BaseModel

from src import inference
from src.audio_utils import load_audio
from src.config import Config, load_config
from src.model import MotorAutoencoder
from src.thresholds import apply_thresholds, load_thresholds_from_local_file, load_thresholds_from_mlflow_run

logger = logging.getLogger("api.inference_tab")

router = APIRouter()

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB - same value as api/main.py and api/labeling.py
ALLOWED_SUFFIXES = {".m4a", ".wav"}
MAX_SPECTROGRAM_WIDTH_PX = 4096

DomainLiteral = Optional[Literal["Garage", "YouTube"]]

# Module-level: real config.yaml, loaded once, read as a bare global by every
# function below (never bound into a default argument) so a test monkeypatch
# takes effect immediately. Used for decoding and for spectrogram rendering
# only - anomaly scoring instead uses the cfg carried by ModelState, which may
# have had the loaded model's own thresholds applied over config.yaml's
# defaults (see load_model_state).
_cfg: Config = load_config()

# {upload_id: {"upload_id", "filename", "domain", "sample_rate", "audio"}}.
# No chunking happens at upload time - GET /diagnose/{id}/results does the
# scoring. Deliberately NOT api/labeling.py's _upload_sessions: a diagnose
# session can never be confirmed into the manifest, and a labeling session can
# never be scored, so an id from one is meaningless to the other.
_diag_sessions: Dict[str, dict] = {}


# ---------------------------------------------------------------------------
# Model loading (lazy, once per process - see module docstring)
# ---------------------------------------------------------------------------


@dataclass
class ModelState:
    cfg: Config
    device: torch.device
    model: MotorAutoencoder
    model_version: str
    model_run_id: Optional[str]


_model_state: Optional[ModelState] = None


def _load_from_mlflow(model_uri: str, device: torch.device) -> tuple[MotorAutoencoder, Optional[str]]:
    import mlflow.pytorch
    from mlflow.models import get_model_info

    model = mlflow.pytorch.load_model(model_uri)
    model.to(device)
    model.eval()

    run_id = None
    try:
        run_id = get_model_info(model_uri).run_id
    except Exception:
        logger.warning("Loaded model from %s but could not resolve its run_id.", model_uri)

    return model, run_id


def load_model_state() -> ModelState:
    """MODEL_URI (MLflow) first, local checkpoint.pth fallback.

    Either path also tries to load the anomaly thresholds computed for that
    specific model (src/train.py, src/thresholds.py) and applies them to the
    cfg carried on the returned state; with none found, config.yaml's own
    anomaly values are used unchanged.
    """
    cfg = load_config()
    device = inference.get_device()

    model_uri = os.environ.get("MODEL_URI")
    if model_uri:
        try:
            model, run_id = _load_from_mlflow(model_uri, device)
            logger.info("Diagnose tab loaded model from MLflow: %s (run_id=%s)", model_uri, run_id)

            thresholds = load_thresholds_from_mlflow_run(run_id) if run_id else None
            if thresholds is not None:
                cfg = apply_thresholds(cfg, thresholds)

            return ModelState(cfg=cfg, device=device, model=model, model_version=model_uri, model_run_id=run_id)
        except Exception as exc:
            logger.warning(
                "Diagnose tab could not load MODEL_URI=%s from MLflow (%s); falling back to local checkpoint.",
                model_uri,
                exc,
            )

    checkpoint_path = cfg.resolve_path(cfg.model.checkpoint_path)
    model = inference.load_model(cfg, checkpoint_path=checkpoint_path, device=device)
    logger.info("Diagnose tab loaded model from local checkpoint: %s", checkpoint_path)

    thresholds = load_thresholds_from_local_file(str(Path(checkpoint_path).parent / "thresholds.json"))
    if thresholds is not None:
        cfg = apply_thresholds(cfg, thresholds)

    return ModelState(cfg=cfg, device=device, model=model, model_version=str(checkpoint_path), model_run_id=None)


def get_model_state() -> ModelState:
    """Lazy singleton: the (slow, ~100MB) load happens on the first scoring
    request rather than at import/startup, so simply having the tool running -
    or using only its labeling tab - never pays for it.

    Declared as a FastAPI dependency on the one route that needs it, which
    also gives tests a clean override point (no model has to exist in CI).
    """
    global _model_state
    if _model_state is None:
        _model_state = load_model_state()
    return _model_state


# ---------------------------------------------------------------------------
# Request/response models
# ---------------------------------------------------------------------------


class DiagnoseUploadResponse(BaseModel):
    upload_id: str
    filename: str
    domain: str
    duration_seconds: float
    sample_rate: int


class DiagnoseSegment(BaseModel):
    index: int
    start_time: float
    end_time: float
    # The rolling-mean *smoothed*, domain-normalized score - i.e. exactly the
    # quantity src.inference.score_segments compares against `threshold`, so
    # `anomaly_score > threshold` and `is_anomalous` can never disagree on the
    # rendered graph.
    anomaly_score: float
    is_anomalous: bool
    confidence: float


class DiagnoseResultsResponse(BaseModel):
    upload_id: str
    domain: str
    duration_seconds: float
    overall_verdict: Literal["healthy", "anomaly"]
    overall_confidence: float
    segments: list[DiagnoseSegment]
    threshold: float  # drawn as the graph's reference line by web/js/diagnose.js


# ---------------------------------------------------------------------------
# Upload
# ---------------------------------------------------------------------------


def _sanitize_filename(filename: str) -> str:
    """Bare basename, non-alphanumeric characters replaced - avoids path
    traversal and keeps the name safe to echo back into the UI."""
    return re.sub(r"[^A-Za-z0-9_.-]", "_", Path(filename).name)


def _get_session(upload_id: str) -> dict:
    session = _diag_sessions.get(upload_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown upload_id '{upload_id}'.")
    return session


@router.post("/diagnose/uploads", response_model=DiagnoseUploadResponse, status_code=201)
async def create_diagnose_upload(
    file: UploadFile = File(...),
    domain: DomainLiteral = Form(None),
) -> DiagnoseUploadResponse:
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in ALLOWED_SUFFIXES:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported file type '{suffix}'. Expected one of {sorted(ALLOWED_SUFFIXES)}.",
        )

    # Read at most one byte over the cap: rejects an oversized upload without
    # ever buffering more of it than needed to know it's too big.
    contents = await file.read(MAX_UPLOAD_BYTES + 1)
    if len(contents) > MAX_UPLOAD_BYTES:
        raise HTTPException(status_code=413, detail="File too large. Maximum upload size is 50 MB.")

    original_name = Path(file.filename or f"upload{suffix}").name

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / original_name
        tmp_path.write_bytes(contents)
        try:
            audio, sr = load_audio(str(tmp_path), sample_rate=_cfg.audio.sample_rate)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not decode audio file: {exc}") from exc

    upload_id = str(uuid.uuid4())
    _diag_sessions[upload_id] = {
        "upload_id": upload_id,
        "filename": _sanitize_filename(original_name),
        # Resolved here, once, rather than left to score_audio_file's own
        # filename heuristic: the audio reaches it as a temp file whose name
        # carries no domain hint at all (see get_diagnose_results).
        "domain": domain or inference.infer_domain(original_name, _cfg),
        "sample_rate": sr,
        "audio": audio,
    }

    return DiagnoseUploadResponse(
        upload_id=upload_id,
        filename=_diag_sessions[upload_id]["filename"],
        domain=_diag_sessions[upload_id]["domain"],
        duration_seconds=len(audio) / sr,
        sample_rate=sr,
    )


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------


@router.get("/diagnose/{upload_id}/results", response_model=DiagnoseResultsResponse)
def get_diagnose_results(
    upload_id: str,
    state: ModelState = Depends(get_model_state),
) -> DiagnoseResultsResponse:
    session = _get_session(upload_id)

    # score_audio_file takes a path, and the upload's own bytes are long gone
    # (only the decoded signal is cached) - so hand it back a temp WAV.
    # subtype="FLOAT" makes that round-trip lossless: librosa returns the exact
    # same float32 samples, so scores are identical to scoring the original.
    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / f"{upload_id}.wav"
        sf.write(str(tmp_path), session["audio"], session["sample_rate"], subtype="FLOAT")
        try:
            result = inference.score_audio_file(
                tmp_path,
                state.model,
                state.cfg,
                device=state.device,
                domain=session["domain"],
            )
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not score audio file: {exc}") from exc

    if not result.segments:
        raise HTTPException(
            status_code=422,
            detail=f"Audio is shorter than the {state.cfg.audio.segment_duration}s analysis window - no segments to score.",
        )

    segment_duration = state.cfg.audio.segment_duration
    return DiagnoseResultsResponse(
        upload_id=upload_id,
        domain=result.domain,
        duration_seconds=len(session["audio"]) / session["sample_rate"],
        overall_verdict="anomaly" if result.is_anomalous else "healthy",
        overall_confidence=result.max_confidence,
        segments=[
            DiagnoseSegment(
                index=index,
                start_time=segment.start_time,
                end_time=segment.start_time + segment_duration,
                anomaly_score=segment.smoothed_score,
                is_anomalous=segment.is_anomalous,
                confidence=segment.confidence,
            )
            for index, segment in enumerate(result.segments)
        ],
        threshold=result.threshold,
    )


# ---------------------------------------------------------------------------
# Spectrogram
# ---------------------------------------------------------------------------


def _render_full_file_spectrogram_image(audio: np.ndarray, sr: int, cfg: Config) -> Image.Image:
    """Whole-file mel spectrogram, one pixel per hop.

    A deliberate copy of api/labeling.py's function of the same name (see this
    module's docstring on why the two routers share no imports). It must stay
    pixel-identical to it: both tabs render the same recording the same way,
    and web/js/diagnose.js derives its entire time->x mapping from this
    image's width.
    """
    mel_cfg = cfg.mel_spectrogram
    S = librosa.feature.melspectrogram(
        y=audio,
        sr=sr,
        n_fft=mel_cfg.n_fft,
        hop_length=mel_cfg.hop_length,
        n_mels=mel_cfg.n_mels,
        fmin=mel_cfg.fmin,
        fmax=mel_cfg.fmax,
    )
    S_db = librosa.power_to_db(S, ref=np.max)

    from matplotlib import colormaps
    from matplotlib.colors import Normalize

    norm = Normalize(vmin=mel_cfg.db_vmin, vmax=mel_cfg.db_vmax, clip=True)
    cmap = colormaps[cfg.image.colormap]
    # Flip rows so the lowest mel band renders at the bottom, matching
    # render_mel_spectrogram_image's imshow(..., origin="lower").
    rgb = (cmap(norm(S_db[::-1, :]))[:, :, :3] * 255).astype(np.uint8)
    image = Image.fromarray(rgb, mode="RGB")

    if image.width > MAX_SPECTROGRAM_WIDTH_PX:
        resample = getattr(Image.Resampling, cfg.image.resample)
        image = image.resize((MAX_SPECTROGRAM_WIDTH_PX, image.height), resample)

    return image


@router.get("/diagnose/{upload_id}/spectrogram")
def get_diagnose_spectrogram(upload_id: str) -> Response:
    session = _get_session(upload_id)

    image = _render_full_file_spectrogram_image(session["audio"], session["sample_rate"], _cfg)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")
