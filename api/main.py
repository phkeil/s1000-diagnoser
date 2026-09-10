"""FastAPI inference service for the CAE anomaly detector (UPGRADE_PLAN.md §5).

POST /predict runs the same load -> chunk -> preprocess -> score pipeline as
src.inference.score_audio_file (which this endpoint calls directly); GET
/health reports which model is currently loaded.

POST /predict accepts an optional `domain` form field ('Garage' or 'YouTube')
that, when supplied, overrides the filename-based heuristic in
src.inference.infer_domain. Uploads over MAX_UPLOAD_BYTES are rejected with a
413 before any audio processing happens.

Model loading (once, at startup): MODEL_URI, if set, is treated as an MLflow
model URI (e.g. "models:/s1000-cae-anomaly/Production") and loaded via
mlflow.pytorch.load_model. If MODEL_URI is unset, or loading from it fails for
any reason, this falls back to the local checkpoint at config.yaml's
model.checkpoint_path via src.inference.load_model - which is a plain
torch.load/state_dict call and needs no MLflow tracking server at all. Local
dev and CI, without any MLflow deployment, always land on this fallback.
"""

import logging
import os
import tempfile
from contextlib import asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import torch
from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile

from api.schemas import DomainLiteral, HealthResponse, PredictionResponse
from src.config import Config, load_config
from src.inference import get_device, load_model, score_audio_file
from src.model import MotorAutoencoder

logger = logging.getLogger("api")

ALLOWED_SUFFIXES = {".wav", ".m4a"}
MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB


@dataclass
class ModelState:
    cfg: Config
    device: torch.device
    model: MotorAutoencoder
    model_version: str
    model_run_id: Optional[str]


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


def resolve_model(cfg: Config, device: torch.device) -> ModelState:
    """MODEL_URI (MLflow) first, local checkpoint.pth fallback - see module docstring."""
    model_uri = os.environ.get("MODEL_URI")
    if model_uri:
        try:
            model, run_id = _load_from_mlflow(model_uri, device)
            logger.info("Loaded model from MLflow: %s (run_id=%s)", model_uri, run_id)
            return ModelState(cfg=cfg, device=device, model=model, model_version=model_uri, model_run_id=run_id)
        except Exception as exc:
            logger.warning(
                "Failed to load MODEL_URI=%s from MLflow (%s); falling back to local checkpoint.",
                model_uri,
                exc,
            )

    checkpoint_path = cfg.resolve_path(cfg.model.checkpoint_path)
    model = load_model(cfg, checkpoint_path=checkpoint_path, device=device)
    logger.info("Loaded model from local checkpoint: %s", checkpoint_path)
    return ModelState(cfg=cfg, device=device, model=model, model_version=str(checkpoint_path), model_run_id=None)


@asynccontextmanager
async def lifespan(app: FastAPI):
    cfg = load_config()
    device = get_device()
    app.state.model_state = resolve_model(cfg, device)
    yield


app = FastAPI(title="s1000-diagnoser inference API", lifespan=lifespan)


def get_model_state(request: Request) -> ModelState:
    return request.app.state.model_state


@app.get("/health", response_model=HealthResponse)
def health(state: ModelState = Depends(get_model_state)) -> HealthResponse:
    return HealthResponse(status="ok", model_version=state.model_version, model_run_id=state.model_run_id)


@app.post("/predict", response_model=PredictionResponse)
async def predict(
    file: UploadFile = File(...),
    domain: DomainLiteral = Form(None),
    state: ModelState = Depends(get_model_state),
) -> PredictionResponse:
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

    # Written under the *original* filename (sanitized to a bare basename, to
    # avoid path traversal) rather than a random temp name, because
    # score_audio_file()/infer_domain() read domain hints (e.g. 'Philip',
    # 'Andre') from the filename to pick the right healthy baseline - unless
    # domain is supplied explicitly above, which takes precedence.
    original_name = Path(file.filename or f"upload{suffix}").name

    with tempfile.TemporaryDirectory() as tmp_dir:
        tmp_path = Path(tmp_dir) / original_name
        tmp_path.write_bytes(contents)

        try:
            result = score_audio_file(tmp_path, state.model, state.cfg, device=state.device, domain=domain)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Could not process audio file: {exc}") from exc

    if not result.segments:
        raise HTTPException(
            status_code=422,
            detail=f"Audio is shorter than the {state.cfg.audio.segment_duration}s analysis window - no segments to score.",
        )

    return PredictionResponse(
        segment_scores=[s.raw_score for s in result.segments],
        relative_scores=[s.relative_score for s in result.segments],
        is_anomalous=result.is_anomalous,
        model_version=state.model_version,
    )
