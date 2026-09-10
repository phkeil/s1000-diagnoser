"""Upload / spectrogram-preview / confirm / retrain endpoints for the local
segment-labeling tool (api/tool_app.py).

Fully decoupled from api/main.py on purpose (see UPGRADE_PLAN-style plan doc):
this router pulls in SQLite writes and `python -m src.train` subprocess
spawning, neither of which the deployed inference app has any business
carrying. ALLOWED_SUFFIXES/MAX_UPLOAD_BYTES duplicate api/main.py's constants
rather than importing them, so a labeling-tool change can never accidentally
touch the inference app's import graph.

Uploaded audio is decoded once and cached server-side in an in-memory
dict keyed by upload_id - a fresh upload is required per labeling session,
and confirm evicts the session once at least one segment is inserted. This
means the tool is single-process/single-user by design (matches running it
as a local `uvicorn --reload` dev server).
"""

import asyncio
import io
import logging
import re
import sys
import tempfile
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import soundfile as sf
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import Response

from api.labeling_schemas import (
    ConfirmRequest,
    ConfirmResponse,
    DomainLiteral,
    SegmentInfo,
    TrainJobResponse,
    TrainRequest,
    UploadResponse,
)
from src import manifest
from src.audio_utils import chunk_audio, load_audio
from src.config import Config, load_config
from src.data import render_mel_spectrogram_image
from src.inference import infer_domain

logger = logging.getLogger("api.labeling")

router = APIRouter()

MAX_UPLOAD_BYTES = 50 * 1024 * 1024  # 50 MB - same value as api/main.py
ALLOWED_SUFFIXES = {".m4a", ".wav"}

MANIFEST_DB_PATH = "data/manifest.db"
UPLOADED_AUDIO_DIR = "data/raw/uploaded"
MANUAL_PNG_DIR = "data/processed/manual"
TRAIN_EXPERIMENT_NAME = "s1000-cae-anomaly"
LOG_TAIL_LINES = 50

# Module-level: real config.yaml, loaded once. Tests monkeypatch this
# attribute (`labeling._cfg = dataclasses.replace(cfg, root_dir=tmp_path)`)
# to keep every path this module writes to under a tmp_path instead of the
# real repo - functions below always read `_cfg` as a bare module global
# (never bind it into a default argument) so a monkeypatch takes effect
# immediately for every call made afterwards.
_cfg: Config = load_config()

# {upload_id: {"upload_id", "filename", "domain", "sample_rate", "audio",
#              "chunks": [(start_time_seconds, segment_samples), ...]}}
_upload_sessions: Dict[str, dict] = {}


@dataclass
class TrainJob:
    job_id: str
    status: str  # "running" | "completed" | "failed"
    started_at: str
    log_lines: List[str] = field(default_factory=list)
    run_id: Optional[str] = None
    metrics: Optional[dict] = None
    error: Optional[str] = None


_train_jobs: Dict[str, TrainJob] = {}
_background_tasks: set = set()


def _sanitize_filename(filename: str) -> str:
    """Bare basename, non-alphanumeric characters replaced - avoids path
    traversal and keeps the persisted file path predictable."""
    name = Path(filename).name
    return re.sub(r"[^A-Za-z0-9_.-]", "_", name)


def stage_uploaded_audio(audio: np.ndarray, sr: int, filename: str, domain: str, cfg: Config) -> dict:
    """Chunk already-decoded audio and register an in-memory upload session.

    Same logic POST /uploads runs from the chunking step onward - exposed as
    a plain function so Phase 3's crawler can hand off a downloaded clip
    in-process, with no HTTP round-trip needed.
    """
    chunks = chunk_audio(audio, sr, cfg.audio.segment_duration, cfg.audio.step_duration)
    upload_id = str(uuid.uuid4())
    session = {
        "upload_id": upload_id,
        "filename": _sanitize_filename(filename),
        "domain": domain,
        "sample_rate": sr,
        "audio": audio,
        "chunks": chunks,
    }
    _upload_sessions[upload_id] = session
    return session


def _session_to_upload_response(session: dict, cfg: Config) -> UploadResponse:
    upload_id = session["upload_id"]
    segments = [
        SegmentInfo(
            segment_id=f"{upload_id}:{index}",
            index=index,
            start_time=start_time,
            end_time=start_time + cfg.audio.segment_duration,
        )
        for index, (start_time, _samples) in enumerate(session["chunks"])
    ]
    return UploadResponse(
        upload_id=upload_id,
        filename=session["filename"],
        domain=session["domain"],
        sample_rate=session["sample_rate"],
        segment_duration_seconds=cfg.audio.segment_duration,
        step_duration_seconds=cfg.audio.step_duration,
        segments=segments,
    )


def _resolve_segment(segment_id: str) -> Tuple[dict, int]:
    upload_id, sep, index_str = segment_id.partition(":")
    session = _upload_sessions.get(upload_id) if sep else None
    if session is None or not index_str.isdigit():
        raise HTTPException(status_code=404, detail=f"Unknown segment_id '{segment_id}'.")

    index = int(index_str)
    if index < 0 or index >= len(session["chunks"]):
        raise HTTPException(status_code=404, detail=f"Unknown segment_id '{segment_id}'.")
    return session, index


@router.post("/uploads", response_model=UploadResponse, status_code=201)
async def create_upload(file: UploadFile = File(...), domain: DomainLiteral = Form(None)) -> UploadResponse:
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
        audio, sr = load_audio(str(tmp_path), sample_rate=_cfg.audio.sample_rate)

    resolved_domain = domain or infer_domain(original_name, _cfg)
    session = stage_uploaded_audio(audio, sr, original_name, resolved_domain, _cfg)
    return _session_to_upload_response(session, _cfg)


@router.get("/uploads/{upload_id}", response_model=UploadResponse)
def get_upload(upload_id: str) -> UploadResponse:
    session = _upload_sessions.get(upload_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown upload_id '{upload_id}'.")
    return _session_to_upload_response(session, _cfg)


@router.get("/spectrogram/{segment_id}")
def get_spectrogram(segment_id: str) -> Response:
    session, index = _resolve_segment(segment_id)
    _start_time, samples = session["chunks"][index]

    image = render_mel_spectrogram_image(samples, session["sample_rate"], _cfg)
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")


def _persist_source_file(conn, session: dict) -> int:
    """Runs once per upload, on the first confirmed segment: writes the
    decoded audio out as a WAV (the canonical form both a manual upload and
    a future crawler download produce - see stage_uploaded_audio) and
    registers it as a source_files row."""
    stem = Path(session["filename"]).stem
    relative_path = f"{UPLOADED_AUDIO_DIR}/{session['upload_id']}_{stem}.wav"
    out_path = _cfg.resolve_path(relative_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), session["audio"], session["sample_rate"])

    return manifest.get_or_create_source_file(
        conn,
        relative_path,
        session["domain"],
        sample_rate=session["sample_rate"],
        duration_seconds=len(session["audio"]) / session["sample_rate"],
    )


def _render_and_save_png(samples: np.ndarray, session: dict, label: str, start_time: float) -> str:
    image = render_mel_spectrogram_image(samples, session["sample_rate"], _cfg)

    stem = Path(session["filename"]).stem
    relative_path = f"{MANUAL_PNG_DIR}/{session['domain']}/{label}/{stem}_{start_time:.2f}s.png"
    out_path = _cfg.resolve_path(relative_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    image.save(out_path, format="PNG")
    return relative_path


@router.post("/uploads/{upload_id}/confirm", response_model=ConfirmResponse)
def confirm_upload(upload_id: str, request: ConfirmRequest) -> ConfirmResponse:
    session = _upload_sessions.get(upload_id)
    if session is None:
        raise HTTPException(status_code=404, detail=f"Unknown upload_id '{upload_id}'.")

    conn = manifest.get_connection(str(_cfg.resolve_path(MANIFEST_DB_PATH)))
    manifest.init_db(conn)

    inserted = 0
    manifest_ids: List[int] = []
    failed: List[dict] = []
    source_file_id: Optional[int] = None

    try:
        for label_entry in request.labels:
            try:
                _, sep, index_str = label_entry.segment_id.partition(":")
                if not sep or not index_str.isdigit():
                    raise ValueError(f"Malformed segment_id '{label_entry.segment_id}'.")
                index = int(index_str)
                if index < 0 or index >= len(session["chunks"]):
                    raise ValueError(f"Segment index {index} out of range for upload '{upload_id}'.")

                if source_file_id is None:
                    source_file_id = _persist_source_file(conn, session)

                start_time, samples = session["chunks"][index]
                png_path = _render_and_save_png(samples, session, label_entry.label, start_time)
                segment_id = manifest.add_segment(
                    conn,
                    source_file_id,
                    start_time_seconds=start_time,
                    duration_seconds=_cfg.audio.segment_duration,
                    label=label_entry.label,
                    domain=session["domain"],
                    rendered_png_path=png_path,
                    approved=True,
                )
                manifest_ids.append(segment_id)
                inserted += 1
            except Exception as exc:
                failed.append({"segment_id": label_entry.segment_id, "error": str(exc)})
    finally:
        conn.close()

    if inserted > 0:
        _upload_sessions.pop(upload_id, None)

    return ConfirmResponse(inserted=inserted, failed=failed, manifest_ids=manifest_ids)


def _current_running_job() -> Optional[TrainJob]:
    return next((job for job in _train_jobs.values() if job.status == "running"), None)


def _job_to_response(job: TrainJob) -> TrainJobResponse:
    return TrainJobResponse(
        job_id=job.job_id,
        status=job.status,
        started_at=job.started_at,
        log_tail="\n".join(job.log_lines),
        run_id=job.run_id,
        metrics=job.metrics,
        error=job.error,
    )


def _find_latest_mlflow_run(started_at: str) -> Tuple[Optional[str], Optional[dict]]:
    """Best-effort: the job succeeded either way - the trained model is
    already registered in MLflow regardless of whether this lookup works, so
    any failure here is logged and swallowed rather than raised."""
    try:
        from mlflow.tracking import MlflowClient

        tracking_uri = f"sqlite:///{_cfg.root_dir / 'mlflow.db'}"
        client = MlflowClient(tracking_uri=tracking_uri)
        experiment = client.get_experiment_by_name(TRAIN_EXPERIMENT_NAME)
        if experiment is None:
            return None, None

        started_at_ms = int(datetime.fromisoformat(started_at).timestamp() * 1000)
        runs = client.search_runs(
            [experiment.experiment_id],
            filter_string=f"attributes.start_time >= {started_at_ms}",
            order_by=["attributes.start_time DESC"],
            max_results=1,
        )
        if not runs:
            return None, None
        run = runs[0]
        return run.info.run_id, dict(run.data.metrics)
    except Exception:
        logger.warning("Could not resolve the MLflow run for a completed training job.", exc_info=True)
        return None, None


async def _run_training_job(job: TrainJob, argv: List[str]) -> None:
    process = await asyncio.create_subprocess_exec(
        *argv,
        cwd=str(_cfg.root_dir),
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.STDOUT,
    )
    assert process.stdout is not None
    while True:
        line = await process.stdout.readline()
        if not line:
            break
        job.log_lines.append(line.decode(errors="replace").rstrip())
        job.log_lines[:] = job.log_lines[-LOG_TAIL_LINES:]

    return_code = await process.wait()
    if return_code == 0:
        job.status = "completed"
        job.run_id, job.metrics = _find_latest_mlflow_run(job.started_at)
    else:
        job.status = "failed"
        job.error = "\n".join(job.log_lines[-LOG_TAIL_LINES:])


def _launch_training_job(job: TrainJob, argv: List[str]) -> None:
    """Separated from _run_training_job so tests can monkeypatch this one
    no-op stub instead of dealing with a real asyncio.Task/subprocess."""
    task = asyncio.create_task(_run_training_job(job, argv))
    _background_tasks.add(task)
    task.add_done_callback(_background_tasks.discard)


@router.post("/train", response_model=TrainJobResponse, status_code=202)
async def start_training(request: TrainRequest) -> TrainJobResponse:
    if _current_running_job() is not None:
        raise HTTPException(status_code=409, detail="A training job is already running.")

    job_id = str(uuid.uuid4())
    started_at = datetime.now(timezone.utc).isoformat()
    job = TrainJob(job_id=job_id, status="running", started_at=started_at)
    _train_jobs[job_id] = job

    manifest_db_path = str(_cfg.resolve_path(MANIFEST_DB_PATH))
    argv = [sys.executable, "-m", "src.train", "--manifest-db", manifest_db_path, "--epochs", str(request.epochs)]
    if request.run_name:
        argv += ["--run-name", request.run_name]
    # --write-local-checkpoint is deliberately never passed - matches
    # src/train.py's own safe-by-default philosophy.

    _launch_training_job(job, argv)
    return _job_to_response(job)


@router.get("/train/{job_id}", response_model=TrainJobResponse)
def get_training_job(job_id: str) -> TrainJobResponse:
    job = _train_jobs.get(job_id)
    if job is None:
        raise HTTPException(status_code=404, detail=f"Unknown job_id '{job_id}'.")
    return _job_to_response(job)
