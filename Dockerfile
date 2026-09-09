# Inference container for the CAE anomaly detector API (UPGRADE_PLAN.md §6).
# Model weights are NOT baked in - api/main.py loads MODEL_URI (an MLflow model
# URI) at startup if set, otherwise falls back to the local checkpoint at
# config/config.yaml's model.checkpoint_path, which must be bind-mounted in.

# ---- Builder: install Python deps into an isolated venv ----
FROM python:3.11-slim AS builder

WORKDIR /app

RUN python -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

COPY requirements-api.txt .
# --extra-index-url resolves torch/torchvision to the CPU-only wheels pinned in
# requirements-api.txt (see that file's own comment) instead of pulling the
# default CUDA build from PyPI, which would needlessly bloat this CPU-only image.
RUN pip install --no-cache-dir \
    --extra-index-url https://download.pytorch.org/whl/cpu \
    -r requirements-api.txt

# ---- Runtime: minimal image, no build tooling ----
FROM python:3.11-slim

# ffmpeg: librosa/audioread's Linux backend for decoding .m4a uploads (no
# CoreAudio/GStreamer available here). libsndfile1: soundfile's native backend
# for .wav.
RUN apt-get update \
    && apt-get install -y --no-install-recommends ffmpeg libsndfile1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /opt/venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH" \
    PYTHONUNBUFFERED=1

RUN useradd --create-home --shell /usr/sbin/nologin appuser
WORKDIR /app

COPY --chown=appuser:appuser src/ ./src/
COPY --chown=appuser:appuser api/ ./api/
COPY --chown=appuser:appuser config/ ./config/

# No models/ directory is copied in - config.yaml's model.checkpoint_path
# (models/s1000_CAE_MEL_annomaly.pth) resolves relative to this WORKDIR, so
# the local-fallback path only exists if you bind-mount it, e.g.:
#   docker run -v "$(pwd)/models:/app/models:ro" ...
# MLFLOW_TRACKING_URI and MODEL_URI are runtime env vars, not build args -
# the same image works across model versions without rebuilding.

USER appuser
EXPOSE 8000

CMD ["uvicorn", "api.main:app", "--host", "0.0.0.0", "--port", "8000"]
