# s1000-diagnoser

Audio-based engine fault detection for BMW S1000R motorcycles, served as a FastAPI inference API with MLflow-tracked model training.

[![CI](https://github.com/phkeil/s1000-diagnoser/actions/workflows/ci.yml/badge.svg)](https://github.com/phkeil/s1000-diagnoser/actions/workflows/ci.yml)
![Python](https://img.shields.io/badge/python-3.11-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.11-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This project detects abnormal engine noise (worn cam chains, rattles, other developing faults) from a recording of the bike idling, without ever being trained on examples of what "broken" sounds like.

The approach is a **convolutional autoencoder (CAE) trained only on healthy engines**. It learns to compress a mel-spectrogram of healthy idle noise down to a small bottleneck and reconstruct it. When it's fed a spectrogram of a genuinely unhealthy engine, the reconstruction is poor — the shape of the noise doesn't match anything the model learned to compress. That reconstruction error becomes the anomaly signal: a healthy segment scores low, an unhealthy segment scores conspicuously higher, and the difference gets confirmed against a recording-environment baseline before it's flagged.

This is unsupervised anomaly detection rather than classification — useful because labeled examples of specific failure modes are scarce, but healthy idle recordings are cheap to collect. The underlying idea and data are grounded in a real ongoing project, documented on the [S1000-Forum](https://www.s1000-forum.de/viewtopic.php?f=5&t=24477).

The project started as a set of exploratory Jupyter notebooks and has since been consolidated into a shared `src/` package, a training script with MLflow experiment tracking, a FastAPI service, tests, a Dockerfile, and CI. `UPGRADE_PLAN.md` documents that migration in detail.

## Architecture

```text
  audio file (.wav / .m4a)
          │
          ▼
  load_audio()             src/audio_utils.py — decode, resample to 44.1kHz mono
          │
          ▼
  chunk_audio()             1.0s sliding windows, 0.25s step
          │
          ▼
  preprocess_audio_segment() src/data.py — mel-spectrogram (n_fft=2048, n_mels=128)
          │                              → magma-rendered 224×224 PNG → normalized tensor
          ▼
  MotorAutoencoder           src/model.py — conv encoder → 128-d bottleneck → conv decoder
          │
          ▼
  score_segments()           src/inference.py — reconstruction MSE per segment,
          │                                     normalized against a per-domain healthy
          │                                     baseline, smoothed, thresholded
          ▼
  PredictionResponse          api/schemas.py — segment_scores, relative_scores,
                                                is_anomalous, model_version
```

The same preprocessing path (`src/data.py`) is used for both training data generation and live inference, so the model never sees an input distribution shift between the two.

## Project structure

```text
s1000-diagnoser/
├── api/
│   ├── main.py              # FastAPI app: GET /health, POST /predict
│   └── schemas.py           # Pydantic request/response models
├── src/
│   ├── audio_utils.py       # load/trim/chunk audio
│   ├── data.py               # mel-spectrogram preprocessing, MelSpectrogramDataset
│   ├── model.py               # MotorAutoencoder, EfficientDiagnoser
│   ├── inference.py           # checkpoint loading, anomaly scoring
│   ├── train.py                # MLflow-tracked CAE training script
│   └── config.py               # config/config.yaml loader
├── tests/                    # pytest suite, synthetic fixtures only (no real data/models needed)
├── notebooks/                 # exploratory work; training now lives in src/train.py
├── config/
│   └── config.yaml            # single source of truth for pipeline constants
├── data/
│   ├── raw/                   # original audio files (gitignored)
│   └── processed/             # rendered mel-spectrogram datasets (gitignored)
├── reports/figures/
├── .github/workflows/ci.yml   # lint, test, docker-build
├── Dockerfile
├── DOCKER.md
├── environment.yml
├── requirements-api.txt       # pinned runtime deps for the served container
├── requirements-dev.txt       # pytest, httpx, ruff
└── UPGRADE_PLAN.md
```

## Setup

```bash
conda env create -f environment.yml
conda activate s1000-diagnoser
```

This installs the full dev/notebook environment (torch, librosa, mlflow, etc.) from `conda-forge`, with a pip fallback for anything unavailable there.

## Running the API locally

```bash
uvicorn api.main:app --reload --port 8000
```

With no `MODEL_URI` set, the app falls back to the local checkpoint at `config/config.yaml`'s `model.checkpoint_path`.

```bash
curl http://localhost:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@/path/to/your/audio.wav;type=audio/wav"
```

Accepts `.wav` or `.m4a` files. `.wav` is recommended to avoid platform-specific AAC decoder variance.

`/health` returns the loaded model version and (if tracked) its MLflow run ID. `/predict` returns per-segment reconstruction scores, their domain-normalized equivalents, and an overall `is_anomalous` flag.

## Running with Docker

```bash
docker build -t s1000-diagnoser-api:latest .
```

Model weights are not baked into the image — the container loads `MODEL_URI` (an MLflow model URI) at startup if set, otherwise falls back to a bind-mounted local checkpoint.

**OneDrive users:** Docker Desktop's VirtioFS file sharing can `stat()` files inside a OneDrive-synced folder but fails to read their contents through a bind mount (`OSError: [Errno 5] Input/output error`), which breaks `torch.load()` on a checkpoint mounted straight from `models/`. Stage the checkpoint outside any cloud-synced folder first:

```bash
mkdir -p /tmp/s1000_model && cp models/s1000_CAE_MEL_annomaly.pth /tmp/s1000_model/

docker run -d --name s1000-diagnoser-api -p 8000:8000 \
  -v /tmp/s1000_model:/app/models:ro \
  s1000-diagnoser-api:latest
```

Same requests as above work against the container:

```bash
curl http://localhost:8000/health

curl -X POST http://127.0.0.1:8000/predict \
  -F "file=@/path/to/your/audio.wav;type=audio/wav"
```

Accepts `.wav` or `.m4a` files. `.wav` is recommended to avoid platform-specific AAC decoder variance.

See `DOCKER.md` for the full VirtioFS/gRPC FUSE fix if you'd rather mount `models/` directly.

## MLflow experiment tracking

Training is a script, not a notebook cell, so every run is tracked rather than overwriting a loose `.pth` file:

```bash
python -m src.train
python -m src.train --epochs 2 --run-name smoke-test   # doesn't touch any local checkpoint
```

Each run logs its hyperparameters, per-epoch loss/learning rate, and the trained model itself to the MLflow Model Registry under `s1000-cae-anomaly`. Training never auto-promotes a model — inspect the run first:

```bash
mlflow ui --backend-store-uri sqlite:///mlflow.db
```

Once you're satisfied with a run, promote it to `Production` in the registry (via the UI, or `mlflow.MlflowClient().transition_model_version_stage(...)`). `api/main.py` reads `MODEL_URI=models:/s1000-cae-anomaly/Production` at startup when you want the served model to come from the registry instead of the local checkpoint fallback.

## Development

```bash
pip install -r requirements-dev.txt

python -m pytest tests/ -v
ruff check src/ api/ tests/
```

The test suite runs entirely against synthetic fixtures (a generated sine wave, a randomly-initialized checkpoint) — no dependency on the real, gitignored `data/` or `models/` directories, so it runs unmodified in CI.

## Known limitations

- **`.wav` vs `.m4a` decoder variance across platforms.** `.m4a` uploads decode via `ffmpeg` inside the (Linux) container but via the macOS native AAC decoder on a local host run, giving slightly different sample values (same order of magnitude, same anomaly verdict). `.wav` doesn't go through `ffmpeg` at all, so results are identical across platforms — prefer `.wav` when exact reproducibility matters.
- **Training data volume.** The CAE is trained on a small set of healthy recordings from a handful of contributors (see `garage_name_hints` in `config/config.yaml`). More data, across more bikes and recording conditions, is needed before the anomaly threshold generalizes reliably beyond the current sources.
