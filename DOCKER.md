# Docker notes

Operational notes for the inference container (see `Dockerfile`, `UPGRADE_PLAN.md` §6).

## Build

```bash
docker build -t s1000-diagnoser-api:latest .
```

Model weights are not baked in. `torch`/`torchvision` install as CPU-only wheels via
`--extra-index-url https://download.pytorch.org/whl/cpu` (confirmed available for
both linux/amd64 and linux/arm64).

## Known issue: VirtioFS can't read bind-mounted files under OneDrive

Docker Desktop's VirtioFS file sharing can `stat()` (size, mtime) files inside a
OneDrive-synced folder fine, but actual content reads through the bind mount fail
with `OSError: [Errno 5] Input/output error`. This breaks `torch.load()` on a
checkpoint mounted straight from `models/` if your repo lives under OneDrive (as
this one does). It's a Docker Desktop/OneDrive interaction, not a Dockerfile bug.

**Workaround:** stage the checkpoint to a plain local path (outside any
cloud-synced folder) before mounting it. Re-run the `cp` whenever the local
checkpoint changes.

**Real fix**, if you'd rather mount `models/` directly: Docker Desktop → Settings →
General → switch "file sharing implementation" between VirtioFS and gRPC FUSE, then
restart Docker Desktop.

## Run (staged checkpoint)

```bash
mkdir -p /tmp/s1000_model && cp models/s1000_CAE_MEL_annomaly.pth /tmp/s1000_model/
docker run -d --name s1000-diagnoser-api -p 8000:8000 \
  -v /tmp/s1000_model:/app/models:ro \
  s1000-diagnoser-api:latest
```

No `MODEL_URI` set → falls back to the local checkpoint at `config.yaml`'s
`model.checkpoint_path`, resolved inside the container as `/app/models/s1000_CAE_MEL_annomaly.pth`.

## Test

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -F "file=@data/raw/S1000R_2015_PhilipKehl.m4a;type=audio/mp4"
```

`.wav` uploads avoid a minor platform variance: `.m4a` decodes via `ffmpeg` inside
the (Linux) container vs. the macOS native AAC decoder on a local host run, giving
slightly different sample values (same order of magnitude, same anomaly verdict).
`.wav` doesn't go through `ffmpeg` at all, so results are identical across platforms.
