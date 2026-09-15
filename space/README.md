---
title: S1000-Diagnoser
emoji: 🏍️
colorFrom: blue
colorTo: red
sdk: static
pinned: false
---

# S1000-Diagnoser

Upload a recording of a BMW S1000 engine at idle and get a per-second anomaly
score back. Everything — audio decoding, mel-spectrogram rendering, and model
inference — runs in your browser. No audio is uploaded anywhere, and the Space
does no server-side compute at load or inference time.

## How it works

| Stage | Runs on | What it is |
| --- | --- | --- |
| Decode | WebAudio | The uploaded `.wav`/`.m4a` is decoded and resampled to the project's `sample_rate`, then averaged to mono. |
| Preprocess | Pyodide (Python in WASM) | The project's **actual** `src/data.py` and `src/audio_utils.py`, executed unmodified: the same `chunk_audio` windowing and the same `render_mel_spectrogram_image` call the training pipeline used. |
| Inference | ONNX Runtime Web (WebGPU, WASM fallback) | The CAE exported by `python -m src.export_onnx`. The anomaly signal is the reconstruction MSE per segment. |
| Scoring | Vanilla JS | A port of `src/inference.py`'s `score_segments`: normalize by the domain's healthy baseline, centered rolling mean, compare to `rel_threshold`, then `anomaly_confidence`'s sigmoid. |

The reason preprocessing runs Python-in-WASM rather than a JavaScript
reimplementation is train/serve parity. The model was trained on mel
spectrograms that went through a matplotlib render and a PNG round-trip; a
hand-written JS spectrogram would look subtly different and the model would
silently score it differently. Running the original code removes that whole
class of bug.

`scripts/check_space_parity.py` plus `dev/parity.html` measure what remains —
see "Verifying parity" below.

## Deployment

The model weights and the staged Python sources are **not** tracked in the main
repository: `space/assets/*.onnx`, `space/assets/thresholds.json` and
`space/python/` are all gitignored there. They are build artifacts, produced
fresh at deploy time. `space/python/src/data.py` is a *copy*; `src/data.py` in
the repo root is the original, and the copy is overwritten on every deploy.

Before pushing to Hugging Face:

1. `./scripts/deploy_space.sh` — exports the ONNX model, then stages
   `models/s1000_cae.onnx`, `models/thresholds.json` (if present) and the
   Python preprocessing sources into `space/`.
2. Test locally: `cd space && python -m http.server 8080`, then open
   <http://localhost:8080/>.
3. `cd space && git add -A && git commit -m "Deploy" && git push huggingface main`

`space/` is its own git repository with the Hugging Face Space as a remote —
that is what lets the artifacts be tracked there while staying gitignored in
the main repo. One-time setup (deliberately not scripted):

```bash
cd space
git init && git branch -m main
git remote add huggingface https://huggingface.co/spaces/<user>/s1000-diagnoser

# The ONNX model is ~110 MB; Hugging Face requires Git LFS above 10 MB.
git lfs install
git lfs track "assets/*.onnx"
git add .gitattributes
```

Note that the browser downloads that 110 MB on first visit, so first load is
slow even though every later visit is served from cache.

There is no CI/CD and no model versioning in the Space: a deploy always ships
whatever is in `models/` locally.

## Verifying parity

```bash
python scripts/check_space_parity.py data/raw/<some-recording>.m4a
cd space && python -m http.server 8080
open http://localhost:8080/dev/parity.html
```

The script runs the real Python pipeline over the recording and writes
reference fixtures into `space/dev/`; the page runs the browser pipeline over
bit-identical samples and reports the difference in tensors, raw MSEs, smoothed
scores and verdicts.

`dev/` is a development harness, not part of the app.

## Known limitations

- **Audio decoding differs from `librosa.load`.** Pyodide has no `soundfile`
  (needs libsndfile) and no m4a decoder, so the browser decodes with WebAudio.
  For a 44.1 kHz WAV this is a no-op; other rates are resampled by the browser
  rather than by soxr.
- **matplotlib is 3.8.4 in every Pyodide build**, against 3.10.x in the
  training environment. `imshow`'s antialiasing defaults changed between those
  versions, so a handful of rendered pixels differ by one or two levels. The
  measured effect on the anomaly score is in the parity report.
- **numba is mocked out.** It has no WebAssembly build. librosa imports it for
  `@jit` decorators that the mel path never enters, so the numbers are
  unaffected — only the speed.
- **First load is slow**: ~10 s for Pyodide plus the ~110 MB ONNX model. Both
  are cached by the browser afterwards.
- Results are a research prototype, not a diagnosis. The model has only ever
  seen healthy audio, so a high score means "unlike the healthy recordings it
  was trained on", not "confirmed fault".
