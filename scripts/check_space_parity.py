"""Generate the reference fixtures that space/dev/parity.html checks the
browser pipeline against.

The Space's claim is that it produces the same numbers as the server. That is
only worth anything if it is measured, so this script runs the REAL pipeline
(src.data.preprocess_audio_segment -> the trained CAE -> src.inference's
score_segments) over one recording and writes out what it got:

  space/dev/parity_audio.wav      the decoded signal, float32 WAV
  space/dev/parity_tensors.f32    the first N segment tensors, raw float32
  space/dev/parity_reference.json shapes, per-segment scores, verdicts

parity_audio.wav is written as float32 at config.yaml's sample_rate on purpose:
the browser decodes it with WebAudio, and at a matching rate and bit depth that
round-trip is lossless, so the browser and this script start from bit-identical
samples. That isolates what is actually being tested - the mel rendering and
the scoring - from how each side decodes audio (which genuinely differs: the
browser has no soxr, Pyodide has no m4a decoder).

Usage:
    python scripts/check_space_parity.py data/raw/S1000R_2015_PhilipKehl.m4a
    python scripts/check_space_parity.py <audio> --seconds 12 --tensor-segments 8
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf
import torch

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from src.audio_utils import chunk_audio, load_audio  # noqa: E402
from src.config import load_config  # noqa: E402
from src.data import build_transform, preprocess_audio_segment  # noqa: E402
from src.inference import infer_domain, load_model, reconstruction_error, score_segments  # noqa: E402
from src.thresholds import apply_thresholds, load_thresholds_from_local_file  # noqa: E402

OUTPUT_DIR = REPO_ROOT / "space" / "dev"

NO_CHECKPOINT_MESSAGE = (
    "No model checkpoint found at {path}. Train a model and write the local mirror with: "
    "python -m src.train --write-local-checkpoint"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate browser-parity reference fixtures.")
    parser.add_argument("audio", help="Audio file to use as the reference recording.")
    parser.add_argument(
        "--seconds",
        type=float,
        default=10.0,
        help="Trim the recording to this many seconds (keeps the fixtures small). 0 = whole file.",
    )
    parser.add_argument(
        "--tensor-segments",
        type=int,
        default=8,
        help="How many segment tensors to dump for the element-wise comparison.",
    )
    parser.add_argument("--domain", default=None, choices=["Garage", "YouTube"])
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config()

    checkpoint_path = cfg.resolve_path(cfg.model.checkpoint_path)
    if not checkpoint_path.is_file():
        raise SystemExit(NO_CHECKPOINT_MESSAGE.format(path=checkpoint_path))

    # Same threshold resolution the Space does (and api/main.py before it).
    thresholds = load_thresholds_from_local_file(str(checkpoint_path.parent / "thresholds.json"))
    if thresholds is not None:
        cfg = apply_thresholds(cfg, thresholds)

    audio_path = Path(args.audio)
    if not audio_path.is_file():
        raise SystemExit(f"Audio file not found: {audio_path}")

    signal, sr = load_audio(str(audio_path), sample_rate=cfg.audio.sample_rate)
    if args.seconds > 0:
        signal = signal[: int(args.seconds * sr)]

    domain = args.domain or infer_domain(str(audio_path), cfg)
    chunks = chunk_audio(signal, sr, cfg.audio.segment_duration, cfg.audio.step_duration)
    if not chunks:
        raise SystemExit(
            f"Audio is shorter than the {cfg.audio.segment_duration}s analysis window - no segments to score."
        )

    print(f"{audio_path.name}: {len(signal) / sr:.2f}s, {len(chunks)} segments, domain={domain}")

    # --- the real preprocessing + the real model ---------------------------
    device = torch.device("cpu")  # CPU so the reference is reproducible anywhere
    model = load_model(cfg, checkpoint_path=checkpoint_path, device=device)
    transform = build_transform(cfg)

    tensors, raw_scores, starts = [], [], []
    for index, (start_time, segment) in enumerate(chunks):
        tensor = preprocess_audio_segment(segment, sr, cfg, transform=transform)
        raw_scores.append(reconstruction_error(model, tensor, device))
        starts.append(start_time)
        if index < args.tensor_segments:
            tensors.append(tensor.numpy().astype(np.float32))
        if (index + 1) % 10 == 0 or index + 1 == len(chunks):
            print(f"  scored {index + 1}/{len(chunks)}", end="\r")
    print()

    result = score_segments(raw_scores, starts, domain, cfg)

    # --- fixtures ----------------------------------------------------------
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    wav_path = OUTPUT_DIR / "parity_audio.wav"
    sf.write(str(wav_path), signal, sr, subtype="FLOAT")

    tensor_path = OUTPUT_DIR / "parity_tensors.f32"
    stacked = np.stack(tensors) if tensors else np.zeros((0, 3, *cfg.image.size), dtype=np.float32)
    tensor_path.write_bytes(stacked.astype(np.float32).tobytes())

    reference = {
        "source_file": audio_path.name,
        "sample_rate": sr,
        "duration_seconds": len(signal) / sr,
        "domain": domain,
        "baseline": result.baseline,
        "threshold": result.threshold,
        "segment_duration": cfg.audio.segment_duration,
        "window_size": cfg.anomaly.window_size,
        "healthy_median_score": cfg.anomaly.healthy_median_score,
        "num_segments": len(chunks),
        "tensor_segments": len(tensors),
        "tensor_shape": [3, cfg.image.size[0], cfg.image.size[1]],
        "starts": starts,
        "raw_scores": raw_scores,
        "relative_scores": [segment.relative_score for segment in result.segments],
        "smoothed_scores": [segment.smoothed_score for segment in result.segments],
        "confidences": [segment.confidence for segment in result.segments],
        "is_anomalous": [segment.is_anomalous for segment in result.segments],
        "overall_verdict": "anomaly" if result.is_anomalous else "healthy",
        "overall_confidence": result.max_confidence,
    }
    reference_path = OUTPUT_DIR / "parity_reference.json"
    reference_path.write_text(json.dumps(reference, indent=2))

    print(f"Wrote {wav_path.relative_to(REPO_ROOT)} ({wav_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Wrote {tensor_path.relative_to(REPO_ROOT)} ({tensor_path.stat().st_size / 1e6:.1f} MB)")
    print(f"Wrote {reference_path.relative_to(REPO_ROOT)}")
    print(
        f"Reference verdict: {reference['overall_verdict']} "
        f"(confidence {reference['overall_confidence']:.3f}, "
        f"{sum(reference['is_anomalous'])}/{len(chunks)} segments anomalous)"
    )
    print("\nNow compare in the browser:")
    print("  cd space && python -m http.server 8080")
    print("  open http://localhost:8080/dev/parity.html")


if __name__ == "__main__":
    main()
