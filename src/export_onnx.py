"""Export the trained CAE (src/model.py's MotorAutoencoder) to ONNX so it can run
in the browser via ONNX Runtime Web (see web/demo/index.html for the WebGPU
proof-of-concept that consumes the exported file).

Usage:
    python -m src.export_onnx
    python -m src.export_onnx --output models/s1000_cae.onnx

Model resolution deliberately mirrors api/main.py's resolve_model(): MODEL_URI,
if set, is loaded as an MLflow model URI; otherwise (or if that load fails for
any reason) this falls back to the local checkpoint at config.yaml's
model.checkpoint_path. The logic is duplicated rather than imported so this
offline tool doesn't pull in FastAPI and the whole serving stack just to read a
state dict.

Export runs on CPU (torch.onnx.export requires it) and is verified twice before
the file is considered good: onnx.checker.check_model() on the graph itself, and
an onnxruntime forward pass compared numerically against PyTorch's own output.

onnx/onnxruntime are dev-only dependencies (environment.yml). They are
intentionally NOT in requirements-api.txt - this script never runs inside the
Docker inference image.
"""

import argparse
import logging
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import onnx
import onnxruntime as ort
import torch

from src.config import Config, load_config
from src.inference import load_model
from src.model import MotorAutoencoder

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT_PATH = "models/s1000_cae.onnx"
OPSET_VERSION = 17
MAX_ABS_DIFF_TOLERANCE = 1e-4

NO_CHECKPOINT_MESSAGE = (
    "No model checkpoint found at {path}. Set MODEL_URI to an MLflow model URI "
    "(e.g. 'models:/s1000-cae-anomaly/Production'), or train a model and write the "
    "local mirror with: python -m src.train --write-local-checkpoint"
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the trained MotorAutoencoder CAE to ONNX.")
    parser.add_argument("--output", default=DEFAULT_OUTPUT_PATH, help="Destination .onnx path (relative to repo root unless absolute).")
    parser.add_argument("--config", default=None, help="Path to config.yaml (defaults to config/config.yaml).")
    return parser.parse_args()


def resolve_model(cfg: Config) -> Tuple[MotorAutoencoder, str]:
    """MODEL_URI (MLflow) first, local checkpoint .pth fallback - the same
    resolution order api/main.py's resolve_model() uses at service startup, so
    the exported graph is the same model the API would serve.

    Always returns a CPU, eval()-mode model: ONNX export must happen on CPU, and
    eval() matters for this architecture's BatchNorm layers.
    """
    device = torch.device("cpu")

    model_uri = os.environ.get("MODEL_URI")
    if model_uri:
        try:
            import mlflow.pytorch

            model = mlflow.pytorch.load_model(model_uri)
            model.to(device)
            model.eval()
            logger.info("Loaded model from MLflow: %s", model_uri)
            return model, model_uri
        except Exception as exc:
            logger.warning(
                "Failed to load MODEL_URI=%s from MLflow (%s); falling back to local checkpoint.",
                model_uri,
                exc,
            )

    checkpoint_path = cfg.resolve_path(cfg.model.checkpoint_path)
    if not checkpoint_path.is_file():
        raise SystemExit(NO_CHECKPOINT_MESSAGE.format(path=checkpoint_path))

    model = load_model(cfg, checkpoint_path=checkpoint_path, device=device)
    model.eval()
    logger.info("Loaded model from local checkpoint: %s", checkpoint_path)
    return model, str(checkpoint_path)


def export(cfg: Config, output_path: Path) -> None:
    model, model_source = resolve_model(cfg)

    height, width = cfg.image.size
    dummy_input = torch.randn(1, 3, height, width, dtype=torch.float32, device="cpu")

    with torch.no_grad():
        torch_output = model(dummy_input)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=OPSET_VERSION,
        input_names=["input"],
        output_names=["reconstruction"],
        dynamic_axes={"input": {0: "batch_size"}, "reconstruction": {0: "batch_size"}},
        # Torch's default (external_data=True) writes the weights beside the graph
        # as a separate <name>.onnx.data file. ONNX Runtime Web would then have to
        # fetch and wire up that second file by hand, so keep the export a single
        # self-contained .onnx that a browser can load from one URL.
        external_data=False,
    )

    onnx.checker.check_model(str(output_path))

    # Sanity check: the exported graph must reproduce PyTorch's own output for
    # the same input, not merely load without complaint.
    session = ort.InferenceSession(str(output_path), providers=["CPUExecutionProvider"])
    (onnx_output,) = session.run(["reconstruction"], {"input": dummy_input.numpy()})

    max_abs_diff = float(np.abs(onnx_output - torch_output.numpy()).max())
    assert max_abs_diff < MAX_ABS_DIFF_TOLERANCE, (
        f"ONNX output diverges from PyTorch: max |diff| = {max_abs_diff:.3e} "
        f"(tolerance {MAX_ABS_DIFF_TOLERANCE:.0e}). The exported model is not trustworthy."
    )

    size_mb = output_path.stat().st_size / (1024 * 1024)
    print(f"Exported model source : {model_source}")
    print(f"ONNX model path       : {output_path}")
    print(f"File size             : {size_mb:.2f} MB")
    print(f"Opset version         : {OPSET_VERSION}")
    print(f"Input shape           : {tuple(dummy_input.shape)} (dynamic batch_size)")
    print(f"PyTorch output shape  : {tuple(torch_output.shape)}")
    print(f"ONNX output shape     : {tuple(onnx_output.shape)}")
    print(f"Max abs diff          : {max_abs_diff:.3e} (tolerance {MAX_ABS_DIFF_TOLERANCE:.0e})")


def main(args: Optional[argparse.Namespace] = None) -> None:
    # WARNING at the root: the dynamo exporter's onnxscript passes log a wall of
    # INFO lines about graph rewrites that say nothing about whether the export
    # is good. This module's own progress lines still come through.
    logging.basicConfig(level=logging.WARNING, format="%(levelname)s: %(message)s")
    logger.setLevel(logging.INFO)
    args = args or parse_args()
    cfg = load_config(args.config) if args.config else load_config()

    output_path = Path(args.output)
    if not output_path.is_absolute():
        output_path = cfg.resolve_path(str(output_path))

    export(cfg, output_path)


if __name__ == "__main__":
    main()
