"""Script version of the CAE training loop in notebooks/04_CAE_Annomalydetector.ipynb,
instrumented with MLflow tracking (UPGRADE_PLAN.md §3).

The notebook trained the same architecture twice and left two undated, unversioned
checkpoints (models/s1000_CAE_MEL_annomaly.pth and ...annomalyV2.pth) with no record
of which one its own evaluation cells actually used. This script resolves that by
making every training run an immutable, named MLflow run: params, per-epoch loss/LR,
and the final model are all logged and the model is registered under the name
's1000-cae-anomaly' in the MLflow Model Registry, which is the canonical, versioned
record going forward — not a loose .pth filename. See "Run a local test" in the
project README / UPGRADE_PLAN.md for how to inspect a run before trusting it.

Usage:
    python -m src.train
    python -m src.train --epochs 2 --run-name smoke-test  # safe: no local .pth is touched
    python -m src.train --write-local-checkpoint           # also overwrites the local checkpoint mirror
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import load_config
from src.data import MelSpectrogramDataset, build_transform
from src.inference import get_device
from src.model import MotorAutoencoder

DEFAULT_DATASET_DIR = "data/processed/dataset_idle_mel"
DEFAULT_EXPERIMENT_NAME = "s1000-cae-anomaly"
REGISTERED_MODEL_NAME = "s1000-cae-anomaly"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MotorAutoencoder CAE and track the run with MLflow.")
    parser.add_argument(
        "--dataset-dir",
        default=DEFAULT_DATASET_DIR,
        help="Directory of healthy mel-spectrogram PNGs (relative to repo root unless absolute). "
        "Training is unsupervised/reconstruction-only, so this must contain healthy images only.",
    )
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--scheduler-patience", type=int, default=5)
    parser.add_argument("--scheduler-factor", type=float, default=0.5)
    parser.add_argument(
        "--tracking-uri",
        default=None,
        help="MLflow tracking URI. Must be a database-backed store (sqlite/postgres/mysql) for "
        "the Model Registry to work - a plain local file store cannot register models. "
        "Defaults to sqlite:///<repo_root>/mlflow.db.",
    )
    parser.add_argument("--experiment-name", default=DEFAULT_EXPERIMENT_NAME)
    parser.add_argument("--run-name", default=None)
    parser.add_argument("--config", default=None, help="Path to config.yaml (defaults to config/config.yaml).")
    parser.add_argument(
        "--write-local-checkpoint",
        action="store_true",
        default=False,
        help="Also write a local .pth mirror to config.yaml's model.checkpoint_path "
        "(the path src/inference.py reads by default). Off by default - every run "
        "already gets an immutable, versioned copy in the MLflow Model Registry, and "
        "this flag overwrites a real file in-place, including whatever the current "
        "production checkpoint is, with no undo. Only pass it once you've checked "
        "the run's metrics/registry entry and actually want it promoted locally.",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> str:
    cfg = load_config(args.config) if args.config else load_config()
    device = get_device()

    dataset_dir = Path(args.dataset_dir)
    if not dataset_dir.is_absolute():
        dataset_dir = cfg.resolve_path(str(dataset_dir))

    transform = build_transform(cfg)
    dataset = MelSpectrogramDataset(dataset_dir, transform=transform)
    train_loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
    print(f"Loaded {len(dataset)} healthy images from {dataset_dir}")

    model = MotorAutoencoder(bottleneck_size=cfg.model.bottleneck_size).to(device)

    # Trained ONLY on healthy data (dataset_dir above) — the same reconstruction-MSE
    # objective as the notebook. There is no held-out validation split here, matching
    # the notebook's own training loop (see UPGRADE_PLAN.md for follow-up ideas).
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=args.scheduler_patience, factor=args.scheduler_factor
    )

    tracking_uri = args.tracking_uri or f"sqlite:///{cfg.root_dir / 'mlflow.db'}"
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(args.experiment_name)

    with mlflow.start_run(run_name=args.run_name) as run:
        mlflow.log_params(
            {
                "sample_rate": cfg.audio.sample_rate,
                "segment_duration": cfg.audio.segment_duration,
                "n_mels": cfg.mel_spectrogram.n_mels,
                "n_fft": cfg.mel_spectrogram.n_fft,
                "hop_length": cfg.mel_spectrogram.hop_length,
                "bottleneck_size": cfg.model.bottleneck_size,
                "learning_rate": args.lr,
                "batch_size": args.batch_size,
                "epochs": args.epochs,
                "scheduler_patience": args.scheduler_patience,
                "scheduler_factor": args.scheduler_factor,
                "dataset_dir": str(dataset_dir),
                "dataset_size": len(dataset),
                "device": str(device),
            }
        )

        print(f"Starting training on {device} (MLflow run {run.info.run_id})...")
        epoch_loss = float("nan")
        for epoch in range(args.epochs):
            model.train()
            running_loss = 0.0

            for inputs, _ in train_loader:
                inputs = inputs.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                loss = criterion(outputs, inputs)
                loss.backward()
                optimizer.step()

                running_loss += loss.item()

            epoch_loss = running_loss / len(train_loader)
            scheduler.step(epoch_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            mlflow.log_metrics({"train_mse_loss": epoch_loss, "lr": current_lr}, step=epoch)
            print(f"Epoch {epoch + 1:02d}/{args.epochs} | MSE Loss: {epoch_loss:.6f} | LR: {current_lr:.6f}")

        mlflow.log_metric("final_train_mse_loss", epoch_loss)

        model.eval()  # important for BatchNorm before the weights are frozen into the artifact
        mlflow.pytorch.log_model(
            pytorch_model=model,
            name="model",
            registered_model_name=REGISTERED_MODEL_NAME,
            # 'pt2' (this MLflow version's default) traces the forward graph and
            # requires an input_example; 'pickle' just pickles the module, matching
            # the plain torch.save/load_state_dict flow the rest of this codebase
            # already uses (src/inference.py's load_model).
            serialization_format="pickle",
        )
        print(f"Registered model '{REGISTERED_MODEL_NAME}' (MLflow run {run.info.run_id}).")
        print(
            "This run is now the versioned record of what was trained; promote it to a "
            "stage (e.g. Production) explicitly via the MLflow UI/CLI once you've validated "
            "it - training does not auto-promote."
        )

        # Convenience local mirror at the path src/inference.py's load_model() reads by
        # default (config/config.yaml: model.checkpoint_path), so the existing inference
        # code keeps working without wiring up MLflow model-URI resolution yet (that's
        # planned for the FastAPI service - UPGRADE_PLAN.md §5). The MLflow run above,
        # not this file, is the source of truth for "which checkpoint is current." Opt-in
        # only (--write-local-checkpoint): this overwrites a real file in place, and a
        # throwaway smoke-test run must never be able to clobber the production checkpoint
        # by default.
        if args.write_local_checkpoint:
            checkpoint_path = cfg.resolve_path(cfg.model.checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            mlflow.log_param("local_checkpoint_mirror", str(checkpoint_path))
            print(f"Local checkpoint mirror written to {checkpoint_path}")
        else:
            print(
                "Skipped local checkpoint mirror (pass --write-local-checkpoint to write "
                f"{cfg.resolve_path(cfg.model.checkpoint_path)}). The model is still safely "
                "versioned in the MLflow Model Registry above."
            )

    return run.info.run_id


if __name__ == "__main__":
    train(parse_args())
