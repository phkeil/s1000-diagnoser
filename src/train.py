"""Script version of the CAE training loop in notebooks/04_CAE_Annomalydetector.ipynb,
instrumented with MLflow tracking (UPGRADE_PLAN.md §3) and driven by the SQLite
manifest (src/manifest.py) instead of a hand-curated directory of PNGs.

The notebook trained the same architecture twice and left two undated, unversioned
checkpoints (models/s1000_CAE_MEL_annomaly.pth and ...annomalyV2.pth) with no record
of which one its own evaluation cells actually used. This script resolves that by
making every training run an immutable, named MLflow run: params, per-epoch loss/LR,
and the final model are all logged and the model is registered under the name
's1000-cae-anomaly' in the MLflow Model Registry, which is the canonical, versioned
record going forward — not a loose .pth filename. See "Run a local test" in the
project README / UPGRADE_PLAN.md for how to inspect a run before trusting it.

Usage:
    python -m src.backfill_manifest                        # one-time, populates data/manifest.db
    python -m src.train
    python -m src.train --epochs 2 --run-name smoke-test    # safe: no local .pth is touched
    python -m src.train --write-local-checkpoint             # also overwrites the local checkpoint mirror
"""

import argparse
from pathlib import Path

import mlflow
import mlflow.pytorch
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import load_config
from src.data import ManifestDataset, build_transform
from src.inference import get_device
from src.manifest import assign_pending_splits, get_connection, get_training_segment_objects, init_db
from src.model import MotorAutoencoder
from src.thresholds import compute_thresholds, save_thresholds_locally

DEFAULT_MANIFEST_DB = "data/manifest.db"
DEFAULT_EXPERIMENT_NAME = "s1000-cae-anomaly"
REGISTERED_MODEL_NAME = "s1000-cae-anomaly"

NO_TRAINING_DATA_MESSAGE = (
    "No training segments found. Run: python -m src.backfill_manifest "
    "(or label + approve segments through the labeling UI) before training."
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the MotorAutoencoder CAE and track the run with MLflow.")
    parser.add_argument(
        "--manifest-db",
        default=DEFAULT_MANIFEST_DB,
        help="Path to the SQLite manifest (relative to repo root unless absolute). "
        "Training is unsupervised/reconstruction-only, so only approved, healthy-labeled "
        "segments are used.",
    )
    parser.add_argument("--val-fraction", type=float, default=0.15, help="Fraction of unassigned healthy segments held out for validation (stratified per domain).")
    parser.add_argument("--split-seed", type=int, default=42, help="Seed for deterministic train/val split assignment.")
    parser.add_argument("--rel-threshold-percentile", type=float, default=99.0, help="Percentile of normalized val scores used as the anomaly threshold.")
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
        "(the path src/inference.py reads by default), plus a sibling thresholds.json. "
        "Off by default - every run already gets an immutable, versioned copy in the "
        "MLflow Model Registry, and this flag overwrites real files in-place, including "
        "whatever the current production checkpoint is, with no undo. Only pass it once "
        "you've checked the run's metrics/registry entry and actually want it promoted locally.",
    )
    return parser.parse_args()


def train(args: argparse.Namespace) -> str:
    cfg = load_config(args.config) if args.config else load_config()
    device = get_device()

    manifest_db_path = Path(args.manifest_db)
    if not manifest_db_path.is_absolute():
        manifest_db_path = cfg.resolve_path(str(manifest_db_path))

    conn = get_connection(str(manifest_db_path))
    init_db(conn)
    assign_pending_splits(conn, val_fraction=args.val_fraction, seed=args.split_seed)
    train_rows = get_training_segment_objects(conn, split="train", label="healthy")
    val_rows = get_training_segment_objects(conn, split="val", label="healthy")
    conn.close()  # rows are plain dataclasses now - the connection isn't picklable across DataLoader workers

    if not train_rows or not val_rows:
        raise SystemExit(NO_TRAINING_DATA_MESSAGE)

    transform = build_transform(cfg)
    train_dataset = ManifestDataset(train_rows, transform=transform, cfg=cfg)
    val_dataset = ManifestDataset(val_rows, transform=transform, cfg=cfg)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    # shuffle=False: compute_thresholds() maps each val score back to its
    # domain by matching batch order against val_dataset.rows' order.
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)
    print(
        f"Loaded {len(train_dataset)} healthy train segments and {len(val_dataset)} healthy "
        f"val segments from manifest at {manifest_db_path}"
    )

    model = MotorAutoencoder(bottleneck_size=cfg.model.bottleneck_size).to(device)

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
                "manifest_db": str(manifest_db_path),
                "val_fraction": args.val_fraction,
                "split_seed": args.split_seed,
                "train_size": len(train_dataset),
                "val_size": len(val_dataset),
                "device": str(device),
            }
        )

        print(f"Starting training on {device} (MLflow run {run.info.run_id})...")
        epoch_loss = float("nan")
        val_loss = float("nan")
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

            model.eval()
            val_running_loss = 0.0
            with torch.no_grad():
                for inputs, _ in val_loader:
                    inputs = inputs.to(device)
                    outputs = model(inputs)
                    val_running_loss += criterion(outputs, inputs).item()
            val_loss = val_running_loss / len(val_loader)

            scheduler.step(val_loss)
            current_lr = optimizer.param_groups[0]["lr"]

            mlflow.log_metrics({"train_mse_loss": epoch_loss, "val_mse_loss": val_loss, "lr": current_lr}, step=epoch)
            print(
                f"Epoch {epoch + 1:02d}/{args.epochs} | Train MSE: {epoch_loss:.6f} | "
                f"Val MSE: {val_loss:.6f} | LR: {current_lr:.6f}"
            )

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

        thresholds = compute_thresholds(
            model, val_loader, cfg, device, percentile=args.rel_threshold_percentile, run_id=run.info.run_id
        )
        mlflow.log_metrics(
            {
                "baseline_garage": thresholds.domain_baselines.get(
                    "Garage", cfg.anomaly.domain_baselines.get("Garage", float("nan"))
                ),
                "baseline_youtube": thresholds.domain_baselines.get(
                    "YouTube", cfg.anomaly.domain_baselines.get("YouTube", float("nan"))
                ),
                "rel_threshold": thresholds.rel_threshold,
                "healthy_median_score": thresholds.healthy_median_score,
                "final_val_mse_loss": val_loss,
            }
        )
        mlflow.log_dict(thresholds.to_dict(), "thresholds.json")
        print(
            f"Computed thresholds: rel_threshold={thresholds.rel_threshold:.4f}, "
            f"domain_baselines={thresholds.domain_baselines}"
        )

        # Convenience local mirror at the path src/inference.py's load_model() reads by
        # default (config/config.yaml: model.checkpoint_path), so the existing inference
        # code keeps working without wiring up MLflow model-URI resolution yet (that's
        # planned for the FastAPI service - UPGRADE_PLAN.md §5). The MLflow run above,
        # not this file, is the source of truth for "which checkpoint is current." Opt-in
        # only (--write-local-checkpoint): this overwrites real files in place, and a
        # throwaway smoke-test run must never be able to clobber the production checkpoint
        # by default.
        if args.write_local_checkpoint:
            checkpoint_path = cfg.resolve_path(cfg.model.checkpoint_path)
            checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), checkpoint_path)
            mlflow.log_param("local_checkpoint_mirror", str(checkpoint_path))
            print(f"Local checkpoint mirror written to {checkpoint_path}")

            thresholds_path = checkpoint_path.parent / "thresholds.json"
            save_thresholds_locally(thresholds, str(thresholds_path))
            print(f"Local thresholds mirror written to {thresholds_path}")
        else:
            print(
                "Skipped local checkpoint mirror (pass --write-local-checkpoint to write "
                f"{cfg.resolve_path(cfg.model.checkpoint_path)}). The model is still safely "
                "versioned in the MLflow Model Registry above."
            )

    return run.info.run_id


if __name__ == "__main__":
    train(parse_args())
