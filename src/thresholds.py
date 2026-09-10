"""Post-training threshold computation and persistence for the CAE anomaly
detector (src/inference.py's score_segments consumes the results).

Kept separate from src/config.py so that module stays a dumb YAML loader with
no MLflow awareness - mlflow itself is only imported lazily, inside the two
functions that actually need it, so this module stays importable in test
environments where no MLflow tracking server is running.
"""

import json
import logging
from dataclasses import asdict, dataclass, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
from torch.utils.data import DataLoader

from src.config import Config

logger = logging.getLogger(__name__)


@dataclass
class ThresholdSet:
    """A trained model's val-set-derived anomaly thresholds - the traceable
    replacement for config.yaml's hand-computed anomaly.rel_threshold /
    anomaly.domain_baselines (see src/train.py, api/main.py's resolve_model)."""

    rel_threshold: float
    domain_baselines: Dict[str, float]
    run_id: Optional[str]
    computed_at: str
    val_segment_counts: Dict[str, int]

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "ThresholdSet":
        return cls(
            rel_threshold=data["rel_threshold"],
            domain_baselines=dict(data["domain_baselines"]),
            run_id=data.get("run_id"),
            computed_at=data["computed_at"],
            val_segment_counts=dict(data["val_segment_counts"]),
        )


@torch.no_grad()
def compute_thresholds(
    model: torch.nn.Module,
    val_loader: DataLoader,
    cfg: Config,
    device: torch.device,
    percentile: float = 99.0,
    run_id: Optional[str] = None,
) -> ThresholdSet:
    """Mirrors score_segments()'s own normalization pipeline: per-domain mean
    raw reconstruction MSE over the val set becomes that domain's baseline,
    each segment's score is normalized against its own domain's baseline, and
    rel_threshold is the `percentile`-th percentile of the pooled, normalized
    scores.

    val_loader.dataset must expose a `.rows` list of ManifestSegment-shaped
    objects (a `.domain` attribute per row, in dataset-index order) and
    val_loader itself must not shuffle (shuffle=False) - segment-to-domain
    mapping here relies on batches arriving in the same order as `.rows`.
    """
    rows = val_loader.dataset.rows

    model.eval()
    raw_scores_by_domain: Dict[str, List[float]] = {}

    position = 0
    for inputs, _ in val_loader:
        inputs = inputs.to(device)
        outputs = model(inputs)
        batch_mse = torch.mean((outputs - inputs) ** 2, dim=(1, 2, 3))
        for mse in batch_mse.tolist():
            domain = rows[position].domain
            raw_scores_by_domain.setdefault(domain, []).append(mse)
            position += 1

    domain_baselines = {domain: float(np.mean(scores)) for domain, scores in raw_scores_by_domain.items()}
    val_segment_counts = {domain: len(scores) for domain, scores in raw_scores_by_domain.items()}

    normalized_scores = [
        score / domain_baselines[domain] for domain, scores in raw_scores_by_domain.items() for score in scores
    ]
    rel_threshold = float(np.percentile(normalized_scores, percentile))

    return ThresholdSet(
        rel_threshold=rel_threshold,
        domain_baselines=domain_baselines,
        run_id=run_id,
        computed_at=datetime.now(timezone.utc).isoformat(),
        val_segment_counts=val_segment_counts,
    )


def save_thresholds_locally(thresholds: ThresholdSet, path: str) -> None:
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(thresholds.to_dict(), indent=2))


def load_thresholds_from_local_file(path: str) -> Optional[ThresholdSet]:
    """Best-effort load - returns None (never raises) if the file is missing
    or malformed, so a pre-Phase-1 checkpoint with no sibling thresholds.json
    falls through to config.yaml's hardcoded defaults instead of crashing."""
    try:
        data = json.loads(Path(path).read_text())
        return ThresholdSet.from_dict(data)
    except (OSError, json.JSONDecodeError, KeyError, TypeError, ValueError):
        return None


def load_thresholds_from_mlflow_run(run_id: str) -> Optional[ThresholdSet]:
    """Best-effort load of the thresholds.json artifact logged by src/train.py
    (mlflow.log_dict) for the given run. Never raises - any MLflow error (run
    not found, artifact missing, tracking server unreachable, ...) falls back
    to config.yaml's hardcoded defaults instead."""
    try:
        import mlflow.artifacts

        data = mlflow.artifacts.load_dict(f"runs:/{run_id}/thresholds.json")
        return ThresholdSet.from_dict(data)
    except Exception:
        logger.warning("Could not load thresholds.json from MLflow run %s", run_id, exc_info=True)
        return None


def apply_thresholds(cfg: Config, thresholds: ThresholdSet) -> Config:
    """Returns a NEW Config with thresholds applied - never mutates `cfg`
    (which is a session-scoped pytest fixture reused across the whole test
    suite, and the same live object every request shares in api/main.py).

    domain_baselines is merged over cfg's existing baselines rather than
    replaced outright, so a domain absent from the run's val set (e.g. no
    Garage segments were ever labeled) keeps its config.yaml fallback instead
    of disappearing from the dict entirely.
    """
    merged_baselines = {**cfg.anomaly.domain_baselines, **thresholds.domain_baselines}
    new_anomaly = replace(cfg.anomaly, rel_threshold=thresholds.rel_threshold, domain_baselines=merged_baselines)
    return replace(cfg, anomaly=new_anomaly)
