"""Pydantic request/response models for the inference API (UPGRADE_PLAN.md §5)."""

from typing import Literal, Optional

from pydantic import BaseModel

# Matches the keys of config.yaml's anomaly.domain_baselines. Shared by the
# /predict Form field (api/main.py) and src.inference.score_audio_file's
# optional domain override.
DomainLiteral = Optional[Literal["Garage", "YouTube"]]


class PredictionResponse(BaseModel):
    segment_scores: list[float]  # raw reconstruction-MSE per segment
    relative_scores: list[float]  # segment_scores normalized against the domain healthy baseline
    is_anomalous: bool  # True if any (smoothed) segment crossed the anomaly threshold
    model_version: str


class HealthResponse(BaseModel):
    status: str
    model_version: str
    model_run_id: Optional[str] = None
