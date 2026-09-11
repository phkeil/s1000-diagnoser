"""Pydantic request/response models for the labeling tool (api/tool_app.py).

Deliberately not shared with api/schemas.py - see api/labeling.py's module
docstring on why the two apps stay fully decoupled.
"""

from typing import Literal, Optional

from pydantic import BaseModel

DomainLiteral = Optional[Literal["Garage", "YouTube"]]


class SegmentInfo(BaseModel):
    segment_id: str  # "{upload_id}:{index}"
    index: int
    start_time: float
    end_time: float


class SourceMetadata(BaseModel):
    """Optional descriptive metadata about the uploaded recording - mirrors
    src.manifest.SourceFileMetadata field-for-field. contributor/exhaust_system/
    oil_type/etc. are filled in by whoever uploads their own bike's recording;
    original_codec is captured automatically from the upload's file extension,
    never supplied by the client."""

    bike_model: Optional[str] = None
    contributor: Optional[str] = None
    recording_device: Optional[str] = None
    original_codec: Optional[str] = None
    exhaust_system: Optional[str] = None
    model_year: Optional[int] = None
    kilometers_on_bike: Optional[float] = None
    oil_type: Optional[str] = None
    kilometers_since_last_oilchange: Optional[float] = None
    known_issues: Optional[str] = None
    notes: Optional[str] = None


class UploadResponse(BaseModel):
    upload_id: str
    filename: str
    domain: str
    sample_rate: int
    segment_duration_seconds: float
    step_duration_seconds: float
    source_metadata: SourceMetadata
    segments: list[SegmentInfo]


class SegmentLabel(BaseModel):
    segment_id: str
    label: Literal["healthy", "defective"]


class ConfirmRequest(BaseModel):
    labels: list[SegmentLabel]  # skipped segments simply absent


class ConfirmResponse(BaseModel):
    inserted: int
    failed: list[dict]  # [{segment_id, error}]
    manifest_ids: list[int]


class TrainRequest(BaseModel):
    epochs: int = 10
    run_name: str = ""


class TrainJobResponse(BaseModel):
    job_id: str
    status: Literal["running", "completed", "failed"]
    started_at: str
    log_tail: str = ""
    run_id: Optional[str] = None
    metrics: Optional[dict] = None
    error: Optional[str] = None
