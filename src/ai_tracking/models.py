"""Core data models used across the soccer tracking pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

if TYPE_CHECKING:  # pragma: no cover - imported only for typing
    import numpy as np


@dataclass(frozen=True)
class BoundingBox:
    """Axis-aligned bounding box expressed in absolute pixel coordinates."""

    x1: int
    y1: int
    x2: int
    y2: int

    def as_tuple(self) -> tuple[int, int, int, int]:
        """Return the bounding box as a tuple convenient for OpenCV primitives."""
        return (self.x1, self.y1, self.x2, self.y2)


@dataclass(slots=True)
class Detection:
    """Represents a single YOLO detection for the person class."""

    bbox: BoundingBox
    confidence: float
    class_id: int
    class_name: str
    track_id: int | None = None


@dataclass(slots=True)
class CropSample:
    """Person crop together with derived features used for clustering."""

    detection: Detection
    hsv_hist: list[float] = field(default_factory=list)
    role: Literal["player", "referee"] = "player"
    cluster_id: int | None = None
    team_id: int | None = None
    filename: str | None = None


@dataclass(slots=True)
class CropExtraction:
    """Wrapper combining the raw crop image and its sample metadata."""

    image: "np.ndarray"
    sample: CropSample


@dataclass(slots=True)
class ClusteringSummary:
    """Metadata describing the outcome of the clustering stage for a frame."""

    k: int
    inertia: float
    cluster_distribution: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class ClusterResult:
    """Container with raw cluster labels and the related summary stats."""

    labels: list[int]
    summary: ClusteringSummary


@dataclass(slots=True)
class FrameSummary:
    """Container with per-frame derived data used for persistence/debugging."""

    frame_index: int
    frame_filename: str | None
    crops: list[CropSample]
    clustering: ClusteringSummary | None = None


@dataclass(slots=True)
class RunnerConfig:
    """Configuration knobs for the YOLO video runner."""

    model_source: str = "yolov8n.pt"
    device: str | None = None
    yolo_kwargs: dict[str, Any] = field(default_factory=lambda: {"classes": [0]})
    video_fourcc: str = "mp4v"
    debug: bool = False
    output_dir: Path | None = None
    torso_height_ratio: float = 0.4
    torso_width_ratio: float = 0.6
    history_window_size: int = 24
    hsv_strategy: Literal["crop_torso", "mask_green"] = "crop_torso"

    def ensure_output_dir(self) -> Path | None:
        """Create the output directory if necessary and return it."""
        if self.output_dir is None:
            return None
        self.output_dir.mkdir(parents=True, exist_ok=True)
        return self.output_dir
