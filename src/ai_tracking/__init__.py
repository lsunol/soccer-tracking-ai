"""Public API for the soccer tracking toolkit."""

from importlib import import_module
from typing import Any

from .clustering import TeamClusterer, cluster_single_frame
from .features import compute_hsv_histogram, detect_referee_from_histogram
from .models import (
	BoundingBox,
	CropExtraction,
	CropSample,
	Detection,
	FrameSummary,
	RunnerConfig,
)

__all__ = [
	"YoloVideoRunner",
	"run_yolo_on_video",
	"run_yolo_tracking_mode",
	"cluster_single_frame",
	"TeamClusterer",
	"compute_hsv_histogram",
	"detect_referee_from_histogram",
	"BoundingBox",
	"CropExtraction",
	"CropSample",
	"Detection",
	"FrameSummary",
	"RunnerConfig",
]

_YOLO_EXPORTS = {"YoloVideoRunner", "run_yolo_on_video", "run_yolo_tracking_mode"}


def __getattr__(name: str) -> Any:  # pragma: no cover - thin re-export layer
	if name in _YOLO_EXPORTS:
		module = import_module(".yolo_video", __name__)
		return getattr(module, name)
	raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
