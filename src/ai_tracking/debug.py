"""Debug helpers for persisting intermediate artifacts of the pipeline."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import cv2

from .features import save_histogram_visualization
from .models import FrameSummary


class DebugArtifactWriter:
	"""Persist crops, histograms and metadata when debug mode is enabled."""

	def __init__(self, output_dir: Path | None) -> None:
		self.output_dir = output_dir
		self.frames: Dict[str, FrameSummary] = {}

	@property
	def enabled(self) -> bool:
		return self.output_dir is not None

	def frame_directory(self, frame_index: int) -> Path | None:
		if not self.enabled:
			return None
		frame_dir = self.output_dir / f"frame_{frame_index:04d}"
		frame_dir.mkdir(parents=True, exist_ok=True)
		return frame_dir

	def save_frame(self, frame_index: int, frame, suffix: str = "") -> None:
		if not self.enabled:
			return
		frame_dir = self.frame_directory(frame_index)
		if frame_dir is None:
			return
		postfix = f"_{suffix}" if suffix else ""
		frame_path = frame_dir / f"frame_{frame_index:04d}{postfix}.jpg"
		cv2.imwrite(str(frame_path), frame)

	def save_crop(self, frame_index: int, crop_index: int, image) -> None:
		if not self.enabled:
			return
		frame_dir = self.frame_directory(frame_index)
		if frame_dir is None:
			return
		crop_path = frame_dir / f"crop_{crop_index:03d}.png"
		cv2.imwrite(str(crop_path), image)

	def save_histogram(self, frame_index: int, crop_index: int, histogram: list[float]) -> None:
		if not self.enabled:
			return
		frame_dir = self.frame_directory(frame_index)
		if frame_dir is None:
			return
		hist_path = frame_dir / f"crop_{crop_index:03d}_hist.png"
		save_histogram_visualization(histogram, hist_path)

	def record_frame_summary(self, frame_key: str, summary: FrameSummary) -> None:
		if self.enabled:
			self.frames[frame_key] = summary

	def flush_metadata(self, video_path: Path) -> None:
		if not self.enabled or not self.frames:
			return
		total_crops = sum(len(summary.crops) for summary in self.frames.values())
		metadata = {
			"video_path": str(video_path),
			"total_frames_processed": len(self.frames),
			"total_crops": total_crops,
			"frames": {
				key: {
					"frame_idx": summary.frame_index,
					"frame_filename": summary.frame_filename,
					"num_crops": len(summary.crops),
					"clustering": {
						"k": summary.clustering.k if summary.clustering else None,
						"inertia": summary.clustering.inertia if summary.clustering else None,
						"cluster_distribution": summary.clustering.cluster_distribution
						if summary.clustering
						else None,
					},
					"crops": [
						{
							"bbox": sample.detection.bbox.as_tuple(),
							"track_id": sample.detection.track_id,
							"role": sample.role,
							"cluster_id": sample.cluster_id,
							"team_id": sample.team_id,
							"filename": sample.filename,
							"confidence": sample.detection.confidence,
						}
						for sample in summary.crops
					],
				}
				for key, summary in self.frames.items()
			},
		}
		json_path = self.output_dir / "crops_metadata.json"
		with open(json_path, "w", encoding="utf-8") as handle:
			json.dump(metadata, handle, indent=2)