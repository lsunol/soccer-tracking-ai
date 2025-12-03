"""High-level orchestration for running YOLO on soccer footage."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any, Iterable, Optional

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results

from .clustering import TeamClusterer
from .debug import DebugArtifactWriter
from .features import compute_hsv_histogram, detect_referee_from_histogram
from .models import (
    BoundingBox,
    CropExtraction,
    CropSample,
    Detection,
    FrameSummary,
    RunnerConfig,
)
from .rendering import draw_cluster_boxes, draw_team_boxes
from .team_assignment import TeamAssignmentManager


class YoloVideoRunner:
    """Execute YOLO on each frame and enrich detections with clustering info."""

    def __init__(
        self,
        model_source: str = "yolov8n.pt",
        *,
        device: Optional[str] = None,
        yolo_kwargs: Optional[dict[str, Any]] = None,
        video_fourcc: str = "mp4v",
        debug: bool = False,
        output_dir: Optional[str | Path] = None,
        torso_height_ratio: float = 0.4,
        torso_width_ratio: float = 0.6,
        history_window_size: int = 24,
        hsv_strategy: str = "crop_torso",
        config: RunnerConfig | None = None,
    ) -> None:
        if config is None:
            config = RunnerConfig(
                model_source=model_source,
                device=device,
                yolo_kwargs={"classes": [0], **(yolo_kwargs or {})},
                video_fourcc=video_fourcc,
                debug=debug,
                output_dir=Path(output_dir) if output_dir else None,
                torso_height_ratio=torso_height_ratio,
                torso_width_ratio=torso_width_ratio,
                history_window_size=history_window_size,
                hsv_strategy=hsv_strategy,  # type: ignore[arg-type]
            )
        self.config = config
        self._model: YOLO | None = None
        self._clusterer = TeamClusterer(n_clusters=2)
        self._team_assigner = TeamAssignmentManager(
            history_window_size=self.config.history_window_size
        )
        self._debug_writer = DebugArtifactWriter(self.config.output_dir if self.config.debug else None)
        self._yolo_writer: cv2.VideoWriter | None = None
        self._output_writer: cv2.VideoWriter | None = None

    @property
    def model(self) -> YOLO:
        if self._model is None:
            model = YOLO(self.config.model_source)
            if self.config.device:
                model.to(self.config.device)
            self._model = model
        return self._model

    def run(
        self,
        video_path: str | Path,
        frames: Optional[int | list[int]] = None,
    ) -> Iterator[Results]:
        yield from self._inference_generator(Path(video_path), frames)

    def _inference_generator(
        self,
        video_path: Path,
        frames: Optional[int | list[int]] = None,
    ) -> Generator[Results, None, None]:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            capture.release()
            raise FileNotFoundError(f"Unable to open video file: {video_path}")

        target_frames = self._normalize_frame_filter(frames)
        self._prepare_video_writers(capture)

        frame_idx = 0
        try:
            while True:
                read_success, frame = capture.read()
                if not read_success:
                    break

                if target_frames is not None and frame_idx not in target_frames:
                    frame_idx += 1
                    continue

                results = self.model.track(
                    frame,
                    persist=True,
                    verbose=False,
                    **self.config.yolo_kwargs,
                )

                for result in results:
                    annotated = result.plot()
                    if self._yolo_writer is not None:
                        self._yolo_writer.write(annotated)

                    samples = self._build_samples(frame_idx, frame, result)
                    summary = self._cluster_players(samples, frame_idx)
                    self._team_assigner.assign(samples)

                    self._write_outputs(frame_idx, frame, samples)
                    self._record_debug_summary(frame_idx, samples, summary)

                    yield result

                frame_idx += 1
                if target_frames is not None and frame_idx > max(target_frames):
                    break
        finally:
            capture.release()
            if self._yolo_writer is not None:
                self._yolo_writer.release()
            if self._output_writer is not None:
                self._output_writer.release()
            self._team_assigner.export_history(self.config.output_dir if self.config.debug else None)
            self._debug_writer.flush_metadata(video_path)

    def _prepare_video_writers(self, capture: cv2.VideoCapture) -> None:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_size = (
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920,
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080,
        )
        fourcc = cv2.VideoWriter_fourcc(*self.config.video_fourcc)

        if self.config.debug and self.config.output_dir:
            self.config.ensure_output_dir()
            yolo_path = self.config.output_dir / "yolo-video.mp4"
            self._yolo_writer = cv2.VideoWriter(str(yolo_path), fourcc, fps, frame_size)
            if not self._yolo_writer.isOpened():
                raise RuntimeError(f"Unable to create YOLO video at: {yolo_path}")

        if self.config.output_dir:
            self.config.ensure_output_dir()
            output_path = self.config.output_dir / "output-video.mp4"
            self._output_writer = cv2.VideoWriter(str(output_path), fourcc, fps, frame_size)
            if not self._output_writer.isOpened():
                raise RuntimeError(f"Unable to create output video at: {output_path}")

    def _build_samples(
        self,
        frame_idx: int,
        frame: np.ndarray,
        result: Results,
    ) -> list[CropSample]:
        extractions = extract_person_crops_from_frame(frame, result)
        samples: list[CropSample] = []
        for crop_index, extraction in enumerate(extractions, start=1):
            hist, crop_with_alpha = compute_hsv_histogram(
                extraction.image,
                torso_height_ratio=self.config.torso_height_ratio,
                torso_width_ratio=self.config.torso_width_ratio,
                hsv_strategy=self.config.hsv_strategy,
            )
            sample = extraction.sample
            sample.hsv_hist = hist
            sample.role = "referee" if detect_referee_from_histogram(hist) else "player"
            sample.filename = f"crop_{crop_index:03d}.png"
            sample.cluster_id = -1 if sample.role == "referee" else None
            samples.append(sample)

            if self.config.debug:
                image_to_save = crop_with_alpha if crop_with_alpha is not None else extraction.image
                self._debug_writer.save_crop(frame_idx, crop_index, image_to_save)
                self._debug_writer.save_histogram(frame_idx, crop_index, hist)

        if self.config.debug:
            self._debug_writer.save_frame(frame_idx, frame.copy(), suffix="raw")
        return samples

    def _cluster_players(
        self,
        samples: list[CropSample],
        frame_idx: int,
    ) -> FrameSummary | None:
        player_samples = [sample for sample in samples if sample.role == "player"]
        if not player_samples:
            return None
        frame_key = f"frame_{frame_idx:04d}"
        cluster_result = self._clusterer.cluster(
            player_samples,
            output_dir=self.config.output_dir,
            frame_key=frame_key,
            debug_plots=self.config.debug,
        )
        for sample, label in zip(player_samples, cluster_result.labels):
            sample.cluster_id = label
        return FrameSummary(
            frame_index=frame_idx,
            frame_filename=f"{frame_key}_raw.jpg" if self.config.debug else None,
            crops=samples,
            clustering=cluster_result.summary,
        )

    def _write_outputs(self, frame_idx: int, frame: np.ndarray, samples: Iterable[CropSample]) -> None:
        if self._output_writer is None:
            return
        frame_with_team = frame.copy()
        draw_team_boxes(frame_with_team, samples)
        self._output_writer.write(frame_with_team)
        if self.config.debug:
            self._debug_writer.save_frame(frame_idx, frame_with_team, suffix="team_id")
            cluster_frame = frame.copy()
            draw_cluster_boxes(cluster_frame, samples)
            self._debug_writer.save_frame(frame_idx, cluster_frame, suffix="cluster_id")

    def _record_debug_summary(
        self,
        frame_idx: int,
        samples: list[CropSample],
        summary: FrameSummary | None,
    ) -> None:
        if not self.config.debug:
            return
        if summary is None:
            summary = FrameSummary(
                frame_index=frame_idx,
                frame_filename=f"frame_{frame_idx:04d}_raw.jpg",
                crops=samples,
            )
        frame_key = f"frame_{frame_idx:04d}"
        self._debug_writer.record_frame_summary(frame_key, summary)

    @staticmethod
    def _normalize_frame_filter(frames: Optional[int | list[int]]) -> set[int] | None:
        if frames is None:
            return None
        if isinstance(frames, int):
            return {frames}
        return set(frames)


def extract_person_crops_from_frame(frame: np.ndarray, results: Results) -> list[CropExtraction]:
    """Convert YOLO detections into crop extractions for downstream processing."""
    extractions: list[CropExtraction] = []
    if results.boxes is None or len(results.boxes) == 0:
        return extractions

    height, width = frame.shape[:2]
    boxes = results.boxes
    for idx in range(len(boxes)):
        cls_raw = boxes.cls[idx]
        cls_id = int(cls_raw.cpu().item() if hasattr(cls_raw, "cpu") else cls_raw)
        if cls_id != 0:
            continue

        box_xyxy = boxes.xyxy[idx]
        coords = box_xyxy.cpu().numpy() if hasattr(box_xyxy, "cpu") else box_xyxy
        x1, y1, x2, y2 = map(int, coords)
        x1 = max(0, min(width - 1, x1))
        y1 = max(0, min(height - 1, y1))
        x2 = max(x1 + 1, min(width, x2))
        y2 = max(y1 + 1, min(height, y2))

        conf_raw = boxes.conf[idx]
        conf = float(conf_raw.cpu().item() if hasattr(conf_raw, "cpu") else conf_raw)
        class_name = results.names.get(cls_id, "unknown")
        track_id = None
        if hasattr(boxes, "id") and boxes.id is not None and len(boxes.id) > idx:
            track_raw = boxes.id[idx]
            track_id = int(track_raw.cpu().item() if hasattr(track_raw, "cpu") else track_raw)

        bbox = BoundingBox(x1=x1, y1=y1, x2=x2, y2=y2)
        detection = Detection(
            bbox=bbox,
            confidence=conf,
            class_id=cls_id,
            class_name=class_name,
            track_id=track_id,
        )
        crop_image = frame[y1:y2, x1:x2].copy()
        sample = CropSample(detection=detection)
        extractions.append(CropExtraction(image=crop_image, sample=sample))
    return extractions


def run_yolo_on_video(
    input_path: str | Path,
    *,
    output_dir: str | Path,
    frames: Optional[int | list[int]] = None,
    model_source: str = "yolov8n.pt",
    device: Optional[str] = None,
    yolo_kwargs: Optional[dict[str, Any]] = None,
    video_fourcc: str = "mp4v",
    debug: bool = False,
    hsv_strategy: str = "crop_torso",
) -> Iterator[Results]:
    """Process a video with YOLO detection and clustering."""
    runner = YoloVideoRunner(
        model_source=model_source,
        device=device,
        yolo_kwargs=yolo_kwargs,
        video_fourcc=video_fourcc,
        debug=debug,
        output_dir=output_dir,
        hsv_strategy=hsv_strategy,
    )
    yield from runner.run(input_path, frames=frames)


def run_yolo_tracking_mode(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model_source: str = "yolov8n.pt",
    device: Optional[str] = None,
    track_kwargs: Optional[dict[str, Any]] = None,
    fourcc: str = "mp4v",
) -> None:
    """Run Ultralytics tracking directly and persist the annotated video."""
    input_path = Path(input_path)
    output_path = Path(output_path)

    model = YOLO(model_source)
    if device:
        model.to(device)

    merged_kwargs = {"classes": [0], "stream": True, "verbose": False, **(track_kwargs or {})}
    results_generator = model.track(source=str(input_path), **merged_kwargs)

    writer: cv2.VideoWriter | None = None
    try:
        for result in results_generator:
            frame = result.orig_img.copy()
            boxes = result.boxes
            if writer is None:
                height, width = frame.shape[:2]
                fps = 30.0
                fourcc_code = cv2.VideoWriter_fourcc(*fourcc)
                writer = cv2.VideoWriter(str(output_path), fourcc_code, fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Unable to create output video at: {output_path}")
            if boxes is not None and boxes.id is not None:
                xywh = boxes.xywh.cpu().numpy() if hasattr(boxes.xywh, "cpu") else boxes.xywh
                ids = boxes.id.cpu().numpy() if hasattr(boxes.id, "cpu") else boxes.id
                for box, track_id in zip(xywh, ids):
                    x_center, y_center, w, h = box
                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    x2 = int(x_center + w / 2)
                    y2 = int(y_center + h / 2)
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    label = f"ID: {int(track_id)}"
                    cv2.putText(
                        frame,
                        label,
                        (x1, y1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2,
                    )
            writer.write(frame)
    finally:
        if writer is not None:
            writer.release()
