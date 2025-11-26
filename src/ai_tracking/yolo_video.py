"""Thin wrapper around a pretrained YOLO model for frame-by-frame video inference."""

from __future__ import annotations

from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any, Optional

import cv2
from ultralytics import YOLO


class YoloVideoRunner:
    """Execute YOLO on each frame of a video, optionally persisting annotated frames."""

    def __init__(
        self,
        model_source: str = "yolov8n.pt",
        *,
        device: Optional[str] = None,
        yolo_kwargs: Optional[dict[str, Any]] = None,
        save_annotated: bool = False,
        annotated_suffix: str = "_annotated",
        annotated_path: Optional[str | Path] = None,
        annotated_fourcc: str = "mp4v",
    ) -> None:
        self.model_source = model_source
        self.device = device
        self.yolo_kwargs = {"classes": [0], **(yolo_kwargs or {})}
        self.save_annotated = save_annotated
        self.annotated_suffix = annotated_suffix
        self.annotated_path = Path(annotated_path) if annotated_path else None
        self.annotated_fourcc = annotated_fourcc
        self._model: Optional[YOLO] = None

    @property
    def model(self) -> YOLO:
        if self._model is None:
            model = YOLO(self.model_source)
            if self.device:
                model.to(self.device)
            self._model = model
        return self._model

    def run(self, video_path: str | Path) -> Iterator[Any]:
        return self._inference_generator(Path(video_path))

    def _inference_generator(self, video_path: Path) -> Generator[Any, None, None]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            raise FileNotFoundError(f"Unable to open video file: {video_path}")

        writer = None
        if self.save_annotated:
            try:
                writer = self._build_writer(video_path, cap)
            except Exception:
                cap.release()
                raise

        try:
            while True:
                read_success, frame = cap.read()
                if not read_success:
                    break

                results = self.model(frame, verbose=False, **self.yolo_kwargs)
                # YOLO returns a list of Results even for a single frame.
                for result in results:
                    if writer is not None:
                        annotated_frame = result.plot()  # BGR ndarray ready for VideoWriter
                        writer.write(annotated_frame)
                    yield result
        finally:
            cap.release()
            if writer is not None:
                writer.release()

    def _build_writer(self, video_path: Path, cap: cv2.VideoCapture) -> cv2.VideoWriter:
        output_path = self.annotated_path or video_path.with_name(
            f"{video_path.stem}{self.annotated_suffix}{video_path.suffix}"
        )
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        fourcc = cv2.VideoWriter_fourcc(*self.annotated_fourcc)  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to create annotated video at: {output_path}")
        return writer


def run_yolo_on_video(
    input_path: str | Path,
    *,
    output_path: Optional[str | Path] = None,
    model_source: str = "yolov8n.pt",
    device: Optional[str] = None,
    yolo_kwargs: Optional[dict[str, Any]] = None,
    save_annotated: bool = False,
    annotated_fourcc: str = "mp4v",
) -> Iterator[Any]:
    """Yield raw YOLO detections and optionally save annotated video to output_path.
    
    Args:
        input_path: Path to input video file
        output_path: Path where annotated video will be saved (required if save_annotated=True)
        model_source: YOLO model weights to load
        device: Device to run inference on
        yolo_kwargs: Additional kwargs for YOLO inference
        save_annotated: Whether to save annotated frames to video
        annotated_fourcc: Video codec fourcc code
    """
    if save_annotated and output_path is None:
        raise ValueError("output_path is required when save_annotated=True")

    runner = YoloVideoRunner(
        model_source=model_source,
        device=device,
        yolo_kwargs=yolo_kwargs,
        save_annotated=save_annotated,
        annotated_suffix="",  # No longer needed with explicit output_path
        annotated_path=output_path,
        annotated_fourcc=annotated_fourcc,
    )
    yield from runner.run(input_path)


def run_yolo_tracking_mode(
    input_path: str | Path,
    output_path: str | Path,
    *,
    model_source: str = "yolov8n.pt",
    device: Optional[str] = None,
    track_kwargs: Optional[dict[str, Any]] = None,
    fourcc: str = "mp4v",
) -> None:
    """Execute YOLO tracking using Ultralytics' built-in tracker and save annotated video.
    
    This function uses model.track() internally, which maintains object IDs across frames.
    It writes a video with bounding boxes and tracking IDs overlaid on each frame.
    
    Args:
        input_path: Path to input video file
        output_path: Path where annotated video will be saved
        model_source: YOLO model weights to load
        device: Device to run inference on (e.g., 'cuda:0', 'cpu')
        track_kwargs: Additional kwargs for model.track() (merged with default person-only filter)
        fourcc: Video codec fourcc code
    """
    input_path = Path(input_path)
    output_path = Path(output_path)
    
    # Initialize model
    model = YOLO(model_source)
    if device:
        model.to(device)
    
    # Merge default person-only filtering with user kwargs
    merged_kwargs = {"classes": [0], "stream": True, "verbose": False, **(track_kwargs or {})}
    
    # Run tracking
    results_generator = model.track(source=str(input_path), **merged_kwargs)
    
    writer: Optional[cv2.VideoWriter] = None
    
    try:
        for result in results_generator:
            # Extract frame and tracking data
            frame = result.orig_img.copy()
            boxes = result.boxes
            
            # Initialize writer on first frame
            if writer is None:
                height, width = frame.shape[:2]
                fps = 30.0  # Ultralytics doesn't expose original FPS, default to 30
                fourcc_code = cv2.VideoWriter_fourcc(*fourcc)  # type: ignore[attr-defined]
                writer = cv2.VideoWriter(str(output_path), fourcc_code, fps, (width, height))
                if not writer.isOpened():
                    raise RuntimeError(f"Unable to create output video at: {output_path}")
            
            # Draw bounding boxes and track IDs
            if boxes is not None and boxes.id is not None:
                xywh = boxes.xywh.cpu().numpy() if hasattr(boxes.xywh, "cpu") else boxes.xywh  # type: ignore[union-attr]
                ids = boxes.id.cpu().numpy() if hasattr(boxes.id, "cpu") else boxes.id  # type: ignore[union-attr]
                for box, track_id in zip(xywh, ids):
                    x_center, y_center, w, h = box
                    x1 = int(x_center - w / 2)
                    y1 = int(y_center - h / 2)
                    x2 = int(x_center + w / 2)
                    y2 = int(y_center + h / 2)
                    
                    # Draw bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    
                    # Draw track ID
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
