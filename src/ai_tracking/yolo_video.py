"""Thin wrapper around a pretrained YOLO model for frame-by-frame video inference."""

from __future__ import annotations

import json
from collections.abc import Generator, Iterator
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from ultralytics import YOLO
from ultralytics.engine.results import Results


class YoloVideoRunner:
    """Execute YOLO on each frame of a video, optionally persisting annotated frames and crops."""

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
        save_crops: bool = False,
        output_dir: Optional[str | Path] = None,
    ) -> None:
        self.model_source = model_source
        self.device = device
        self.yolo_kwargs = {"classes": [0], **(yolo_kwargs or {})}
        self.save_annotated = save_annotated
        self.annotated_suffix = annotated_suffix
        self.annotated_path = Path(annotated_path) if annotated_path else None
        self.annotated_fourcc = annotated_fourcc
        self.save_crops = save_crops
        self.output_dir = Path(output_dir) if output_dir else None
        self._model: Optional[YOLO] = None

    @property
    def model(self) -> YOLO:
        if self._model is None:
            model = YOLO(self.model_source)
            if self.device:
                model.to(self.device)
            self._model = model
        return self._model

    def run(
        self,
        video_path: str | Path,
        frames: Optional[int | list[int]] = None,
    ) -> Iterator[Any]:
        return self._inference_generator(Path(video_path), frames)

    def _inference_generator(
        self,
        video_path: Path,
        frames: Optional[int | list[int]] = None,
    ) -> Generator[Any, None, None]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            cap.release()
            raise FileNotFoundError(f"Unable to open video file: {video_path}")

        # Normalize frames to a set for quick lookup
        target_frames: Optional[set[int]] = None
        if frames is not None:
            target_frames = {frames} if isinstance(frames, int) else set(frames)

        writer = None
        if self.save_annotated:
            try:
                writer = self._build_writer(video_path, cap)
            except Exception:
                cap.release()
                raise

        frame_idx = 0
        frames_metadata: dict[str, dict[str, Any]] = {}

        try:
            while True:
                read_success, frame = cap.read()
                if not read_success:
                    break

                # Skip frame if not in target set
                if target_frames is not None and frame_idx not in target_frames:
                    frame_idx += 1
                    continue

                results = self.model(frame, verbose=False, **self.yolo_kwargs)
                # YOLO returns a list of Results even for a single frame.
                for result in results:
                    if writer is not None:
                        annotated_frame = result.plot()  # BGR ndarray ready for VideoWriter
                        writer.write(annotated_frame)

                    # Extract and save crops if enabled
                    if self.save_crops and self.output_dir:
                        # Create frame-specific directory
                        frame_dir = self.output_dir / f"frame_{frame_idx:04d}"
                        frame_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save the full frame image
                        frame_filename = f"frame_{frame_idx:04d}.jpg"
                        frame_path = frame_dir / frame_filename
                        cv2.imwrite(str(frame_path), frame)
                        
                        # Extract and save crops
                        crops_with_meta = extract_person_crops_from_frame(frame, result)
                        frame_crops = []
                        for crop_idx, (crop, meta) in enumerate(crops_with_meta, start=1):
                            crop_filename = f"crop_{crop_idx:04d}.jpg"
                            crop_path = frame_dir / crop_filename
                            cv2.imwrite(str(crop_path), crop)

                            meta["filename"] = crop_filename
                            frame_crops.append(meta)
                        
                        # Store frame metadata with its crops
                        frame_key = f"frame_{frame_idx:04d}"
                        frames_metadata[frame_key] = {
                            "frame_idx": frame_idx,
                            "frame_filename": frame_filename,
                            "num_crops": len(frame_crops),
                            "crops": frame_crops,
                        }

                    yield result

                frame_idx += 1

                # Early exit if we've processed all target frames
                if target_frames is not None and frame_idx > max(target_frames):
                    break

        finally:
            cap.release()
            if writer is not None:
                writer.release()

            # Save crops metadata JSON if any crops were saved
            if self.save_crops and self.output_dir and frames_metadata:
                json_path = self.output_dir / "crops_metadata.json"
                total_crops = sum(frame_data["num_crops"] for frame_data in frames_metadata.values())
                metadata_summary = {
                    "video_path": str(video_path),
                    "total_frames_processed": len(frames_metadata),
                    "total_crops": total_crops,
                    "frames": frames_metadata,
                }
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(metadata_summary, f, indent=2)

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


def extract_person_crops_from_frame(
    frame: np.ndarray,
    results: Results,
) -> list[tuple[np.ndarray, dict[str, Any]]]:
    """Extract person crops and metadata from a YOLO Results object.
    
    Args:
        frame: Original frame as BGR numpy array
        results: YOLO Results object containing detections
        
    Returns:
        List of (crop, metadata) tuples where:
            - crop: np.ndarray with cropped person image
            - metadata: dict with bbox, class_id, class_name, confidence
    """
    crops_with_meta = []
    
    if results.boxes is None or len(results.boxes) == 0:
        return crops_with_meta
    
    boxes = results.boxes
    for idx in range(len(boxes)):
        # Extract box coordinates (xyxy format)
        box_xyxy = boxes.xyxy[idx].cpu().numpy() if hasattr(boxes.xyxy[idx], "cpu") else boxes.xyxy[idx]
        x1, y1, x2, y2 = map(int, box_xyxy)
        
        # Extract class and confidence
        cls_id = int(boxes.cls[idx].cpu().item() if hasattr(boxes.cls[idx], "cpu") else boxes.cls[idx])
        conf = float(boxes.conf[idx].cpu().item() if hasattr(boxes.conf[idx], "cpu") else boxes.conf[idx])
        class_name = results.names.get(cls_id, "unknown")
        
        # Only process person detections (class 0)
        if cls_id != 0:
            continue
        
        # Crop the bounding box from frame
        crop = frame[y1:y2, x1:x2].copy()
        
        metadata = {
            "bbox": [x1, y1, x2, y2],
            "class_id": cls_id,
            "class_name": class_name,
            "confidence": conf,
        }
        
        crops_with_meta.append((crop, metadata))
    
    return crops_with_meta


def run_yolo_on_video(
    input_path: str | Path,
    *,
    output_dir: Optional[str | Path] = None,
    frames: Optional[int | list[int]] = None,
    model_source: str = "yolov8n.pt",
    device: Optional[str] = None,
    yolo_kwargs: Optional[dict[str, Any]] = None,
    save_annotated: bool = False,
    annotated_fourcc: str = "mp4v",
    save_crops: bool = False,
) -> Iterator[Any]:
    """Yield raw YOLO detections and optionally save annotated video and/or person crops.
    
    Args:
        input_path: Path to input video file
        output_dir: Directory where outputs (annotated video, frame folders, crops_metadata.json) will be saved
        frames: Optional frame index (int) or list of frame indices to process. If None, process all frames.
        model_source: YOLO model weights to load
        device: Device to run inference on
        yolo_kwargs: Additional kwargs for YOLO inference
        save_annotated: Whether to save annotated frames to video
        annotated_fourcc: Video codec fourcc code
        save_crops: Whether to save person crops and full frames to disk
    """
    output_dir_path = Path(output_dir) if output_dir else None
    
    # Determine annotated video path if needed
    annotated_path = None
    if save_annotated:
        if output_dir_path is None:
            raise ValueError("output_dir is required when save_annotated=True")
        annotated_path = output_dir_path / "annotated-video.mp4"
    
    if save_crops and output_dir_path is None:
        raise ValueError("output_dir is required when save_crops=True")

    runner = YoloVideoRunner(
        model_source=model_source,
        device=device,
        yolo_kwargs=yolo_kwargs,
        save_annotated=save_annotated,
        annotated_suffix="",
        annotated_path=annotated_path,
        annotated_fourcc=annotated_fourcc,
        save_crops=save_crops,
        output_dir=output_dir_path,
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
