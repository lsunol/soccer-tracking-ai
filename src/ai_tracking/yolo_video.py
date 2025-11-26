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
    video_path: str | Path,
    *,
    model_source: str = "yolov8n.pt",
    device: Optional[str] = None,
    yolo_kwargs: Optional[dict[str, Any]] = None,
    save_annotated: bool = False,
    annotated_suffix: str = "_annotated",
    annotated_path: Optional[str | Path] = None,
    annotated_fourcc: str = "mp4v",
) -> Iterator[Any]:
    """Yield raw YOLO detections and optionally dump the annotated clip beside the source video."""

    runner = YoloVideoRunner(
        model_source=model_source,
        device=device,
        yolo_kwargs=yolo_kwargs,
        save_annotated=save_annotated,
        annotated_suffix=annotated_suffix,
        annotated_path=annotated_path,
        annotated_fourcc=annotated_fourcc,
    )
    yield from runner.run(video_path)
