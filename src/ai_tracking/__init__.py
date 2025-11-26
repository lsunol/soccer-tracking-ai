"""Minimal YOLO video inference API."""

from .yolo_video import YoloVideoRunner, run_yolo_on_video

__all__ = ["YoloVideoRunner", "run_yolo_on_video"]
