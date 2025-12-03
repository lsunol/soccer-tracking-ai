"""Rendering utilities for drawing detections and team assignments."""

from __future__ import annotations

from typing import Iterable

import cv2

from .models import CropSample


COLORBLIND_SAFE_PALETTE_BGR: list[tuple[int, int, int]] = [
    (178, 114, 0),
    (0, 159, 230),
    (66, 228, 240),
    (167, 121, 204),
    (115, 158, 0),
]


def draw_team_boxes(frame, crops: Iterable[CropSample]) -> None:
    """Annotate the provided frame with team-colored bounding boxes."""
    for crop in crops:
        bbox = crop.detection.bbox.as_tuple()
        track_id = crop.detection.track_id
        x1, y1, x2, y2 = bbox

        if crop.role == "referee" or (crop.team_id or 0) == 0:
            color = (128, 128, 128)
            label = "REF"
        else:
            palette_idx = ((crop.team_id or 1) - 1) % len(COLORBLIND_SAFE_PALETTE_BGR)
            color = COLORBLIND_SAFE_PALETTE_BGR[palette_idx]
            label = str(track_id) if track_id is not None else f"T{crop.team_id}"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - 8),
            (x1 + label_size[0] + 4, y1),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )


def draw_cluster_boxes(frame, crops: Iterable[CropSample]) -> None:
    """Annotate the frame using cluster identifiers only (pre-voting visualization)."""
    for crop in crops:
        bbox = crop.detection.bbox.as_tuple()
        x1, y1, x2, y2 = bbox
        cluster_id = crop.cluster_id or 0
        track_id = crop.detection.track_id

        if crop.role == "referee" or cluster_id < 0:
            color = (128, 128, 128)
            label = "REF"
        else:
            palette_idx = cluster_id % len(COLORBLIND_SAFE_PALETTE_BGR)
            color = COLORBLIND_SAFE_PALETTE_BGR[palette_idx]
            label = str(track_id) if track_id is not None else f"C{cluster_id}"
            if crop.team_id is not None and cluster_id != crop.team_id - 1:
                label += "*"

        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
        label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
        cv2.rectangle(
            frame,
            (x1, y1 - label_size[1] - 8),
            (x1 + label_size[0] + 4, y1),
            color,
            -1,
        )
        cv2.putText(
            frame,
            label,
            (x1 + 2, y1 - 4),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2,
        )
