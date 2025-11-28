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


def compute_hsv_histogram(
    crop: np.ndarray,
    *,
    h_bins: int = 8,
    s_bins: int = 4,
    v_bins: int = 4,
    shirt_region_ratio: float = 0.5,
) -> list[float]:
    """Calculate normalized HSV histogram from the upper portion (shirt region) of a person crop.
    
    Args:
        crop: BGR image as numpy array (from OpenCV)
        h_bins: Number of bins for Hue channel (0-180 in OpenCV)
        s_bins: Number of bins for Saturation channel (0-255)
        v_bins: Number of bins for Value channel (0-255)
        shirt_region_ratio: Proportion of upper crop to analyze (0.5 = top 50%)
        
    Returns:
        Normalized histogram as 1D list of floats (length = h_bins * s_bins * v_bins)
    """
    if crop.size == 0:
        # Return zero vector for empty crops
        return [0.0] * (h_bins * s_bins * v_bins)
    
    # Extract upper region (shirt area)
    height = crop.shape[0]
    shirt_height = int(height * shirt_region_ratio)
    if shirt_height == 0:
        shirt_height = 1
    shirt_region = crop[:shirt_height, :]
    
    # Convert to HSV
    hsv = cv2.cvtColor(shirt_region, cv2.COLOR_BGR2HSV)
    
    # Calculate 3D histogram
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],  # H, S, V channels
        None,
        [h_bins, s_bins, v_bins],
        [0, 180, 0, 256, 0, 256]  # H range: 0-180, S/V range: 0-256
    )
    
    # Normalize histogram
    cv2.normalize(hist, hist)
    
    # Flatten to 1D list
    return hist.flatten().tolist()


def _save_histogram_visualization(
    hsv_hist: list[float],
    output_path: Path,
    *,
    h_bins: int = 8,
    s_bins: int = 4,
    v_bins: int = 4,
) -> None:
    """Save a visual representation of the HSV histogram as a bar chart with color coding.
    
    Args:
        hsv_hist: Flattened HSV histogram vector
        output_path: Path where the visualization image will be saved
        h_bins: Number of hue bins used in histogram
        s_bins: Number of saturation bins used in histogram
        v_bins: Number of value bins used in histogram
    """
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb
    
    # Generate color for each bin based on its H, S, V coordinates
    colors = []
    for i in range(len(hsv_hist)):
        # Decode bin index back to h, s, v coordinates
        h_idx = i // (s_bins * v_bins)
        remaining = i % (s_bins * v_bins)
        s_idx = remaining // v_bins
        v_idx = remaining % v_bins
        
        # Convert to normalized HSV values (0-1 range for matplotlib)
        # H: 0-180 in OpenCV, map to 0-1
        # S: 0-255 in OpenCV, map to 0-1
        # V: 0-255 in OpenCV, map to 0-1
        h_normalized = (h_idx + 0.5) / h_bins  # Center of bin
        s_normalized = (s_idx + 0.5) / s_bins
        v_normalized = (v_idx + 0.5) / v_bins
        
        # Convert HSV to RGB for matplotlib
        rgb = hsv_to_rgb([h_normalized, s_normalized, v_normalized])
        colors.append(rgb)
    
    # Create figure with two subplots: histogram + color bar
    fig, (ax_hist, ax_color) = plt.subplots(2, 1, figsize=(12, 5), 
                                             gridspec_kw={'height_ratios': [4, 0.3], 'hspace': 0.15})
    
    # Plot histogram bars with corresponding colors
    x = np.arange(len(hsv_hist))
    ax_hist.bar(x, hsv_hist, width=1.0, color=colors, edgecolor='black', linewidth=0.3)
    
    # Styling for histogram
    ax_hist.set_ylabel('Normalized Frequency', fontsize=10)
    ax_hist.set_title(f'HSV Histogram ({h_bins}H × {s_bins}S × {v_bins}V = {len(hsv_hist)} bins)', fontsize=11)
    ax_hist.grid(axis='y', alpha=0.3)
    ax_hist.set_xlim(-0.5, len(hsv_hist) - 0.5)
    
    # Create color reference bar at bottom
    color_array = np.array(colors).reshape(1, -1, 3)
    ax_color.imshow(color_array, aspect='auto', extent=[0, len(hsv_hist), 0, 1])
    ax_color.set_yticks([])
    ax_color.set_xlabel('Histogram Bin Index (color = H-S-V value)', fontsize=10)
    ax_color.set_xlim(0, len(hsv_hist))
    
    # Add vertical lines to separate Hue blocks
    for h in range(1, h_bins):
        bin_boundary = h * s_bins * v_bins
        ax_hist.axvline(bin_boundary - 0.5, color='white', linewidth=2, linestyle='--', alpha=0.7)
        ax_color.axvline(bin_boundary, color='white', linewidth=2, linestyle='--', alpha=0.7)
    
    # Save figure
    plt.tight_layout()
    plt.savefig(str(output_path), dpi=100, bbox_inches='tight')
    plt.close(fig)


class YoloVideoRunner:
    """Execute YOLO on each frame of a video, generating team-labeled output video."""

    def __init__(
        self,
        model_source: str = "yolov8n.pt",
        *,
        device: Optional[str] = None,
        yolo_kwargs: Optional[dict[str, Any]] = None,
        video_fourcc: str = "mp4v",
        debug: bool = False,
        output_dir: Optional[str | Path] = None,
        clustering_k_min: int = 2,
        clustering_k_max: int = 5,
    ) -> None:
        self.model_source = model_source
        self.device = device
        self.yolo_kwargs = {"classes": [0], **(yolo_kwargs or {})}
        self.video_fourcc = video_fourcc
        self.debug = debug
        self.output_dir = Path(output_dir) if output_dir else None
        self.clustering_k_min = clustering_k_min
        self.clustering_k_max = clustering_k_max
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

        # Prepare video writers
        yolo_writer = None
        output_writer = None
        if self.debug:
            try:
                yolo_writer = self._build_yolo_writer(video_path, cap)
            except Exception:
                cap.release()
                raise
        
        if self.output_dir:
            try:
                output_writer = self._build_output_writer(cap)
            except Exception:
                cap.release()
                if yolo_writer:
                    yolo_writer.release()
                raise

        frame_idx = 0
        frames_metadata: dict[str, dict[str, Any]] = {}  # Only used in debug mode

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
                    # Save YOLO annotated frame in debug mode
                    if yolo_writer is not None:
                        annotated_frame = result.plot()  # BGR ndarray ready for VideoWriter
                        yolo_writer.write(annotated_frame)

                    # Always extract crops and compute histograms for clustering
                    crops_with_meta = extract_person_crops_from_frame(frame, result)
                    frame_crops = []
                    
                    # Prepare frame directory if debug mode is enabled
                    frame_dir = None
                    frame_filename = None
                    frame_key = f"frame_{frame_idx:04d}"
                    
                    if self.debug and self.output_dir:
                        frame_dir = self.output_dir / frame_key
                        frame_dir.mkdir(parents=True, exist_ok=True)
                        
                        # Save the full frame image
                        frame_filename = f"frame_{frame_idx:04d}.jpg"
                        frame_path = frame_dir / frame_filename
                        cv2.imwrite(str(frame_path), frame)
                    
                    # Process each crop and compute histograms
                    for crop_idx, (crop, meta) in enumerate(crops_with_meta, start=1):
                        crop_filename = f"crop_{crop_idx:04d}.jpg"
                        
                        # Save crop image to disk if debug mode
                        if self.debug and frame_dir:
                            crop_path = frame_dir / crop_filename
                            cv2.imwrite(str(crop_path), crop)
                        
                        # Always compute HSV histogram for clustering
                        hsv_hist = compute_hsv_histogram(crop)
                        meta["hsv_hist"] = hsv_hist
                        meta["filename"] = crop_filename
                        
                        # Save histogram visualization in debug mode
                        if self.debug and frame_dir:
                            _save_histogram_visualization(
                                hsv_hist,
                                frame_dir / f"crop_{crop_idx:04d}_hist.png"
                            )
                        
                        frame_crops.append(meta)
                    
                    # Perform clustering on this frame immediately
                    from .clustering import cluster_single_frame
                    
                    cluster_labels, clustering_info = cluster_single_frame(
                        frame_crops,
                        k_min=self.clustering_k_min,
                        k_max=self.clustering_k_max,
                        output_dir=self.output_dir,
                        frame_key=frame_key,
                        debug_plots=self.debug,
                    )
                    
                    # Assign cluster IDs to crops
                    for i, label in enumerate(cluster_labels):
                        frame_crops[i]["cluster_id"] = label
                    
                    # Generate and write output frame with cluster colors
                    output_frame = None
                    if output_writer is not None:
                        output_frame = self._draw_cluster_boxes(
                            frame.copy(),
                            frame_crops,
                            clustering_info.get("optimal_k", 1)
                        )
                        output_writer.write(output_frame)
                        
                        # Save clustered frame in debug mode
                        if self.debug and frame_dir:
                            clustered_filename = f"frame_{frame_idx:04d}_clustered.jpg"
                            clustered_path = frame_dir / clustered_filename
                            cv2.imwrite(str(clustered_path), output_frame)
                    
                    # Store frame metadata (optional, for debug JSON)
                    if self.debug:
                        frames_metadata[frame_key] = {
                            "frame_idx": frame_idx,
                            "frame_filename": frame_filename,
                            "num_crops": len(frame_crops),
                            "crops": frame_crops,
                            "clustering": clustering_info,
                        }

                    yield result

                frame_idx += 1

                # Early exit if we've processed all target frames
                if target_frames is not None and frame_idx > max(target_frames):
                    break

        finally:
            cap.release()
            if yolo_writer is not None:
                yolo_writer.release()
            if output_writer is not None:
                output_writer.release()
                if self.output_dir:
                    print(f"\nOutput video saved to: {self.output_dir / 'output-video.mp4'}")

            # Save metadata JSON in debug mode
            if self.debug and self.output_dir and frames_metadata:
                total_crops = sum(frame_data["num_crops"] for frame_data in frames_metadata.values())
                metadata_summary = {
                    "video_path": str(video_path),
                    "total_frames_processed": len(frames_metadata),
                    "total_crops": total_crops,
                    "frames": frames_metadata,
                }
                
                json_path = self.output_dir / "crops_metadata.json"
                with open(json_path, "w", encoding="utf-8") as f:
                    json.dump(metadata_summary, f, indent=2)
                print(f"Metadata saved to: {json_path}")

    def _build_yolo_writer(self, video_path: Path, cap: cv2.VideoCapture) -> cv2.VideoWriter:
        """Build video writer for YOLO annotated frames (debug only)."""
        if not self.output_dir:
            raise ValueError("output_dir required for YOLO video")
        
        output_path = self.output_dir / "yolo-video.mp4"
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        fourcc = cv2.VideoWriter_fourcc(*self.video_fourcc)  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to create YOLO video at: {output_path}")
        return writer
    
    def _build_output_writer(self, cap: cv2.VideoCapture) -> cv2.VideoWriter:
        """Build video writer for cluster-colored output video."""
        if not self.output_dir:
            raise ValueError("output_dir required for output video")
        
        output_path = self.output_dir / "output-video.mp4"
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 1920
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 1080
        fourcc = cv2.VideoWriter_fourcc(*self.video_fourcc)  # type: ignore[attr-defined]
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        if not writer.isOpened():
            raise RuntimeError(f"Unable to create output video at: {output_path}")
        return writer
    
    def _draw_cluster_boxes(
        self,
        frame: np.ndarray,
        crops_data: list[dict[str, Any]],
        max_k: int,
    ) -> np.ndarray:
        """Draw cluster-colored bounding boxes on frame.
        
        Args:
            frame: Original frame to draw on
            crops_data: List of crop metadata with bbox and cluster_id
            max_k: Maximum number of clusters (for color selection)
            
        Returns:
            Frame with drawn bounding boxes
        """
        import matplotlib.pyplot as plt
        
        # Get cluster colors from tab10 colormap (same as PCA visualization)
        cluster_colors_rgb = plt.cm.tab10.colors[:max_k]
        cluster_colors_bgr = [
            (int(rgb[2] * 255), int(rgb[1] * 255), int(rgb[0] * 255))
            for rgb in cluster_colors_rgb
        ]
        
        # Draw cluster-colored rectangles
        for crop_meta in crops_data:
            bbox = crop_meta["bbox"]
            cluster_id = crop_meta.get("cluster_id", 0)
            
            x1, y1, x2, y2 = bbox
            color = cluster_colors_bgr[cluster_id]
            
            # Draw rectangle with cluster color (thickness 2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw cluster ID label
            label = f"C{cluster_id}"
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
            cv2.rectangle(
                frame,
                (x1, y1 - label_size[1] - 8),
                (x1 + label_size[0] + 4, y1),
                color,
                -1
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
        
        return frame


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
    output_dir: str | Path,
    frames: Optional[int | list[int]] = None,
    model_source: str = "yolov8n.pt",
    device: Optional[str] = None,
    yolo_kwargs: Optional[dict[str, Any]] = None,
    video_fourcc: str = "mp4v",
    debug: bool = False,
    clustering_k_min: int = 2,
    clustering_k_max: int = 5,
) -> Iterator[Any]:
    """Process video with YOLO detection and team clustering, generating labeled output video.
    
    Args:
        input_path: Path to input video file
        output_dir: Directory where output-video.mp4 will be saved (and debug files if debug=True)
        frames: Optional frame index (int) or list of frame indices to process. If None, process all frames.
        model_source: YOLO model weights to load
        device: Device to run inference on
        yolo_kwargs: Additional kwargs for YOLO inference
        video_fourcc: Video codec fourcc code
        debug: If True, save crops, histograms, clustering plots, yolo-video.mp4, and metadata to disk
        clustering_k_min: Minimum number of clusters to try (default: 2)
        clustering_k_max: Maximum number of clusters to try (default: 5)
    """
    output_dir_path = Path(output_dir)

    runner = YoloVideoRunner(
        model_source=model_source,
        device=device,
        yolo_kwargs=yolo_kwargs,
        video_fourcc=video_fourcc,
        debug=debug,
        output_dir=output_dir_path,
        clustering_k_min=clustering_k_min,
        clustering_k_max=clustering_k_max,
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
