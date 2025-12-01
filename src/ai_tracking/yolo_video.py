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
    torso_height_ratio: float = 0.4,
    torso_width_ratio: float = 0.6,
    hsv_strategy: str = "crop_torso",
    green_h_range: tuple[int, int] = (35, 85),
    green_s_min: int = 30,
    green_v_min: int = 30,
) -> tuple[list[float], Optional[np.ndarray]]:
    """Calculate normalized HSV histogram from a person crop.
    
    Supports two strategies:
    - "crop_torso": Extract centered torso region to minimize grass (default)
    - "mask_green": Mask out green pixels (grass) before computing histogram
    
    Args:
        crop: BGR image as numpy array (from OpenCV)
        h_bins: Number of bins for Hue channel (0-180 in OpenCV)
        s_bins: Number of bins for Saturation channel (0-255)
        v_bins: Number of bins for Value channel (0-255)
        torso_height_ratio: Proportion of crop height for torso region
        torso_width_ratio: Proportion of crop width for torso region
        hsv_strategy: "crop_torso" or "mask_green"
        green_h_range: Hue range for green grass (H ∈ [min, max])
        green_s_min: Minimum saturation for green detection
        green_v_min: Minimum value for green detection
        
    Returns:
        Tuple of (histogram as 1D list, transparency mask as BGRA crop or None)
    """
    if crop.size == 0:
        # Return zero vector for empty crops
        return [0.0] * (h_bins * s_bins * v_bins), None
    
    # Calculate torso region dimensions
    height, width = crop.shape[:2]
    
    # Select region based on strategy
    if hsv_strategy == "crop_torso":
        # Extract centered torso region (spatial cropping)
        torso_height = int(height * torso_height_ratio)
        torso_width = int(width * torso_width_ratio)
        
        # Ensure minimum dimensions
        if torso_height == 0:
            torso_height = 1
        if torso_width == 0:
            torso_width = 1
        
        # Center the torso region
        y_start = (height - torso_height) // 2
        x_start = (width - torso_width) // 2
        
        # Extract centered torso region
        region = crop[y_start:y_start + torso_height, x_start:x_start + torso_width]
    else:
        # Use full crop for mask_green strategy (masking instead of spatial cropping)
        region = crop
    
    # Convert to HSV
    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    
    mask = None
    crop_with_alpha = None
    
    if hsv_strategy == "mask_green":
        # Create mask to exclude green pixels (grass)
        lower_green = np.array([green_h_range[0], green_s_min, green_v_min])
        upper_green = np.array([green_h_range[1], 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Invert mask: 255 = keep (non-green), 0 = exclude (green)
        mask = cv2.bitwise_not(green_mask)
        
        # Create BGRA version with transparency for saved crop
        alpha_channel = mask.copy()
        crop_with_alpha = cv2.cvtColor(region, cv2.COLOR_BGR2BGRA)
        crop_with_alpha[:, :, 3] = alpha_channel
    
    # Calculate 3D histogram (with or without mask)
    hist = cv2.calcHist(
        [hsv],
        [0, 1, 2],  # H, S, V channels
        mask,  # None for crop_torso, mask for mask_green
        [h_bins, s_bins, v_bins],
        [0, 180, 0, 256, 0, 256]  # H range: 0-180, S/V range: 0-256
    )
    
    # Normalize histogram
    cv2.normalize(hist, hist)
    
    # Flatten to 1D list
    return hist.flatten().tolist(), crop_with_alpha


def _detect_referee_from_histogram(
    hsv_hist: list[float],
    *,
    h_bins: int = 8,
    s_bins: int = 4,
    v_bins: int = 4,
    yellow_h_range: tuple[int, int] = (20, 40),
    yellow_s_min: float = 0.6,
    yellow_v_min: float = 0.6,
    dark_v_max: float = 0.3,
    yellow_threshold: float = 0.30,
    dark_threshold: float = 0.15,
) -> bool:
    """Detect if a crop corresponds to a referee based on HSV histogram.
    
    Referees typically have:
    - Strong yellow in upper body (jersey): H ≈ 30 (yellow in OpenCV 0-180 scale)
    - Dark/black in lower body (shorts): Low V values
    
    Args:
        hsv_hist: Flattened HSV histogram vector
        h_bins: Number of hue bins used in histogram
        s_bins: Number of saturation bins used in histogram
        v_bins: Number of value bins used in histogram
        yellow_h_range: Hue range for yellow (H ∈ [min, max], OpenCV scale 0-180)
        yellow_s_min: Minimum normalized saturation for yellow (0-1)
        yellow_v_min: Minimum normalized value for yellow (0-1)
        dark_v_max: Maximum normalized value for dark colors (0-1)
        yellow_threshold: Minimum proportion of histogram that must be yellow (0-1)
        dark_threshold: Minimum proportion of histogram that must be dark (0-1)
        
    Returns:
        True if the crop is likely a referee, False otherwise
    """
    hist_array = np.array(hsv_hist)
    total_mass = hist_array.sum()
    
    if total_mass == 0:
        return False
    
    # Analyze each bin
    yellow_mass = 0.0
    dark_mass = 0.0
    
    for i, mass in enumerate(hist_array):
        # Decode bin index to h, s, v coordinates
        h_idx = i // (s_bins * v_bins)
        remaining = i % (s_bins * v_bins)
        s_idx = remaining // v_bins
        v_idx = remaining % v_bins
        
        # Convert bin indices to normalized HSV values (0-1)
        h_value = ((h_idx + 0.5) / h_bins) * 180  # Back to OpenCV 0-180 scale
        s_value = (s_idx + 0.5) / s_bins
        v_value = (v_idx + 0.5) / v_bins
        
        # Check if bin represents yellow
        if (yellow_h_range[0] <= h_value <= yellow_h_range[1] and
            s_value >= yellow_s_min and
            v_value >= yellow_v_min):
            yellow_mass += mass
        
        # Check if bin represents dark/black
        if v_value <= dark_v_max:
            dark_mass += mass
    
    # Calculate proportions
    yellow_proportion = yellow_mass / total_mass
    dark_proportion = dark_mass / total_mass
    
    # Referee detection: significant yellow AND significant dark
    return (yellow_proportion >= yellow_threshold and 
            dark_proportion >= dark_threshold)


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
    
    # Save figure (suppress tight_layout warning for colorbar axis)
    import warnings
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', message='This figure includes Axes that are not compatible with tight_layout')
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
        torso_height_ratio: float = 0.4,
        torso_width_ratio: float = 0.6,
        history_window_size: int = 24,
        hsv_strategy: str = "crop_torso",
    ) -> None:
        self.model_source = model_source
        self.device = device
        self.yolo_kwargs = {"classes": [0], **(yolo_kwargs or {})}
        self.video_fourcc = video_fourcc
        self.debug = debug
        self.output_dir = Path(output_dir) if output_dir else None
        self.torso_height_ratio = torso_height_ratio
        self.torso_width_ratio = torso_width_ratio
        self.history_window_size = history_window_size
        self.hsv_strategy = hsv_strategy
        self._model: Optional[YOLO] = None
        self._frame_history: list[dict[str, Any]] = []  # Circular buffer for last N frames
        
        # Team tracking state for persistent team identification across frames
        self._team_colors_bgr = self._get_colorblind_safe_palette()
        self._next_team_id = 1  # Counter for assigning new team IDs (0 reserved for referees)
        self._team_assignments: dict[int, int] = {}  # track_id -> team_id mapping

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

                # Use tracking mode to get persistent track IDs across frames
                results = self.model.track(frame, persist=True, verbose=False, **self.yolo_kwargs)
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
                        crop_filename = f"crop_original_{crop_idx:03d}.jpg"
                        
                        # Save crop image to disk if debug mode
                        if self.debug and frame_dir:
                            crop_path = frame_dir / crop_filename
                            cv2.imwrite(str(crop_path), crop)
                        
                        # Always compute HSV histogram for clustering
                        hsv_hist, crop_with_alpha = compute_hsv_histogram(
                            crop,
                            torso_height_ratio=self.torso_height_ratio,
                            torso_width_ratio=self.torso_width_ratio,
                            hsv_strategy=self.hsv_strategy,
                        )
                        meta["hsv_hist"] = hsv_hist
                        
                        # Detect referee based on HSV histogram (yellow + black pattern)
                        is_referee = _detect_referee_from_histogram(hsv_hist)
                        meta["role"] = "referee" if is_referee else "player"
                        
                        # Determine file extension based on strategy
                        crop_ext = ".png" if self.hsv_strategy == "mask_green" else ".jpg"
                        crop_filename = f"crop_{crop_idx:03d}{crop_ext}"
                        meta["filename"] = crop_filename
                        
                        # Save crop image to disk if debug mode
                        if self.debug and frame_dir:
                            crop_path = frame_dir / crop_filename
                            if crop_with_alpha is not None:
                                # Save PNG with transparency
                                cv2.imwrite(str(crop_path), crop_with_alpha)
                            else:
                                # Save regular JPG
                                cv2.imwrite(str(crop_path), crop)
                        
                        # Save histogram visualization in debug mode
                        if self.debug and frame_dir:
                            _save_histogram_visualization(
                                hsv_hist,
                                frame_dir / f"crop_{crop_idx:04d}_hist.png"
                            )
                        
                        frame_crops.append(meta)
                    
                    # Separate referees from players before clustering
                    player_crops = [crop for crop in frame_crops if crop.get("role") == "player"]
                    referee_crops = [crop for crop in frame_crops if crop.get("role") == "referee"]
                    
                    # Perform clustering only on players
                    from .clustering import cluster_single_frame
                    
                    cluster_labels = []
                    clustering_info = {}
                    
                    if player_crops:
                        cluster_labels, clustering_info = cluster_single_frame(
                            player_crops,
                            k=2,  # Fixed K=2 for two teams
                            output_dir=self.output_dir,
                            frame_key=frame_key,
                            debug_plots=self.debug,
                        )
                    
                    # Assign cluster IDs to player crops
                    player_idx = 0
                    for crop in frame_crops:
                        if crop.get("role") == "player":
                            if player_idx < len(cluster_labels):
                                crop["cluster_id"] = cluster_labels[player_idx]
                                player_idx += 1
                            else:
                                crop["cluster_id"] = 0
                        else:
                            # Referees don't have cluster_id (will get team_id=0 later)
                            crop["cluster_id"] = -1
                    
                    # Assign persistent team IDs based on tracking history
                    self._assign_team_ids(frame_crops)
                    
                    # Update frame history circular buffer
                    self._update_frame_history(frame_idx, frame_crops)
                    
                    # Generate and write output frame with cluster colors
                    output_frame = None
                    if output_writer is not None:
                        output_frame = self._draw_cluster_boxes(
                            frame.copy(),
                            frame_crops,
                            clustering_info.get("k", 2)
                        )
                        output_writer.write(output_frame)
                        
                        # Save team_id frame in debug mode (after voting)
                        if self.debug and frame_dir:
                            team_id_filename = f"frame_{frame_idx:04d}_team_id.jpg"
                            team_id_path = frame_dir / team_id_filename
                            cv2.imwrite(str(team_id_path), output_frame)
                        
                        # Save cluster_id frame in debug mode (pure K-Means, before voting)
                        if self.debug and frame_dir:
                            cluster_frame = self._draw_cluster_boxes_kmeans(
                                frame.copy(),
                                frame_crops,
                                clustering_info.get("k", 2)
                            )
                            cluster_id_filename = f"frame_{frame_idx:04d}_cluster_id.jpg"
                            cluster_id_path = frame_dir / cluster_id_filename
                            cv2.imwrite(str(cluster_id_path), cluster_frame)
                    
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

            # Save frame history to disk (debug only)
            if self.debug and self.output_dir:
                self._save_frame_history()

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
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
        
        # Ensure output directory exists
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
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
        """Draw team-colored bounding boxes on frame.
        
        Args:
            frame: Original frame to draw on
            crops_data: List of crop metadata with bbox, team_id, and track_id
            max_k: Maximum number of clusters (unused, kept for compatibility)
            
        Returns:
            Frame with drawn bounding boxes colored by team_id
        """
        # Draw team-colored rectangles
        for crop_meta in crops_data:
            bbox = crop_meta["bbox"]
            team_id = crop_meta.get("team_id", 0)
            track_id = crop_meta.get("track_id")
            role = crop_meta.get("role", "player")
            
            x1, y1, x2, y2 = bbox
            
            # Special handling for referees: gray color
            if role == "referee" or team_id == 0:
                color = (128, 128, 128)  # Gray in BGR
                label = "REF"
            else:
                # Use colorblind-safe palette based on team_id
                # Offset by -1 since team_id starts at 1 for players
                palette_idx = (team_id - 1) % len(self._team_colors_bgr)
                color = self._team_colors_bgr[palette_idx]
                # Draw track ID label (or team ID if track not available)
                label = f"{track_id}" if track_id is not None else f"T{team_id}"
            
            # Draw rectangle with team color (thickness 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
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
    
    def _draw_cluster_boxes_kmeans(
        self,
        frame: np.ndarray,
        crops_data: list[dict[str, Any]],
        max_k: int,
    ) -> np.ndarray:
        """Draw cluster-colored bounding boxes on frame (pure K-Means, no voting).
        
        Uses cluster_id directly from K-Means output, marking with asterisk (*)
        those crops whose cluster differs from their voted team_id (in transition).
        
        Args:
            frame: Original frame to draw on
            crops_data: List of crop metadata with bbox, cluster_id, team_id, and track_id
            max_k: Maximum number of clusters (unused, kept for compatibility)
            
        Returns:
            Frame with drawn bounding boxes colored by cluster_id
        """
        # Draw cluster-colored rectangles
        for crop_meta in crops_data:
            bbox = crop_meta["bbox"]
            cluster_id = crop_meta.get("cluster_id", 0)
            team_id = crop_meta.get("team_id", 0)
            track_id = crop_meta.get("track_id")
            role = crop_meta.get("role", "player")
            
            x1, y1, x2, y2 = bbox
            
            # Special handling for referees: gray color
            if role == "referee" or cluster_id == -1:
                color = (128, 128, 128)  # Gray in BGR
                label = "REF"
            else:
                # Use colorblind-safe palette based on cluster_id
                palette_idx = cluster_id % len(self._team_colors_bgr)
                color = self._team_colors_bgr[palette_idx]
                # Draw track ID label
                label = f"{track_id}" if track_id is not None else f"C{cluster_id}"
                
                # Mark with asterisk if cluster_id differs from team_id (in transition)
                # team_id is 1-indexed, so cluster_id should be team_id - 1 when stable
                if cluster_id != team_id - 1:
                    label += "*"
            
            # Draw rectangle with cluster color (thickness 1)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 1)
            
            label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 1)
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

    def _get_colorblind_safe_palette(self) -> list[tuple[int, int, int]]:
        """Get colorblind-safe palette for team identification (BGR format).
        
        Returns 5 distinct colors optimized for deuteranopia/protanopia:
        - Blue, Orange, Yellow, Purple, Pink
        
        Returns:
            List of 5 BGR color tuples
        """
        # Colors chosen for maximum distinction with colorblindness
        colors_rgb = [
            (0, 114, 178),    # Blue
            (230, 159, 0),    # Orange  
            (240, 228, 66),   # Yellow
            (204, 121, 167),  # Pink/Magenta
            (0, 158, 115),    # Teal/Green
        ]
        # Convert RGB to BGR for OpenCV
        return [(rgb[2], rgb[1], rgb[0]) for rgb in colors_rgb]
    
    def _assign_team_ids(self, crops: list[dict[str, Any]]) -> None:
        """Assign persistent team_id to each crop based on tracking history.
        
        Logic:
        - Referees: Always get team_id=0 (reserved)
        - Players, first frame: cluster_id → team_id (1:1 mapping, starting from 1)
        - Players, subsequent frames: voting across last 7 frames (70% consensus)
        - New track_ids: inherit team_id from cluster's dominant team
        
        Modifies crops in-place by adding 'team_id' field.
        
        Args:
            crops: List of crop dicts with track_id, cluster_id, and role
        """
        # First pass: assign team_id=0 to all referees
        for crop in crops:
            if crop.get("role") == "referee":
                crop["team_id"] = 0
                track_id = crop.get("track_id")
                if track_id is not None:
                    self._team_assignments[track_id] = 0
        
        # Second pass: assign team_ids to players
        if len(self._frame_history) == 0:
            # First frame: direct mapping cluster_id → team_id (starting from 1)
            for crop in crops:
                if crop.get("role") == "referee":
                    continue  # Already handled
                
                cluster_id = crop.get("cluster_id", 0)
                track_id = crop.get("track_id")
                
                # Assign team_id = cluster_id + 1 (to avoid team_id=0 reserved for referees)
                team_id = cluster_id + 1
                crop["team_id"] = team_id
                
                # Store mapping if we have track_id
                if track_id is not None:
                    self._team_assignments[track_id] = team_id
                    self._next_team_id = max(self._next_team_id, team_id + 1)
        else:
            # Subsequent frames: use voting logic for players only
            for crop in crops:
                if crop.get("role") == "referee":
                    continue  # Already handled
                
                track_id = crop.get("track_id")
                cluster_id = crop.get("cluster_id", 0)
                
                if track_id is None:
                    # No tracking info, fallback to cluster_id + 1
                    crop["team_id"] = cluster_id + 1
                    continue
                
                # Check if this track_id has history
                if track_id in self._team_assignments:
                    # Vote based on last 7 frames
                    team_id = self._vote_team_id(track_id, cluster_id)
                    crop["team_id"] = team_id
                    self._team_assignments[track_id] = team_id
                else:
                    # New track_id: inherit from cluster's dominant team
                    team_id = self._get_cluster_dominant_team(cluster_id, crops)
                    crop["team_id"] = team_id
                    self._team_assignments[track_id] = team_id
    
    def _vote_team_id(self, track_id: int, current_cluster_id: int) -> int:
        """Vote for team_id based on REAL cluster_id history (not voted team_id).
        
        IMPORTANT: Votes based on the actual cluster_id from K-Means, not the
        team_id that was assigned after voting. This ensures that if K-Means
        consistently says a player changed teams, the change will eventually happen.
        
        Args:
            track_id: YOLO tracking ID
            current_cluster_id: Current frame's cluster assignment from K-Means
            
        Returns:
            team_id based on cluster_id consensus over last 7 frames
        """
        # Look at last 7 frames for voting
        voting_window = min(7, len(self._frame_history))
        recent_frames = self._frame_history[-voting_window:]
        
        # Count CLUSTER_ID occurrences (real K-Means output), not team_id
        cluster_votes: dict[int, int] = {}
        total_appearances = 0
        
        for frame_data in recent_frames:
            for crop_data in frame_data["crops"]:
                if crop_data.get("track_id") == track_id:
                    # Vote on REAL cluster_id from K-Means, not voted team_id
                    cluster_id = crop_data.get("cluster_id", 0)
                    if cluster_id >= 0:  # Exclude referees (cluster_id=-1)
                        cluster_votes[cluster_id] = cluster_votes.get(cluster_id, 0) + 1
                        total_appearances += 1
        
        if total_appearances == 0:
            # No history, use current cluster as team (offset by +1 for players)
            return current_cluster_id + 1
        
        # Find cluster with most votes (majority wins)
        winning_cluster = max(cluster_votes.items(), key=lambda x: x[1])
        consensus_ratio = winning_cluster[1] / total_appearances
        
        # Simple majority rule: if winning cluster has >50%, use it
        # This allows changes when there's clear evidence from K-Means
        if consensus_ratio > 0.50:
            # Majority agrees on this cluster, assign corresponding team_id
            return winning_cluster[0] + 1
        
        # No clear majority, keep current assignment if available
        current_team_id = self._team_assignments.get(track_id)
        return current_team_id if current_team_id is not None else current_cluster_id + 1
    
    def _get_cluster_dominant_team(self, cluster_id: int, current_crops: list[dict[str, Any]]) -> int:
        """Find dominant team_id among crops in the same cluster.
        
        For new track_ids, inherit team from their cluster's existing members.
        
        Args:
            cluster_id: Cluster ID to analyze
            current_crops: Current frame's crops being processed
            
        Returns:
            Dominant team_id for this cluster, or cluster_id as fallback
        """
        # Look at current frame's crops with same cluster (exclude referees)
        team_counts: dict[int, int] = {}
        
        for crop in current_crops:
            if (crop.get("cluster_id") == cluster_id and 
                "team_id" in crop and 
                crop.get("role") == "player"):
                team_id = crop["team_id"]
                team_counts[team_id] = team_counts.get(team_id, 0) + 1
        
        if team_counts:
            return max(team_counts.items(), key=lambda x: x[1])[0]
        
        # No existing team in this cluster, assign new team_id
        new_team_id = self._next_team_id
        self._next_team_id += 1
        return new_team_id

    def _update_frame_history(self, frame_idx: int, crops: list[dict[str, Any]]) -> None:
        """Update circular buffer with current frame's crop data.
        
        Maintains a FIFO buffer of the last N frames, each containing:
        - frame_idx: Frame number
        - crops: List of dicts with bbox, track_id, cluster_id, and team_id
        
        Args:
            frame_idx: Current frame index
            crops: List of crop metadata (must include bbox, track_id, cluster_id, team_id)
        """
        # Extract only the essential fields for history
        simplified_crops = []
        for crop in crops:
            simplified_crops.append({
                "bbox": crop["bbox"],
                "track_id": crop.get("track_id"),
                "cluster_id": crop.get("cluster_id", 0),
                "team_id": crop.get("team_id", 0),
                "role": crop.get("role", "player"),
            })
        
        # Add current frame to history
        self._frame_history.append({
            "frame_idx": frame_idx,
            "crops": simplified_crops,
        })
        
        # Remove oldest frame if we exceed the window size (FIFO)
        if len(self._frame_history) > self.history_window_size:
            self._frame_history.pop(0)
    
    def _save_frame_history(self) -> None:
        """Save the circular buffer to disk for debugging purposes.
        
        Writes a JSON file named 'temporal_frame_history.json' containing
        the last N frames stored in the circular buffer.
        """
        if not self.output_dir:
            return
        
        history_path = self.output_dir / "temporal_frame_history.json"
        history_data = {
            "window_size": self.history_window_size,
            "frames_in_buffer": len(self._frame_history),
            "history": self._frame_history,
        }
        
        with open(history_path, "w", encoding="utf-8") as f:
            json.dump(history_data, f, indent=2)
        
        print(f"Frame history saved to: {history_path}")


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
        
        # Extract track ID if available (from tracking mode)
        track_id = None
        if hasattr(boxes, 'id') and boxes.id is not None and len(boxes.id) > idx:
            track_id = int(boxes.id[idx].cpu().item() if hasattr(boxes.id[idx], "cpu") else boxes.id[idx])
        
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
            "track_id": track_id,
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
    hsv_strategy: str = "crop_torso",
) -> Iterator[Any]:
    """Process video with YOLO detection and team clustering, generating labeled output video.
    
    Uses K-Means with K=2 to identify two teams, plus automatic referee detection.
    
    Args:
        input_path: Path to input video file
        output_dir: Directory where output-video.mp4 will be saved (and debug files if debug=True)
        frames: Optional frame index (int) or list of frame indices to process. If None, process all frames.
        model_source: YOLO model weights to load
        device: Device to run inference on
        yolo_kwargs: Additional kwargs for YOLO inference
        video_fourcc: Video codec fourcc code
        debug: If True, save crops, histograms, clustering plots, yolo-video.mp4, and metadata to disk
        hsv_strategy: HSV histogram strategy - "crop_torso" or "mask_green" (default: "crop_torso")
    """
    output_dir_path = Path(output_dir)

    runner = YoloVideoRunner(
        model_source=model_source,
        device=device,
        yolo_kwargs=yolo_kwargs,
        video_fourcc=video_fourcc,
        debug=debug,
        output_dir=output_dir_path,
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
