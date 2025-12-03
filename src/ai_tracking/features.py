"""Feature extraction utilities for player crops."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import cv2
import numpy as np


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
    """Return a normalized HSV histogram for a person crop.

    Args:
        crop: BGR image containing the detection crop.
        h_bins: Number of histogram bins for the hue channel.
        s_bins: Number of histogram bins for the saturation channel.
        v_bins: Number of histogram bins for the value channel.
        torso_height_ratio: Fraction of the crop height used for the torso window.
        torso_width_ratio: Fraction of the crop width used for the torso window.
        hsv_strategy: Either ``"crop_torso"`` or ``"mask_green"`` to select the ROI logic.
        green_h_range: Inclusive hue bounds considered grass when masking.
        green_s_min: Minimum saturation value used to detect grass pixels.
        green_v_min: Minimum brightness value used to detect grass pixels.

    Returns:
        Tuple with the flattened histogram and an optional BGRA crop with transparency when masking.
    """
    if crop.size == 0:
        return [0.0] * (h_bins * s_bins * v_bins), None

    height, width = crop.shape[:2]
    region = crop

    if hsv_strategy == "crop_torso":
        torso_height = max(1, int(height * torso_height_ratio))
        torso_width = max(1, int(width * torso_width_ratio))
        y_start = max(0, (height - torso_height) // 2)
        x_start = max(0, (width - torso_width) // 2)
        region = crop[y_start : y_start + torso_height, x_start : x_start + torso_width]

    hsv = cv2.cvtColor(region, cv2.COLOR_BGR2HSV)
    mask = None
    crop_with_alpha = None

    if hsv_strategy == "mask_green":
        lower_green = np.array([green_h_range[0], green_s_min, green_v_min])
        upper_green = np.array([green_h_range[1], 255, 255])
        green_mask = cv2.inRange(hsv, lower_green, upper_green)
        mask = cv2.bitwise_not(green_mask)
        crop_with_alpha = cv2.cvtColor(region, cv2.COLOR_BGR2BGRA)
        crop_with_alpha[:, :, 3] = mask

    histogram = cv2.calcHist(
        [hsv],
        [0, 1, 2],
        mask,
        [h_bins, s_bins, v_bins],
        [0, 180, 0, 256, 0, 256],
    )
    cv2.normalize(histogram, histogram)
    return histogram.flatten().tolist(), crop_with_alpha


def detect_referee_from_histogram(
    hsv_hist: Sequence[float],
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
    """Return ``True`` when the histogram matches the yellow/black referee pattern."""
    hist_array = np.asarray(hsv_hist, dtype=np.float32)
    total_mass = hist_array.sum()
    if total_mass <= 0:
        return False

    yellow_mass = 0.0
    dark_mass = 0.0

    for idx, mass in enumerate(hist_array):
        h_idx = idx // (s_bins * v_bins)
        remaining = idx % (s_bins * v_bins)
        s_idx = remaining // v_bins
        v_idx = remaining % v_bins

        hue_value = ((h_idx + 0.5) / h_bins) * 180
        saturation = (s_idx + 0.5) / s_bins
        value = (v_idx + 0.5) / v_bins

        if (
            yellow_h_range[0] <= hue_value <= yellow_h_range[1]
            and saturation >= yellow_s_min
            and value >= yellow_v_min
        ):
            yellow_mass += float(mass)

        if value <= dark_v_max:
            dark_mass += float(mass)

    yellow_ratio = yellow_mass / total_mass
    dark_ratio = dark_mass / total_mass
    return yellow_ratio >= yellow_threshold and dark_ratio >= dark_threshold


def save_histogram_visualization(
    histogram: Sequence[float],
    output_path: Path,
    *,
    h_bins: int = 8,
    s_bins: int = 4,
    v_bins: int = 4,
) -> None:
    """Persist a colored bar chart representation of an HSV histogram."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.colors import hsv_to_rgb

    colors: list[tuple[float, float, float]] = []
    for idx in range(len(histogram)):
        h_idx = idx // (s_bins * v_bins)
        remaining = idx % (s_bins * v_bins)
        s_idx = remaining // v_bins
        v_idx = remaining % v_bins

        h_norm = (h_idx + 0.5) / h_bins
        s_norm = (s_idx + 0.5) / s_bins
        v_norm = (v_idx + 0.5) / v_bins
        colors.append(tuple(hsv_to_rgb([h_norm, s_norm, v_norm])))

    fig, (ax_hist, ax_color) = plt.subplots(
        2,
        1,
        figsize=(12, 5),
        gridspec_kw={"height_ratios": [4, 0.3], "hspace": 0.15},
    )

    x = np.arange(len(histogram))
    ax_hist.bar(x, histogram, width=1.0, color=colors, edgecolor="black", linewidth=0.3)
    ax_hist.set_ylabel("Normalized Frequency", fontsize=10)
    ax_hist.set_title(
        f"HSV Histogram ({h_bins}H × {s_bins}S × {v_bins}V)",
        fontsize=11,
    )
    ax_hist.grid(axis="y", alpha=0.3)
    ax_hist.set_xlim(-0.5, len(histogram) - 0.5)

    color_array = np.asarray(colors).reshape(1, -1, 3)
    ax_color.imshow(color_array, aspect="auto", extent=[0, len(histogram), 0, 1])
    ax_color.set_yticks([])
    ax_color.set_xlabel("Histogram Bin Index", fontsize=10)
    ax_color.set_xlim(0, len(histogram))

    for h in range(1, h_bins):
        boundary = h * s_bins * v_bins
        ax_hist.axvline(boundary - 0.5, color="white", linewidth=2, linestyle="--", alpha=0.7)
        ax_color.axvline(boundary, color="white", linewidth=2, linestyle="--", alpha=0.7)

    import warnings

    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message="This figure includes Axes that are not compatible with tight_layout",
        )
        plt.tight_layout()
    plt.savefig(str(output_path), dpi=100, bbox_inches="tight")
    plt.close(fig)
