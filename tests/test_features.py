"""Unit tests for HSV feature utilities."""

from __future__ import annotations

import numpy as np

from ai_tracking.features import compute_hsv_histogram, detect_referee_from_histogram


def test_compute_hsv_histogram_returns_expected_bin_count() -> None:
	crop = np.zeros((32, 32, 3), dtype=np.uint8)
	hist, mask = compute_hsv_histogram(crop)
	assert len(hist) == 8 * 4 * 4
	assert mask is None


def test_detect_referee_from_histogram_identifies_yellow_black_pattern() -> None:
	bins = 8 * 4 * 4
	histogram = [0.0] * bins
	yellow_idx = (1 * 4 * 4) + (3 * 4) + 3  # High saturation/value bin
	dark_idx = (0 * 4 * 4) + (0 * 4) + 0  # Dark bin
	histogram[yellow_idx] = 0.5
	histogram[dark_idx] = 0.5
	assert detect_referee_from_histogram(histogram)


def test_detect_referee_from_histogram_rejects_non_referee_pattern() -> None:
	histogram = [1.0] + [0.0] * (8 * 4 * 4 - 1)
	assert not detect_referee_from_histogram(histogram)
