"""Tests for the TeamClusterer component."""

from __future__ import annotations

from ai_tracking.clustering import TeamClusterer
from ai_tracking.models import BoundingBox, CropSample, Detection


def _make_sample(value: float) -> CropSample:
	bbox = BoundingBox(0, 0, 10, 10)
	detection = Detection(bbox=bbox, confidence=0.9, class_id=0, class_name="person")
	histogram = [value, 1 - value] + [0.0] * (8 * 4 * 4 - 2)
	return CropSample(detection=detection, hsv_hist=histogram)


def test_team_clusterer_assigns_distinct_labels() -> None:
	samples = [_make_sample(0.9), _make_sample(0.1)]
	clusterer = TeamClusterer(n_clusters=2)
	result = clusterer.cluster(samples)
	assert set(result.labels) == {0, 1}
	assert result.summary.k == 2


def test_team_clusterer_handles_insufficient_samples() -> None:
	samples = [_make_sample(0.8)]
	clusterer = TeamClusterer(n_clusters=2)
	result = clusterer.cluster(samples)
	assert result.labels == [0]
	assert result.summary.k == 1
