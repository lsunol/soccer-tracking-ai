"""Logic for mapping cluster outputs to persistent team identifiers."""

from __future__ import annotations

import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Deque, Dict, List

from .models import CropSample


@dataclass(slots=True)
class _TrackSnapshot:
	track_id: int | None
	cluster_id: int
	role: str


class TeamAssignmentManager:
	"""Assign consistent team identifiers leveraging short-term history."""

	def __init__(self, *, history_window_size: int = 24) -> None:
		self.history_window_size = history_window_size
		self._history: Deque[List[_TrackSnapshot]] = deque(maxlen=history_window_size)
		self._team_assignments: Dict[int, int] = {}
		self._next_team_id = 1

	def assign(self, crops: list[CropSample]) -> None:
		"""Mutate ``crops`` assigning ``team_id`` according to the history window."""
		for crop in crops:
			if crop.role == "referee":
				crop.team_id = 0
				if crop.detection.track_id is not None:
					self._team_assignments[crop.detection.track_id] = 0

		if not self._history:
			self._assign_first_frame(crops)
		else:
			self._assign_subsequent_frames(crops)

		self._update_history(crops)

	def export_history(self, output_dir: Path | None) -> None:
		"""Persist the temporal buffer for offline inspection."""
		if output_dir is None or not self._history:
			return
		history_path = output_dir / "temporal_frame_history.json"
		payload = {
			"window_size": self.history_window_size,
			"frames_in_buffer": len(self._history),
			"history": [
				[
					{
						"track_id": snap.track_id,
						"cluster_id": snap.cluster_id,
						"role": snap.role,
					}
					for snap in frame
				]
				for frame in self._history
			],
		}
		with open(history_path, "w", encoding="utf-8") as handle:
			json.dump(payload, handle, indent=2)

	def _assign_first_frame(self, crops: list[CropSample]) -> None:
		for crop in crops:
			if crop.role == "referee":
				continue
			cluster_id = crop.cluster_id or 0
			team_id = cluster_id + 1
			crop.team_id = team_id
			track_id = crop.detection.track_id
			if track_id is not None:
				self._team_assignments[track_id] = team_id
				self._next_team_id = max(self._next_team_id, team_id + 1)

	def _assign_subsequent_frames(self, crops: list[CropSample]) -> None:
		for crop in crops:
			if crop.role == "referee":
				continue
			cluster_id = crop.cluster_id or 0
			track_id = crop.detection.track_id
			if track_id is None:
				crop.team_id = cluster_id + 1
				continue
			if track_id in self._team_assignments:
				team_id = self._vote_team_id(track_id, cluster_id)
				crop.team_id = team_id
				self._team_assignments[track_id] = team_id
			else:
				team_id = self._get_cluster_dominant_team(cluster_id, crops)
				crop.team_id = team_id
				self._team_assignments[track_id] = team_id

	def _vote_team_id(self, track_id: int, current_cluster_id: int) -> int:
		voting_frames = list(self._history)[-7:]
		votes: Dict[int, int] = {}
		total = 0
		for frame in voting_frames:
			for snap in frame:
				if snap.track_id == track_id and snap.role != "referee" and snap.cluster_id >= 0:
					votes[snap.cluster_id] = votes.get(snap.cluster_id, 0) + 1
					total += 1
		if total == 0:
			return current_cluster_id + 1
		cluster_id, count = max(votes.items(), key=lambda item: item[1])
		if count / total > 0.50:
			return cluster_id + 1
		return self._team_assignments.get(track_id, current_cluster_id + 1)

	def _get_cluster_dominant_team(self, cluster_id: int, crops: list[CropSample]) -> int:
		team_counts: Dict[int, int] = {}
		for crop in crops:
			if crop.cluster_id == cluster_id and crop.team_id and crop.role == "player":
				team_counts[crop.team_id] = team_counts.get(crop.team_id, 0) + 1
		if team_counts:
			return max(team_counts.items(), key=lambda item: item[1])[0]
		team_id = self._next_team_id
		self._next_team_id += 1
		return team_id

	def _update_history(self, crops: list[CropSample]) -> None:
		snapshot: list[_TrackSnapshot] = []
		for crop in crops:
			snapshot.append(
				_TrackSnapshot(
					track_id=crop.detection.track_id,
					cluster_id=crop.cluster_id or -1,
					role=crop.role,
				)
			)
		self._history.append(snapshot)
