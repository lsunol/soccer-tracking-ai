"""Clustering helpers for grouping players by jersey colors."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from .models import ClusterResult, ClusteringSummary, CropSample


class TeamClusterer:
    """K-Means based clustering component with optional debug visualizations."""

    def __init__(
        self,
        *,
        n_clusters: int = 2,
        random_state: int = 42,
        n_init: int = 10,
    ) -> None:
        self.n_clusters = n_clusters
        self._kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=n_init)
        self._scaler = StandardScaler()

    def cluster(
        self,
        crops_data: Sequence[CropSample] | Sequence[dict[str, object]],
        *,
        output_dir: Path | None = None,
        frame_key: str | None = None,
        debug_plots: bool = False,
    ) -> ClusterResult:
        """Run clustering on the provided crops and optionally emit debug plots."""
        feature_matrix = self._build_feature_matrix(crops_data)
        num_crops = len(feature_matrix)

        if num_crops == 0:
            return ClusterResult(labels=[], summary=ClusteringSummary(k=0, inertia=0.0))

        if num_crops < self.n_clusters:
            labels = [0] * num_crops
            summary = ClusteringSummary(k=1, inertia=0.0, cluster_distribution={"0": num_crops})
            return ClusterResult(labels=labels, summary=summary)

        feature_matrix_scaled = self._scaler.fit_transform(feature_matrix)
        self._kmeans.fit(feature_matrix_scaled)
        labels = self._kmeans.labels_.tolist()

        distribution: dict[str, int] = {}
        for label in labels:
            distribution[str(label)] = distribution.get(str(label), 0) + 1

        summary = ClusteringSummary(
            k=self.n_clusters,
            inertia=float(self._kmeans.inertia_),
            cluster_distribution=distribution,
        )

        if debug_plots and output_dir and frame_key:
            frame_dir = output_dir / frame_key
            frame_dir.mkdir(parents=True, exist_ok=True)
            _visualize_clusters(frame_dir, feature_matrix_scaled, labels, self.n_clusters)

        return ClusterResult(labels=labels, summary=summary)

    @staticmethod
    def _build_feature_matrix(
        crops_data: Sequence[CropSample] | Sequence[dict[str, object]]
    ) -> np.ndarray:
        """Extract HSV histograms from either CropSample objects or metadata dicts."""
        features: list[list[float]] = []
        for crop in crops_data:
            if isinstance(crop, CropSample):
                features.append(crop.hsv_hist)
            else:
                hist = crop.get("hsv_hist")  # type: ignore[assignment]
                if isinstance(hist, Iterable):
                    features.append(list(hist))
        return np.asarray(features, dtype=np.float32)


def cluster_single_frame(
    crops_data: Sequence[CropSample] | Sequence[dict[str, object]],
    k: int = 2,
    output_dir: Path | None = None,
    frame_key: str | None = None,
    debug_plots: bool = False,
) -> tuple[list[int], dict[str, int | float | dict[str, int]]]:
    """Backward compatible helper that delegates to :class:`TeamClusterer`."""

    clusterer = TeamClusterer(n_clusters=k)
    result = clusterer.cluster(
        crops_data,
        output_dir=output_dir,
        frame_key=frame_key,
        debug_plots=debug_plots,
    )
    info: dict[str, int | float | dict[str, int]] = {
        "k": result.summary.k,
        "inertia": result.summary.inertia,
        "cluster_distribution": result.summary.cluster_distribution,
    }
    return result.labels, info


def _visualize_clusters(
    output_dir: Path,
    feature_matrix: np.ndarray,
    cluster_labels: Sequence[int],
    n_clusters: int,
) -> None:
    """Project clusters via PCA and persist the image for debugging purposes."""

    fig, ax = plt.subplots(figsize=(10, 8))
    pca = PCA(n_components=2, random_state=42)
    coords_pca = pca.fit_transform(feature_matrix)
    colors = plt.cm.tab10.colors[:n_clusters]
    cmap_discrete = matplotlib.colors.ListedColormap(colors)

    scatter = ax.scatter(
        coords_pca[:, 0],
        coords_pca[:, 1],
        c=cluster_labels,
        cmap=cmap_discrete,
        s=100,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
        vmin=-0.5,
        vmax=n_clusters - 0.5,
    )

    for idx, (x_coord, y_coord) in enumerate(coords_pca):
        ax.annotate(
            str(idx + 1),
            (x_coord, y_coord),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=8,
            fontweight="bold",
            color="black",
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="none", alpha=0.15),
        )

    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=11)
    ax.set_title(f"PCA Projection (K={n_clusters} clusters)", fontsize=13, fontweight="bold")
    ax.grid(alpha=0.3)

    cbar = plt.colorbar(scatter, ax=ax, ticks=range(n_clusters), boundaries=np.arange(-0.5, n_clusters, 1))
    cbar.set_label("Cluster ID", fontsize=11)
    cbar.ax.set_yticklabels([str(i) for i in range(n_clusters)])

    plt.tight_layout()
    pca_path = output_dir / "clusters_pca.png"
    plt.savefig(str(pca_path), dpi=150, bbox_inches="tight")
    plt.close(fig)
