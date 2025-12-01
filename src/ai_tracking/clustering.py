"""Unsupervised clustering of person crops based on HSV histogram features."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def cluster_single_frame(
    crops_data: list[dict[str, Any]],
    k: int = 2,
    output_dir: Optional[Path] = None,
    frame_key: Optional[str] = None,
    debug_plots: bool = False,
) -> tuple[list[int], dict[str, Any]]:
    """Execute K-Means clustering on a single frame's crops with K=2 (two teams).
    
    Args:
        crops_data: List of crop metadata dicts with hsv_hist field
        k: Number of clusters (fixed at 2 for two teams)
        output_dir: Directory to save debug visualizations (frame subdir)
        frame_key: Frame identifier for debug output
        debug_plots: Whether to generate debug visualizations
        
    Returns:
        Tuple of (cluster_labels, clustering_info_dict)
    """
    num_crops = len(crops_data)
    
    # Handle edge cases
    if num_crops == 0:
        return [], {}
    
    if num_crops < k:
        # Not enough crops for K clusters, assign all to cluster 0
        labels = [0] * num_crops
        info = {
            "k": 1,
            "inertia": 0.0,
            "cluster_distribution": {"0": num_crops}
        }
        return labels, info
    
    # Extract feature matrix
    feature_matrix = np.array([crop["hsv_hist"] for crop in crops_data])
    
    # Normalize features
    scaler = StandardScaler()
    feature_matrix_scaled = scaler.fit_transform(feature_matrix)
    
    # Run K-Means with K=2
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(feature_matrix_scaled)
    
    cluster_labels = kmeans.labels_.tolist()
    
    # Build clustering info
    cluster_distribution = {}
    for label in cluster_labels:
        cluster_distribution[str(label)] = cluster_distribution.get(str(label), 0) + 1
    
    clustering_info = {
        "k": k,
        "inertia": float(kmeans.inertia_),
        "cluster_distribution": cluster_distribution,
    }
    
    # Generate debug plots if requested
    if debug_plots and output_dir and frame_key:
        frame_dir = output_dir / frame_key
        frame_dir.mkdir(parents=True, exist_ok=True)
        
        # Save PCA visualization
        _visualize_clusters(
            feature_matrix_scaled,
            np.array(cluster_labels),
            k,
            frame_dir,
        )
    
    return cluster_labels, clustering_info


def _visualize_clusters(
    feature_matrix: np.ndarray,
    cluster_labels: np.ndarray,
    k: int,
    output_dir: Path,
) -> None:
    """Generate PCA visualization of clusters.
    
    Args:
        feature_matrix: Scaled feature matrix (num_crops, num_features)
        cluster_labels: Cluster assignments for each crop
        k: Number of clusters
        output_dir: Frame directory to save visualizations
    """
    # PCA projection (2D)
    pca = PCA(n_components=2, random_state=42)
    coords_pca = pca.fit_transform(feature_matrix)
    
    # Create discrete colormap with exactly K colors (colorblind-friendly)
    colors = plt.cm.tab10.colors[:k]
    cmap_discrete = matplotlib.colors.ListedColormap(colors)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        coords_pca[:, 0],
        coords_pca[:, 1],
        c=cluster_labels,
        cmap=cmap_discrete,
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5,
        vmin=-0.5,
        vmax=k - 0.5
    )
    
    # Annotate each point with its crop number (1-indexed)
    for i, (x, y) in enumerate(coords_pca):
        ax.annotate(
            str(i + 1),  # Crop number (1-indexed)
            (x, y),
            xytext=(5, 5),  # Offset from point
            textcoords='offset points',
            fontsize=8,
            fontweight='bold',
            color='black',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor='none', alpha=0.15)
        )
    
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
    ax.set_title(f'PCA Projection (K={k} clusters)', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add discrete colorbar with exactly K colors
    cbar = plt.colorbar(scatter, ax=ax, ticks=range(k), boundaries=np.arange(-0.5, k, 1))
    cbar.set_label('Cluster ID', fontsize=11)
    cbar.ax.set_yticklabels([str(i) for i in range(k)])
    
    plt.tight_layout()
    pca_path = output_dir / "clusters_pca.png"
    plt.savefig(str(pca_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
