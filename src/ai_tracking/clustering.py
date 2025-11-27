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
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False


def run_hsv_clustering(
    metadata: dict[str, Any],
    k_min: int = 2,
    k_max: int = 5,
    output_dir: Optional[str | Path] = None,
    debug_plots: bool = True,
) -> dict[str, Any]:
    """Execute K-Means clustering on HSV histograms with elbow method for K selection.
    
    Performs clustering independently for each frame, applies elbow method to select
    optimal K, and updates metadata with cluster assignments.
    
    Args:
        metadata: Crops metadata dict with hsv_hist features
        k_min: Minimum number of clusters to try
        k_max: Maximum number of clusters to try
        output_dir: Directory to save visualizations and reports
        debug_plots: Whether to generate PCA/UMAP visualizations
        
    Returns:
        Updated metadata dict with cluster_id field added to each crop
    """
    output_path = Path(output_dir) if output_dir else None
    
    # Create clustering subdirectory for outputs
    if output_path:
        clustering_dir = output_path / "clustering"
        clustering_dir.mkdir(exist_ok=True)
    else:
        clustering_dir = None
    
    clustering_report = {
        "k_range": [k_min, k_max],
        "frames": {}
    }
    
    # Process each frame independently
    for frame_key, frame_data in metadata["frames"].items():
        print(f"Clustering {frame_key} ({frame_data['num_crops']} crops)...")
        
        # Extract feature matrix
        feature_matrix, crop_indices = _extract_feature_matrix(frame_data)
        
        if len(feature_matrix) < k_min:
            print(f"  Skipping {frame_key}: only {len(feature_matrix)} crops (< k_min={k_min})")
            # Assign all to cluster 0
            for i in crop_indices:
                frame_data["crops"][i]["cluster_id"] = 0
            continue
        
        # Normalize features
        scaler = StandardScaler()
        feature_matrix_scaled = scaler.fit_transform(feature_matrix)
        
        # Try different K values and compute metrics
        k_range = range(k_min, min(k_max + 1, len(feature_matrix) + 1))
        inertias = []
        models = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(feature_matrix_scaled)
            inertias.append(kmeans.inertia_)
            models.append(kmeans)
        
        # Select optimal K using elbow method
        optimal_k_idx = _auto_select_k_elbow(list(k_range), inertias)
        optimal_k = list(k_range)[optimal_k_idx]
        optimal_model = models[optimal_k_idx]
        
        print(f"  Optimal K selected: {optimal_k}")
        
        # Assign cluster IDs to crops
        cluster_labels = optimal_model.labels_
        for i, crop_idx in enumerate(crop_indices):
            frame_data["crops"][crop_idx]["cluster_id"] = int(cluster_labels[i])
        
        # Save frame clustering report
        frame_report = {
            "num_crops": len(feature_matrix),
            "k_tested": list(k_range),
            "inertias": inertias,
            "optimal_k": optimal_k,
            "cluster_sizes": {
                int(label): int(count) 
                for label, count in zip(*np.unique(cluster_labels, return_counts=True))
            }
        }
        clustering_report["frames"][frame_key] = frame_report
        
        # Generate visualizations if requested
        if debug_plots and clustering_dir:
            _visualize_clusters(
                feature_matrix_scaled,
                cluster_labels,
                frame_key,
                optimal_k,
                clustering_dir
            )
            _visualize_elbow(
                list(k_range),
                inertias,
                optimal_k,
                frame_key,
                clustering_dir
            )
        
        # Save centroids
        if clustering_dir:
            centroids_path = clustering_dir / f"{frame_key}_centroids.npy"
            np.save(str(centroids_path), optimal_model.cluster_centers_)
    
    # Save clustering report
    if clustering_dir:
        report_path = clustering_dir / "clustering_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(clustering_report, f, indent=2)
        print(f"\nClustering report saved to {report_path}")
    
    return metadata


def _extract_feature_matrix(frame_data: dict[str, Any]) -> tuple[np.ndarray, list[int]]:
    """Extract HSV histogram feature matrix from frame crops.
    
    Args:
        frame_data: Frame metadata dict containing crops with hsv_hist
        
    Returns:
        Tuple of (feature_matrix, crop_indices) where:
            - feature_matrix: (num_crops, num_features) array
            - crop_indices: List of crop indices that have valid histograms
    """
    features = []
    indices = []
    
    for i, crop in enumerate(frame_data["crops"]):
        if "hsv_hist" in crop and crop["hsv_hist"]:
            features.append(crop["hsv_hist"])
            indices.append(i)
    
    return np.array(features), indices


def _auto_select_k_elbow(k_values: list[int], inertias: list[float]) -> int:
    """Select optimal K using elbow method heuristic.
    
    Strategy: Choose K where the improvement starts to plateau significantly.
    Uses second derivative approach to find the point of maximum curvature.
    
    Args:
        k_values: List of K values tested
        inertias: List of corresponding inertia values
        
    Returns:
        Index of optimal K in k_values list
    """
    if len(inertias) <= 1:
        return 0
    
    if len(inertias) == 2:
        return 0  # Default to smallest K if only 2 values
    
    # Calculate first derivative (improvements = rate of inertia reduction)
    improvements = []
    for i in range(1, len(inertias)):
        improvement = inertias[i-1] - inertias[i]
        improvements.append(improvement)
    
    # Calculate second derivative (rate of change of improvements)
    if len(improvements) < 2:
        return 0
    
    second_derivatives = []
    for i in range(1, len(improvements)):
        second_deriv = improvements[i-1] - improvements[i]
        second_derivatives.append(second_deriv)
    
    # Find elbow at maximum positive second derivative
    # (point where improvements start declining most rapidly)
    if second_derivatives:
        max_curv_idx = second_derivatives.index(max(second_derivatives))
        # Add 1 because second_derivatives[i] corresponds to k_values[i+1]
        return max_curv_idx + 1
    
    # Fallback to middle value
    return len(k_values) // 2


def _visualize_clusters(
    feature_matrix: np.ndarray,
    cluster_labels: np.ndarray,
    frame_key: str,
    k: int,
    output_dir: Path,
) -> None:
    """Generate PCA and UMAP visualizations of clusters.
    
    Args:
        feature_matrix: Scaled feature matrix (num_crops, num_features)
        cluster_labels: Cluster assignments for each crop
        frame_key: Frame identifier for filename
        k: Number of clusters
        output_dir: Directory to save visualizations
    """
    # PCA projection (2D)
    pca = PCA(n_components=2, random_state=42)
    coords_pca = pca.fit_transform(feature_matrix)
    
    fig, ax = plt.subplots(figsize=(10, 8))
    scatter = ax.scatter(
        coords_pca[:, 0],
        coords_pca[:, 1],
        c=cluster_labels,
        cmap='tab10',
        s=100,
        alpha=0.7,
        edgecolors='black',
        linewidth=0.5
    )
    ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=11)
    ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=11)
    ax.set_title(f'{frame_key} - PCA Projection (K={k})', fontsize=13, fontweight='bold')
    ax.grid(alpha=0.3)
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label('Cluster ID', fontsize=11)
    
    plt.tight_layout()
    pca_path = output_dir / f"{frame_key}_clusters_pca_k{k}.png"
    plt.savefig(str(pca_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # UMAP projection (2D) if available
    if UMAP_AVAILABLE:
        reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=min(15, len(feature_matrix) - 1))
        coords_umap = reducer.fit_transform(feature_matrix)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        scatter = ax.scatter(
            coords_umap[:, 0],
            coords_umap[:, 1],
            c=cluster_labels,
            cmap='tab10',
            s=100,
            alpha=0.7,
            edgecolors='black',
            linewidth=0.5
        )
        ax.set_xlabel('UMAP 1', fontsize=11)
        ax.set_ylabel('UMAP 2', fontsize=11)
        ax.set_title(f'{frame_key} - UMAP Projection (K={k})', fontsize=13, fontweight='bold')
        ax.grid(alpha=0.3)
        
        # Add colorbar
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Cluster ID', fontsize=11)
        
        plt.tight_layout()
        umap_path = output_dir / f"{frame_key}_clusters_umap_k{k}.png"
        plt.savefig(str(umap_path), dpi=150, bbox_inches='tight')
        plt.close(fig)


def _visualize_elbow(
    k_values: list[int],
    inertias: list[float],
    optimal_k: int,
    frame_key: str,
    output_dir: Path,
) -> None:
    """Generate elbow method visualization.
    
    Args:
        k_values: List of K values tested
        inertias: List of corresponding inertia values
        optimal_k: Selected optimal K value
        frame_key: Frame identifier for filename
        output_dir: Directory to save visualization
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot inertia curve
    ax.plot(k_values, inertias, 'bo-', linewidth=2, markersize=8, label='Inertia')
    
    # Highlight optimal K
    optimal_idx = k_values.index(optimal_k)
    ax.plot(optimal_k, inertias[optimal_idx], 'ro', markersize=15, 
            label=f'Selected K={optimal_k}', zorder=5)
    
    # Styling
    ax.set_xlabel('Number of Clusters (K)', fontsize=12)
    ax.set_ylabel('Inertia (Within-Cluster Sum of Squares)', fontsize=12)
    ax.set_title(f'{frame_key} - Elbow Method for Optimal K', fontsize=14, fontweight='bold')
    ax.set_xticks(k_values)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    elbow_path = output_dir / f"{frame_key}_elbow_method.png"
    plt.savefig(str(elbow_path), dpi=150, bbox_inches='tight')
    plt.close(fig)
