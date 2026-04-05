import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

try:
    from ..pipeline_contracts import (
        ARTIFACT_CLUSTER_LABELS,
        ARTIFACT_MATRIX,
        ARTIFACT_MODEL,
        ArtifactSpec,
        StageContract,
    )
except ImportError:  # pragma: no cover - supports direct imports from src/
    from pipeline_contracts import (  # type: ignore
        ARTIFACT_CLUSTER_LABELS,
        ARTIFACT_MATRIX,
        ARTIFACT_MODEL,
        ArtifactSpec,
        StageContract,
    )


def adjusted_kmeans(data_points, k, min_size=100):
    """
    maknu se small clusters i onda se dodaju uklonjene točke najbližem centroidu
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_points)

    N = data_scaled.shape[0]


    # Track original indices
    active_idx = np.arange(N)
    removed_idx = np.array([], dtype=int)

    while True:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        if data_scaled[active_idx].shape[0] < k:
            raise ValueError(f"Not Enough structure to run this method for k={k} and min_size={min_size}. Consider reducing k or min_size.")
        kmeans.fit(data_scaled[active_idx])
        
        labels = kmeans.labels_
        unique, counts = np.unique(labels, return_counts=True)
        
        small_clusters = unique[counts < min_size]
        
        if len(small_clusters) == 0:
            break
        
        mask_small = np.isin(labels, small_clusters)
        
        # Move small cluster indices to removed
        removed_idx = np.concatenate([removed_idx, active_idx[mask_small]])
        
        # Keep only large clusters
        active_idx = active_idx[~mask_small]

    # Final clustering on surviving points
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    kmeans.fit(data_scaled[active_idx])

    final_labels_active = kmeans.labels_

    # Assign removed points to nearest centroid
    if len(removed_idx) == 0:
        final_labels = np.empty(N, dtype=int)
        final_labels[active_idx] = final_labels_active
    else:
        removed_points = data_scaled[removed_idx]
        distances = kmeans.transform(removed_points)
        final_labels_removed = np.argmin(distances, axis=1)

        # Construct final label array
        final_labels = np.empty(N, dtype=int)
        final_labels[active_idx] = final_labels_active
        final_labels[removed_idx] = final_labels_removed

    return final_labels


def merge_kmeans(data_points, k, min_size=100):
    """
    funkcija koja implementira adjusted kmeans koji uvijek daje clustere koji imaju barem min_size članova
    vraća cluster labels, indekse aktivnih i uklonjenih točaka
    """
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_points)

    N = data_scaled.shape[0]

        # Over-cluster
    k_over = k * 2

    kmeans = KMeans(
        n_clusters=k_over,
        n_init=20,
        random_state=42
    )

    labels = kmeans.fit_predict(data_scaled)
    centroids = kmeans.cluster_centers_

    # Count cluster sizes
    unique, counts = np.unique(labels, return_counts=True)

    # Identify big and small clusters
    big_clusters = unique[counts >= min_size]
    small_clusters = unique[counts < min_size]

    # Mapping from old cluster -> new cluster
    final_centroids = []
    cluster_map = {}

    # Keep big clusters
    for c in big_clusters:
        cluster_map[c] = len(final_centroids)
        final_centroids.append(centroids[c])

    final_centroids = np.array(final_centroids)

    # Merge small clusters
    for c in small_clusters:
        dists = np.linalg.norm(
            final_centroids - centroids[c],
            axis=1
        )

        nearest_big = np.argmin(dists)
        cluster_map[c] = nearest_big

    # Reassign labels
    final_labels = np.array([cluster_map[l] for l in labels])

    return final_labels


class AdjustedKMeansClustering:
    """Pipeline wrapper around ``adjusted_kmeans``."""

    input_type = ARTIFACT_MATRIX
    output_type = ARTIFACT_CLUSTER_LABELS
    input_artifacts = {
        "data_points": ArtifactSpec(
            name="data_points",
            kind=ARTIFACT_MATRIX,
            dense=True,
            description="Dense embedding matrix from the previous stage.",
        )
    }
    output_artifacts = {
        "clusters": ArtifactSpec(
            name="clusters",
            kind=ARTIFACT_CLUSTER_LABELS,
            dense=True,
            description="Cluster labels aligned with the input rows.",
        ),
        "model": ArtifactSpec(
            name="model",
            kind=ARTIFACT_MODEL,
            dense=True,
            description="Fitted clustering stage instance.",
        ),
    }
    contract = StageContract(
        input_type=input_type,
        output_type=output_type,
        input_artifacts=input_artifacts,
        output_artifacts=output_artifacts,
        dense=True,
        description="Adjusted KMeans clustering stage for embeddings.",
    )

    def __init__(self, k: int, min_size: int = 100) -> None:
        self.k = int(k)
        self.min_size = int(min_size)
        self.cluster_labels_: np.ndarray | None = None

    def fit(self, data_points):
        """Fit the clustering stage and store cluster labels."""

        self.cluster_labels_ = adjusted_kmeans(data_points, self.k, self.min_size)
        return self

    def export_artifacts(self):
        """Export cluster labels and the fitted model."""

        if self.cluster_labels_ is None:
            raise RuntimeError("The clustering stage must be fitted before exporting artifacts.")
        return {
            "clusters": self.cluster_labels_,
            "model": self,
        }


class MergeKMeansClustering:
    """Pipeline wrapper around ``merge_kmeans``."""

    input_type = ARTIFACT_MATRIX
    output_type = ARTIFACT_CLUSTER_LABELS
    input_artifacts = {
        "data_points": ArtifactSpec(
            name="data_points",
            kind=ARTIFACT_MATRIX,
            dense=True,
            description="Dense embedding matrix from the previous stage.",
        )
    }
    output_artifacts = {
        "clusters": ArtifactSpec(
            name="clusters",
            kind=ARTIFACT_CLUSTER_LABELS,
            dense=True,
            description="Cluster labels aligned with the input rows.",
        ),
        "model": ArtifactSpec(
            name="model",
            kind=ARTIFACT_MODEL,
            dense=True,
            description="Fitted clustering stage instance.",
        ),
    }
    contract = StageContract(
        input_type=input_type,
        output_type=output_type,
        input_artifacts=input_artifacts,
        output_artifacts=output_artifacts,
        dense=True,
        description="Merge KMeans clustering stage for embeddings.",
    )

    def __init__(self, k: int, min_size: int = 100) -> None:
        self.k = int(k)
        self.min_size = int(min_size)
        self.cluster_labels_: np.ndarray | None = None

    def fit(self, data_points):
        """Fit the clustering stage and store cluster labels."""

        self.cluster_labels_ = merge_kmeans(data_points, self.k, self.min_size)
        return self

    def export_artifacts(self):
        """Export cluster labels and the fitted model."""

        if self.cluster_labels_ is None:
            raise RuntimeError("The clustering stage must be fitted before exporting artifacts.")
        return {
            "clusters": self.cluster_labels_,
            "model": self,
        }
