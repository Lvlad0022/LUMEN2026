import numpy as np
from sklearn.cluster import KMeans
from sklearn.discriminant_analysis import StandardScaler

def my_adjusted_kmeans(data_points, k, min_size=100):
    scaler = StandardScaler()
    data_scaled = scaler.fit_transform(data_points)

    N = data_scaled.shape[0]


    # Track original indices
    active_idx = np.arange(N)
    removed_idx = np.array([], dtype=int)

    while True:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
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



import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import AgglomerativeClustering
from scipy.spatial.distance import cdist


def hierarchical_stratification( #ovo je samo neka koju je gpt dao
    X,
    target_k=10,
    min_size=100,
    use_pca=True,
    pca_var=0.95
):
    # 1️⃣ Scale
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 2️⃣ PCA (recommended for customer data)
    if use_pca:
        pca = PCA(n_components=pca_var)
        X_scaled = pca.fit_transform(X_scaled)

    # 3️⃣ Over-segment
    over_k = target_k * 3
    model = AgglomerativeClustering(
        n_clusters=over_k,
        linkage="ward"
    )

    labels = model.fit_predict(X_scaled)

    # 4️⃣ Compute centroids
    unique = np.unique(labels)
    centroids = np.vstack([
        X_scaled[labels == i].mean(axis=0)
        for i in unique
    ])

    sizes = np.array([
        np.sum(labels == i)
        for i in unique
    ])

    # 5️⃣ Merge small clusters
    big_mask = sizes >= min_size
    big_clusters = unique[big_mask]
    small_clusters = unique[~big_mask]

    final_centroids = centroids[big_mask]
    cluster_map = {}

    # map big clusters
    for idx, c in enumerate(big_clusters):
        cluster_map[c] = idx

    # merge small clusters to nearest big cluster
    for c in small_clusters:
        c_centroid = centroids[unique == c]
        dists = cdist(c_centroid, final_centroids)
        nearest_big = np.argmin(dists)
        cluster_map[c] = nearest_big

    final_labels = np.array([cluster_map[l] for l in labels])

    return final_labels