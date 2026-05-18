import numpy as np

def silhouette_score(X, labels):
    """
    Compute the mean Silhouette Score for given points and cluster labels.
    X: np.ndarray of shape (n_samples, n_features)
    labels: np.ndarray of shape (n_samples,)
    Returns: float
    """
    # Write code here
    # diff (N, N, K), D (N, N)
    X = np.asarray(X)
    diff = X[:, None, :] - X[None, :, :] 
    D = np.sqrt(np.sum(diff**2, axis=-1))

    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return 0.0

    mapped_labels = np.searchsorted(unique_labels, labels) # -> (N,)
    num_clusters = len(unique_labels)

    # one_hot (N, C)
    one_hot = np.zeros((len(labels), num_clusters))
    one_hot[np.arange(len(labels)), mapped_labels] = 1

    
    cluster_counts = np.sum(one_hot, axis=0) # get the number of points of each cluster

    # compute a(i)
    cluster_dis_sums = D @ one_hot # -> (N, C)

    own_cluster_dis_sums = cluster_dis_sums[np.arange(len(labels)), mapped_labels] # (N,)
    own_cluster_counts = cluster_counts[mapped_labels] # (n,)
    a = own_cluster_dis_sums / np.maximum(own_cluster_counts-1, 1)


    # compute b(i)
    all_cluster_mean = cluster_dis_sums / np.maximum(cluster_counts, 1)
    mask_own = one_hot.astype(bool)
    all_cluster_mean[mask_own] = np.inf
    b = np.min(all_cluster_mean, axis=1)

    s = (b-a) / np.maximum(a, b)

    return np.mean(s)