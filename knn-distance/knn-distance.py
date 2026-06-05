import numpy as np

def knn_distance(X_train, X_test, k):
    """
    Compute pairwise distances and return k nearest neighbor indices.
    """
    # Write code here
    X_train = np.array(X_train).reshape(-1, 1) if np.array(X_train).ndim == 1 else np.array(X_train)
    X_test = np.array(X_test).reshape(-1, 1) if np.array(X_test).ndim == 1 else np.array(X_test)

    diff = X_test[:, np.newaxis] - X_train[np.newaxis]
    distances = np.sqrt((diff ** 2).sum(axis=-1))

    n_train = X_train.shape[0]
    result = np.full((len(X_test), k), -1, dtype=int)
    result[:, :n_train] = np.argsort(distances, axis=1)[:, :k]

    return result