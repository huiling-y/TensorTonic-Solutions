import numpy as np

def pca_projection(X, k):
    """
    Project data onto the top-k principal components.
    """
    # Write code here

    X = np.asarray(X, dtype=float)
    X_centered = X - np.mean(X, axis=0, keepdims=True)

    N, d = X_centered.shape

    C = (X_centered.T @ X_centered) / (N - 1)

    W = []

    for i in range(k):
        v = np.random.rand(d)
        # orthogonalize against already-found components
        for w in W:
            v = v - (w @ v) * w
        nrm = np.linalg.norm(v)
        v = v / nrm if nrm > 1e-12 else v

        for _ in range(100):
            v_new = C @ v
            nv = np.linalg.norm(v_new)
            if nv < 1e-12:          # eigenvalue ~ 0: matrix exhausted
                break
            v = v_new / nv

        W.append(v)
        eigen_val = (v.T @ C) @ v
        C -= eigen_val * np.outer(v, v)

    W = np.column_stack(W)
    result = (X_centered @ W)

    return result.tolist()