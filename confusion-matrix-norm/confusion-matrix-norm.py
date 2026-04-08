import numpy as np

def confusion_matrix_norm(y_true, y_pred, num_classes=None, normalize='none'):
    """
    Compute confusion matrix with optional normalization.
    """
    # Write code here
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)

    # Handle empty arrays
    if y_true.size == 0:
        K = num_classes if num_classes is not None else 0
        return np.zeros((K, K), dtype=int if normalize == 'none' else float)

    # Determine number of classes
    K = num_classes if num_classes is not None else int(max(y_true.max(), y_pred.max()) + 1)

    # Validate label ranges
    if y_true.min() < 0 or y_true.max() >= K:
        raise ValueError(f"y_true contains labels outside valid range [0, {K-1}]")
    if y_pred.min() < 0 or y_pred.max() >= K:
        raise ValueError(f"y_pred contains labels outside valid range [0, {K-1}]")

    # Vectorized bincount over flattened indices
    indices = y_true * K + y_pred
    cm = np.bincount(indices, minlength=K * K).reshape(K, K)

    if normalize == 'none':
        return cm

    cm = cm.astype(float)

    if normalize == 'true':       # normalize over actual (rows)
        denom = cm.sum(axis=1, keepdims=True)
    elif normalize == 'pred':     # normalize over predicted (cols)
        denom = cm.sum(axis=0, keepdims=True)
    elif normalize == 'all':      # normalize over everything
        denom = cm.sum(keepdims=True)
    else:
        raise ValueError(f"normalize must be 'none', 'true', 'pred', or 'all', got '{normalize}'")

    # Avoid division by zero: replace zero denominators with 1
    denom[denom == 0] = 1
    return cm / denom