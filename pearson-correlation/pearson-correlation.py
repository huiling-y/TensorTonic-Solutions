import numpy as np

def pearson_correlation(X):
    """
    Compute Pearson correlation matrix from dataset X.
    """
    # Write code here
    X = np.asarray(X, dtype=float)

    if X.ndim != 2:
        return None
    if len(X) < 2:
        return None

    centered = X - np.mean(X, axis=0, keepdims=True)
    cov = (centered.T @ centered) / X.shape[0]
    
    std = np.std(X, axis=0)
    
    zero_var = (std == 0)
    safe_std = np.where(zero_var, 1.0, std)
    denominator = np.outer(safe_std, safe_std)

    pearson = cov / denominator

    pearson[zero_var, :] = np.nan
    pearson[:, zero_var] = np.nan

    valid = ~zero_var
    pearson[valid, valid] = 1.0

    return pearson