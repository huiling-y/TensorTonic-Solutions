import numpy as np

def gini_impurity(y_left, y_right):
    """
    Compute weighted Gini impurity for a binary split.
    """
    # Write code here
    def gini(count_k):
        p_k = count_k / sum(count_k)
        return 1 - np.sum(p_k**2)

    y_left = np.asarray(y_left)
    y_right = np.asarray(y_right)

    _, count_left = np.unique(y_left, return_counts=True)
    _, count_right = np.unique(y_right, return_counts=True)

    N = len(y_left) + len(y_right)
    if N == 0:
        return 0.0

    # Check the input arrays, not the counts derived from them
    gini_left = 0 if len(y_left) == 0 else len(y_left) / N * gini(count_left)
    gini_right = 0 if len(y_right) == 0 else len(y_right) / N * gini(count_right)

    return float(gini_left + gini_right)