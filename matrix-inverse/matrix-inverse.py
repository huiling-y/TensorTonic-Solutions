import numpy as np

def matrix_inverse(A):
    """
    Returns: A_inv of shape (n, n) such that A @ A_inv ≈ I
    """
    # Write code here
    A = np.asarray(A, dtype=float)

    if A.ndim == 2 and A.shape[0] == A.shape[1]:
        if abs(np.linalg.det(A)) < 1e-10:
            return None
        return np.linalg.inv(A)
    return None
