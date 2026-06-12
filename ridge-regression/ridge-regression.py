import numpy as np

def ridge_regression(X, y, lam):
    """
    Compute ridge regression weights using the closed-form solution.
    """
    # Write code here
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)

    d = X.shape[1]

    return list(np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y))

    