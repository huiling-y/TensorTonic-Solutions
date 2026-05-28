import numpy as np

def mean_squared_error(y_pred, y_true):
    """
    Returns: float MSE
    """
    # Write code here
    y_pred = np.asarray(y_pred)
    y_true = np.asarray(y_true)

    N = len(y_true)

    if len(y_true) != len(y_pred):
        return None

    mse = np.sum((y_pred-y_true)**2) / N

    return mse
