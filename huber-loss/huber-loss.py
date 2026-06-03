import numpy as np

def huber_loss(y_true, y_pred, delta=1.0):
    """
    Compute Huber Loss for regression.
    """
    # Write code here
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)

    err = y_true - y_pred

    l = np.mean(np.where(np.abs(err)<=delta, 0.5*err**2, delta*(np.abs(err)-0.5*delta)))
    return l