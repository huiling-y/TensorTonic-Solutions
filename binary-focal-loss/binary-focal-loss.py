import math
import numpy as np

def binary_focal_loss(predictions, targets, alpha, gamma):
    """
    Compute the mean binary focal loss.
    """
    # Write code here
    predictions = np.asarray(predictions)
    targets = np.asarray(targets)

    p = np.where(targets == 1, predictions, 1-predictions)

    fl = - alpha * (1-p)**gamma * np.log(p)

    return np.mean(fl)