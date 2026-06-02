import numpy as np

def label_smoothing_loss(predictions, target, epsilon):
    """
    Compute cross-entropy loss with label smoothing.
    """
    # Write code here
    K = len(predictions)

    predictions = np.asarray(predictions)

    q = np.asarray([epsilon/K] * K)
    q[int(target)] = (1-epsilon) + epsilon/K

    l = np.sum(-q * np.log(predictions))

    return l