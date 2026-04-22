import numpy as np


def _sigmoid(x):
    """Numerically stable sigmoid function"""
    return np.where(x >= 0, 1.0/(1.0+np.exp(-x)), np.exp(x)/(1.0+np.exp(x)))

def swish(x):
    """
    Implement Swish activation function.
    """
    # Write code here
    x = np.asarray(x)

    return x * _sigmoid(x)