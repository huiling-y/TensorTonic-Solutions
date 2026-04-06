import numpy as np

def tanh(x):
    """
    Implement Tanh activation function.
    """
    # Write code here
    x = np.asarray(x)

    result = (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

    return result