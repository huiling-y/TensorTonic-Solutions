import numpy as np

def relu(x):
    """
    Implement ReLU activation function.
    """
    # Write code here
    x = np.asarray(x)

    x = np.where(x>0, x, 0)

    return x