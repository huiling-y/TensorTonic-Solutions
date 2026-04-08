import numpy as np 


def elu(x, alpha):
    """
    Apply ELU activation to each element.
    """
    # Write code here
    x = np.array(x)

    result = np.where(x <= 0, alpha * (np.exp(x)-1), x)
    return list(result)