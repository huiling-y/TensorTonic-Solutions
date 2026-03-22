import numpy as np

def dropout(x, p=0.5, rng=None):
    """
    Apply dropout to input x with probability p.
    Return (output, dropout_pattern).
    """
    # Write code 
    x = np.array(x)
    
    if rng:
        arr_rng = rng.random(size=x.shape)
    else:
        arr_rng = np.random.random(size=x.shape)
    
    dropout_pattern = np.where(arr_rng < 1 - p, 1/(1-p), 0)

    output = x * dropout_pattern

    return (output, dropout_pattern)