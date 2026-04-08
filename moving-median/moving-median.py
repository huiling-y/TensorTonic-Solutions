import numpy as np

def moving_median(values, window_size):
    """
    Compute the rolling median for each window position.
    """
    # Write code here

    n = len(values)
    num_window = n - window_size + 1

    medians = []

    for i in range(num_window):

        medians.append(np.median(values[i:i+window_size]))

    return medians