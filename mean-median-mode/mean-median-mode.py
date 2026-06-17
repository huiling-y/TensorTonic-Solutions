import numpy as np
from collections import Counter

def mean_median_mode(x):
    """
    Compute mean, median, and mode.
    """
    # Write code here
    counts = Counter(x)

    return float(np.mean(x)), float(np.median(x)), float(counts.most_common(1)[0][0])