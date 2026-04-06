def binning(values, num_bins):
    """
    Assign each value to an equal-width bin.
    """
    # Write code here

    max_val, min_val = max(values), min(values)
    n = len(values)
    if min_val == max_val:
        return [0] * n

    w = (max_val - min_val) / num_bins

    result = [0] * n 

    for i in range(n):
        bin_idx = min((values[i] - min_val)/w, num_bins-1)
        result[i] = int(bin_idx)
    return result