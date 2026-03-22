import numpy as np

def kfold_split(N, k, shuffle=True, rng=None):
    """
    Returns: list of length k with tuples (train_idx, val_idx)
    """
    # Write code here
    idx = np.arange(N)
    if rng:
        idx = rng.permutation(idx)
    else:
        np.random.shuffle(idx)
    
    chunks = np.array_split(idx, k)
    results = []

    for i in range(k):
        val_idx = chunks[i]
        train_idx = np.concatenate(chunks[:i] + chunks[i+1:])

        results.append((train_idx, val_idx))

    return results
        
