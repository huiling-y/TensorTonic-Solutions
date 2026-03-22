import numpy as np

def batch_generator(X, y, batch_size, rng=None, drop_last=False):
    """
    Randomly shuffle a dataset and yield mini-batches (X_batch, y_batch).
    """
    # Write code here
    idx = list(range(len(y)))

    if rng == None:
        np.random.shuffle(idx)
    else:
        rng.shuffle(idx)

    X = np.array(X)
    y = np.array(y)

    idx = np.array(idx)

    for start_idx in range(0, len(y), batch_size):
        end_idx = start_idx + batch_size
        batch_idx = idx[start_idx:end_idx]
        if drop_last and len(batch_idx) < batch_size:
            break
        yield X[batch_idx], y[batch_idx]