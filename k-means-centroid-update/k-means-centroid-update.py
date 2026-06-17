import numpy as np

def k_means_centroid_update(points, assignments, k):
    """
    Compute new centroids as the mean of assigned points.
    """
    # Write code here

    points = np.asarray(points, dtype=float)
    assignments = np.asarray(assignments)

    centroids = np.zeros((k, points.shape[1]))

    for i in range(k):
        mask = assignments == i

        if mask.any():
            centroids[i] = points[mask].mean(axis=0)


    return centroids.tolist()