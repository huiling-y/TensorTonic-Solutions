import math

def cosine_embedding_loss(x1, x2, label, margin):
    """
    Compute cosine embedding loss for a pair of vectors.
    """
    # Write code here

    cos = sum(v1*v2 for v1,v2 in zip(x1, x2)) / (math.sqrt(sum(v1**2 for v1 in x1)) * math.sqrt(sum(v2**2 for v2 in x2)))

    return 1 - cos if label == 1 else max(0, cos - margin)