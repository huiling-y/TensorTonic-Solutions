import numpy as np

def matrix_trace(A):
    """
    Compute the trace of a square matrix (sum of diagonal elements).
    """
    # Write code here

    A = np.asarray(A)

    n = A.shape[0]

    trace = 0

    for i in range(n):
        trace += A[i,i]

    return trace
