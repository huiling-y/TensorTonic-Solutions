import numpy as np

def adamw_step(w, m, v, grad, lr=0.001, beta1=0.9, beta2=0.999, weight_decay=0.01, eps=1e-8):
    """
    Perform one AdamW update step.
    """
    # Write code here
    w = np.asarray(w)
    m = np.asarray(m)
    v = np.asarray(v)
    grad = np.asarray(grad)

    mt = beta1 * m + (1-beta1) * grad
    vt = beta2 * v + (1-beta2) * grad**2

    wt = w - lr * (weight_decay * w) - lr * (mt / (np.sqrt(vt) + eps))

    return wt, mt, vt