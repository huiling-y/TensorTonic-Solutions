import numpy as np

def rnn_step_backward(dh, cache):
    """
    Returns:
        dx_t: gradient wrt input x_t      (shape: D,)
        dh_prev: gradient wrt previous h (shape: H,)
        dW: gradient wrt W               (shape: H x D)
        dU: gradient wrt U               (shape: H x H)
        db: gradient wrt bias            (shape: H,)
    """
    # Write code here
    # cache: [x_t, h_prev, h_t, W, U, b]

    x_t, h_prev, h_t, W, U, b = cache

    dh = np.asarray(dh)
    x_t = np.asarray(x_t)
    h_prev = np.asarray(h_prev)
    h_t = np.asarray(h_t)
    W = np.asarray(W)
    U = np.asarray(U)
    b = np.asarray(b)
    
    tanh_derivative = 1 - h_t ** 2  # shape (H,) - element-wise derivative
    
    # Chain rule: dL/dz = dL/dh_t ⊙ tanh'(z)  (⊙ = element-wise product)
    dz = dh * tanh_derivative  # shape (H,) - upstream gradient modulated by activation derivative

    dx_t = dz @ W # W.T @ dz
    dh_prev = dz @ U # U.T @ dz 
    dW = np.outer(dz, x_t)
    dU = np.outer(dz, h_prev)
    db = dz 

    return dx_t, dh_prev, dW, dU, db
