import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    X = np.array(X)
    y = np.array(y)
    
    w = np.zeros(X.shape[1],)
    b = 0
    
    # Write code here
    for i in range(steps):
        z = np.dot(X, w) + b
        p = _sigmoid(z)

        delta_w = np.dot(X.T, (p - y)) / y.shape[0]
        delta_b = np.mean(p - y)

        w -= lr * delta_w
        b -= lr * delta_b
    return (w, b)