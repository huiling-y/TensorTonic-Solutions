def gradient_descent_quadratic(a, b, c, x0, lr, steps):
    """
    Return final x after 'steps' iterations.
    """
    # Write code here
    
    for i in range(steps):
        delta_fx = 2 * a * x0 + b
        x0 = x0 - lr * delta_fx
    return x0