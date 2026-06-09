import numpy as np

def naive_bayes_bernoulli(X_train, y_train, X_test):
    """
    Compute log-likelihood P(y|x) for Bernoulli Naive Bayes.
    """
    # Write code here
    X_train = np.asarray(X_train)
    y_train = np.asarray(y_train)
    X_test = np.asarray(X_test)

    classes, counts = np.unique(y_train, return_counts=True)
    n_train = len(y_train)

    # Log priors: log P(y)
    log_priors = np.log(counts / n_train)  # shape (n_classes,)

    # Feature likelihoods with Laplace smoothing (alpha=1)
    # theta[c, i] = P(x_i=1 | y=c)
    theta = np.array([
        (X_train[y_train == c].sum(axis=0) + 1) / (counts[i] + 2)
        for i, c in enumerate(classes)
    ])  # shape (n_classes, d)

    # Log posteriors for each test sample
    # log P(y|x) ∝ log P(y) + sum_i [ x_i * log(theta) + (1-x_i) * log(1-theta) ]
    log_posts = (
        log_priors[np.newaxis, :]                          # (1, n_classes)
        + X_test @ np.log(theta).T                         # (n_test, n_classes)
        + (1 - X_test) @ np.log(1 - theta).T              # (n_test, n_classes)
    )  # shape (n_test, n_classes)

    return log_posts

    