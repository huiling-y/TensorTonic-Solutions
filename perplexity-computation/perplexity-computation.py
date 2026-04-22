import math

def perplexity(prob_distributions, actual_tokens):
    """
    Compute the perplexity of a token sequence given predicted distributions.
    """
    # Write code here

    log_probs = []

    for i in range(len(actual_tokens)):
        prob = prob_distributions[i][actual_tokens[i]]

        log_probs.append(-math.log(prob))

    H = (sum(log_probs) / len(log_probs))

    return math.exp(H)