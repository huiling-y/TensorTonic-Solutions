def bigram_probabilities(tokens):
    """
    Returns: (counts, probs)
      counts: dict mapping (w1, w2) -> integer count
      probs: dict mapping (w1, w2) -> float P(w2 | w1) with add-1 smoothing
    """
    # Your code here
    V = list(set(tokens))

    probs_l = [[1] * len(V) for _ in V]

    counts = dict()
    
    for i in range(len(tokens)-1):
        w1 = tokens[i]
        w2 = tokens[i+1]

        idx1 = V.index(w1)
        idx2 = V.index(w2)

        probs_l[idx1][idx2] += 1

        if (w1, w2) not in counts:
            counts[(w1, w2)] = 1
        else:
            counts[(w1, w2)] += 1

    
    probs = dict()

    for i in range(len(V)):

        denominator = sum(probs_l[i])

        for j in range(len(V)):

            bigram = (V[i], V[j])

            #counts[bigram] = probs_l[i][j]
            probs[bigram] = probs_l[i][j] / denominator
            

    return counts, probs
        