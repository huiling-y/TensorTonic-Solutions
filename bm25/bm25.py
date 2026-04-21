import numpy as np
from collections import Counter
import math

def bm25_score(query_tokens, docs, k1=1.2, b=0.75):
    """
    Returns numpy array of BM25 scores for each document.
    """
    # Write code here

    if not docs:
        return np.array([], dtype=float)

    doc_lengths = np.array([len(doc) for doc in docs], dtype=float)
    avg_doc_length = np.mean(doc_lengths)

    N = len(docs)

    doc_freq = Counter()
    for doc in docs:
        unique_terms = set(doc)
        doc_freq.update(unique_terms)

    scores = np.zeros(N, dtype=float)

    query_tokens = list(dict.fromkeys(query_tokens))

    term_freqs = [Counter(doc) for doc in docs]

    for term in query_tokens:

        df = doc_freq.get(term, 0)
        idf = math.log((N - df + 0.5) / (df + 0.5) + 1)

        for doc_idx in range(N):
            #tf = term_freqs[doc_idx].get(term, 0) / doc_lengths[doc_idx]

            tf = term_freqs[doc_idx].get(term, 0)
            
            if tf > 0:
                norm_factor = 1 - b + b *(doc_lengths[doc_idx] / avg_doc_length)
                denominator = tf + k1 * norm_factor

                score_component = idf * (tf * (k1 + 1) / denominator)
                scores[doc_idx] += score_component
    return scores

    
        