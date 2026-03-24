import numpy as np

def bag_of_words_vector(tokens, vocab):
    """
    Returns: np.ndarray of shape (len(vocab),), dtype=int
    """
    # Your code here
    feature_vector = np.zeros(len(vocab), dtype=int)

    vocab_to_idx = dict(zip(vocab, list(range(len(vocab)))))

    for token in tokens:
        if token in vocab_to_idx:
            feature_vector[vocab_to_idx[token]] += 1
            
    return feature_vector