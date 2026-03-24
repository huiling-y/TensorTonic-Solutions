import numpy as np
from collections import Counter
import math

def tfidf_vectorizer(documents):
    """
    Build TF-IDF matrix from a list of text documents.
    Returns tuple of (tfidf_matrix, vocabulary).
    """
    # Write code here
    vocab = []
    doc_words = []

    for doc in documents:
        words = doc.split()
        vocab += words 
        doc_words.append(words)

    vocab = sorted(list(set(vocab)))

    vocab_to_idx = dict(zip(vocab, list(range(len(vocab)))))

    N = len(doc_words)
    V = len(vocab)
    tf_vector = np.zeros((N, V))
    idf_vector = np.zeros((N, V))

    for i,doc in enumerate(doc_words):
        doc_counter = dict(Counter(doc))
        doc_num_words = len(doc)
        for word in doc_counter:
            tf_vector[i, vocab_to_idx[word]] = doc_counter[word] / doc_num_words
            
    for word in vocab:
        df = 0
        for doc in doc_words:
            if word in doc:
                df += 1
        idf_vector[:, vocab_to_idx[word]] = np.log(N / df)

    tfidf = tf_vector * idf_vector

    return tfidf, vocab