def word_count_dict(sentences):
    """
    Returns: dict[str, int] - global word frequency across all sentences
    """
    # Your code here
    word_freq = dict()

    for sent in sentences:
        for word in sent:
            if word not in word_freq:
                word_freq[word] = 1
            else:
                word_freq[word] += 1
    return word_freq