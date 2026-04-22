def remove_stopwords(tokens, stopwords):
    """
    Returns: list[str] - tokens with stopwords removed (preserve order)
    """
    # Your code here
    new_tokens = [token for token in tokens if token not in stopwords]
    return new_tokens