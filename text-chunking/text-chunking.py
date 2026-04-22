def text_chunking(tokens, chunk_size, overlap):
    """
    Split tokens into fixed-size chunks with optional overlap.
    """
    # Write code here
    step = chunk_size - overlap

    result = []

    for i in range(0, len(tokens), step):
        result.append(tokens[i:i+chunk_size])
        if (i + chunk_size) >= len(tokens):
            break
    return result