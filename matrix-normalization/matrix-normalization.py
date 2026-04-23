import numpy as np

def matrix_normalization(matrix, axis=None, norm_type='l2'):
    """
    Normalize a 2D matrix along specified axis using specified norm.
    """
    try:
        matrix = np.asarray(matrix, dtype=float)
    except (ValueError, TypeError):
        return None
    
    # Check if matrix is empty using size
    if matrix.size == 0:
        return None

    if matrix.ndim != 2:
        return None
    
    # Validate axis parameter
    if axis is not None and (axis >= matrix.ndim or axis <= -matrix.ndim - 1):
        return None
    
    # Validate norm_type
    if norm_type == 'l2':
        norm = np.sqrt(np.sum(matrix**2, axis=axis, keepdims=True))
    elif norm_type == 'l1':
        norm = np.sum(np.abs(matrix), axis=axis, keepdims=True)
    elif norm_type == 'max':
        norm = np.max(np.abs(matrix), axis=axis, keepdims=True)
    else:
        return None
    
    # Divide by norm, avoiding division by zero
    result = np.divide(matrix, norm, out=np.zeros_like(matrix), where=norm != 0)
    return result