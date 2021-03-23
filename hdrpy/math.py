from typing import Union, Optional

import numpy as np

def gmean(
    a: np.ndarray,
    axis: Optional[int] = None,
    eps: Optional[float] = 1e-6):
    """Compute the geometric mean along the specified axis.
    Return the geometric average of the array elements.
    That is:  n-th root of (x1 * x2 * ... * xn)

    Args:
        a: Input array
        axis: Axis along which the geometric mean is computed.
            If None, compute over the whole array `a`.
        eps: Small value for stability

    Returns:
        gmean: The geometric mean of `a`

    Examples:
    >>> gmean(np.array([1, 4]))
    2.0
    >>> gmean(np.array([1, 2, 3, 4, 5, 6, 7]))
    3.3800151591412964
    """
    a_c = a.copy()
    if eps is not None:
        a_c[a_c <= 0] = eps
        a_c[np.isnan(a_c)] = eps
    log_a = np.log(a)
    return np.exp(log_a.mean(axis=axis))
