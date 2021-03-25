from typing import Union, Optional

import numpy as np


def gmean(
    a: np.ndarray,
    axis: Optional[int] = None,
    eps: Optional[float] = 1e-6):
    """Computes the geometric mean along the specified axis.
    Returns the geometric average of the array elements.
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
    >>> import hdrpy
    >>> hdr = hdrpy.io.read("./data/CandleGlass.exr")
    >>> gmean(hdrpy.image.get_luminance(hdr))
    0.00040137774000534386
    """
    a_c = a.copy()
    if eps is not None:
        a_c[a_c <= 0] = eps
        a_c[np.isnan(a_c)] = eps
    log_a = np.log(a_c)
    return np.exp(log_a.mean(axis=axis))


def min_max_normalization(
    a: np.ndarray,
    min_: Optional[Union[float, int]] = None,
    max_: Optional[Union[float, int]] = None) -> np.ndarray:
    """Normalizes the given array `a`
    so that its range is in [0, 1]
    
    Args:
        a: Input array
        min: Minimum possible value of the array
        max: Minimum possible value of the array
    Returns:
        array in [0, 1] range
    
    Examples:
    >>> a = min_max_normalization(np.random.randn(100))
    >>> np.all(a <= 1)
    True
    >>> np.all(a >= 0)
    True
    """
    if min_ is None:
        min_ = float(np.amin(a))
    
    if max_ is None:
        max_ = float(np.amax(a))
    
    return (a - min_) / (max_ - min_)


if __name__ == "__main__":
    import doctest
    doctest.testmod()


