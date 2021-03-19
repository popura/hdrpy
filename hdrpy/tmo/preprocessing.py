import sys
from pathlib import Path
from typing import Union, Optional

import numpy as np
from scipy.stats import gmean


def multiply_scalar(
    intensity: np.ndarray,
    factor: Optional[float] = None,
    ev: float = 0) -> np.ndarray:
    """Compensates exposure of given image.
    Args:
        image: ndarray with a size of (H, W, C)
    Returns:
    >>> image = multiply_scalar(np.ones((5, 5)))
    >>> np.isclose(gmean(image, axis=None), 0.18)
    True
    >>> image = multiply_scalar(np.ones((5, 5, 2)), ev=1)
    >>> np.isclose(gmean(image, axis=None), 0.36)
    True
    >>> image = multiply_scalar(np.ones(5), ev=-2)
    >>> np.isclose(gmean(image, axis=None), 0.045)
    True
    >>> image1 = np.ones((10, 10))
    >>> image2 = multiply_scalar(image1, factor=1.5)
    >>> np.allclose(image1 * 1.5, image2)
    True
    """
    eps = sys.float_info.epsilon
    replaced_intensity = intensity
    replaced_intensity[intensity == 0] = eps
    if factor is None:
        factor = 0.18 * 2**ev / gmean(intensity, axis=None)

    return intensity * factor


if __name__ == "__main__":
    import doctest
    doctest.testmod()