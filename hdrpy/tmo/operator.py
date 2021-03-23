from typing import Union

import copy
import numpy as np
from scipy.stats import gmean
import sys
from colour import oetf, RGB_COLOURSPACES, RGB_luminance
import cv2

from hdrpy import get_luminance


class ColorProcessing(object):
    """Base class for process image color.
    """
    def __init__(self):
        pass

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Processes image
        Args:
            image: RGB image with a size of (H, W, 3)
        Returns:
            New RGB image
        """
        raise NotImplementedError()


class ReplaceLuminance(ColorProcessing):
    """Replaces luminance of given image with given luminance.
    Attributes:
        luminance: New luminance that the resulting image will have
        org_luminance: current luminance of image
    Examples:
    >>> f = ReplaceLuminance(luminance)
    >>> new_image = f(image)
    >>> np.allclose(get_luminance(new_image), lumiance)
    True
    """
    def __init__(
        self,
        luminance: np.ndarray,
        org_luminance: Union[np.ndarray, None] = None) -> None:
        super().__init__()
        self.luminance = np.clip(luminance, 0, np.finfo(np.float32).max)
        self.org_lumiannce = org_luminance

    def __call__(
        self,
        image: np.ndarray) -> np.ndarray:
        """Replaces luminance of given image with given luminance.
        Args:
            image: RGB image with a size of (H, W, 3)
        Returns:
            New RGB image
        """
        return replace_luminance(image, self.luminance, self.org_luminance)


def replace_luminance(
    image: np.ndarray,
    luminance: np.ndarray,
    org_luminance: Union[np.ndarray, None] = None) -> np.ndarray:
    """Replaces luminance of given image with given luminance.
    Args:
        image: RGB image with a size of (H, W, 3)
        luminance: New luminance that the resulting image will have
        org_luminance: Luminance of image `rgb`
    Returns:
        New RGB image
    """
    luminance = np.clip(lum_new, 0, np.finfo(np.float32).max)

    if org_lumiannce is None:
        org_luminance = get_luminance(image)

    ratio = self.luminance / org_luminance
    ratio[org_luminance == 0] = 0

    return image * ratio


class LuminanceProcessing(object):
    """Base class for processing image luminance.
    """
    def __init__(self):
        pass

    def __call__(self, luminance: np.ndarray) -> np.ndarray:
        """Processes luminance
        Args:
            luminance: luminancea with a size of (H, W)
        Returns:
            New luminannce
        """
        raise NotImplementedError()


if __name__ == "__main__":
    import doctest
    doctest.testmod()
