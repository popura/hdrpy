from typing import Union
from collections.abc import Sequence

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
    """Replaces luminance of a given image with luminance obtained
    by applying LuminanceProcessing to the image.
    Attributes:
        processing: luminance processing to be applied to images
    Examples:
    >>> processing = 
    >>> f = ReplaceLuminance(processing)
    >>> new_image = f(image)
    >>> luminance = get_luminance(image)
    >>> new_luminance = processing(luminance)
    >>> np.allclose(get_luminance(new_image), new_lumiance)
    True
    """
    def __init__(
        self,
        processing: LuminanceProcessing) -> None:
        super().__init__()
        self.processing = processing

    def __call__(
        self,
        image: np.ndarray) -> np.ndarray:
        """Replaces luminance of given image with given luminance.
        Args:
            image: RGB image with a size of (H, W, 3)
        Returns:
            New RGB image
        """
        org_luminance = get_luminance(image)
        luminance = self.processing(org_luminance)
        return replace_luminance(image, luminance, org_luminance)


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
    luminance = np.clip(luminance, 0, np.finfo(np.float32).max)

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


class Compose(ColorProcessing, LuminanceProcessing):
    """Class for compiling a sequence of color or luminance processings
    Attributes:
        processings: a sequence of processings
    Examples:
    >>> processings = [ExposureCompensation(), NormalizeRange()]
    >>> f = Compose(processings)
    >>> f(np.random.rand(100, 100))
    """
    def __init__(
        self,
        processings: Union[Sequence[ColorProcessing], Sequence[LuminanceProcessing]]) -> None:
        self.processings = processings

    def __call__(self, image: np.ndarray) -> np.ndarray:
        x = np.copy(image)
        for f in processings:
            x = f(x)
        return x


if __name__ == "__main__":
    import doctest
    doctest.testmod()
