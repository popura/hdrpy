from typing import Union, Optional

import numpy as np

import hdrpy
from hdrpy import get_luminance
from hdrpy.tmo import ColorProcessing, LuminanceProcessing


def multiply_scalar(
    intensity: np.ndarray,
    factor: float,
    nan_sub: Optional[float] = 0,
    inf_sub: Union[Optional[float], str] = "Inf",
    minus_sub: Optional[float] = 0) -> np.ndarray:
    """Compensates exposure of given image.
    Args:
        image: ndarray with a size of (H, W, C)
    Returns:
        multiplied image
    Raise:
        ValueError
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

    new_intensity = intensity * factor

    if nan_sub is not None:
        new_intensity[np.isnan(new_intensity)] = nan_sub

    if inf_sub is None:
        pass
    elif isinstance(inf_sub, str) and inf_sub.lower() == "Inf".lower():
        new_intensity[np.isinf(new_intensity)] = np.finfo(np.float32).max
    elif isinstance(inf_sub, float):
        new_intensity[np.isinf(new_intensity)] = inf_sub
    else:
        raise ValueError()

    if minus_sub is not None:
        new_intensity[new_intensity <= 0] = minus_sub

    return new_intensity


class ExposureCompensation(ColorProcessing, LuminanceProcessing):
    """Adjusts exposure, i.e., brightness, of images based on geometric mean.
    Attributes:
        ev: target exposure value
        eps: small value for stability
    Examples:
    >>> f = ExposureCompensation()
    >>> new_luminance = f(luminance)
    >>> np.isclose(gmean(new_luminannce, axis=None), 0.18)
    True
    >>> f = ExposureCompensation(ev=1)
    >>> new_luminance = f(luminance)
    >>> np.isclose(gmean(new_luminannce, axis=None), 0.36)
    True
    """
    def __init__(
        self,
        ev: Optional[float] = 0,
        eps: float = 1e-6) -> None:
        super().__init__()
        self.ev = ev
        self.eps = eps

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Adjusts exposure, i.e., brightness, of images based on geometric mean.
        Args:
            image: image with a size of (H, W, 3) or
                   luminance with a size of (H, W)
        Returns:
            New image or luminance with the same size of the input
        """
        gmean = hdrpy.gmean(image, self.eps)
        factor = self.ev2gmean(self.ev) / gmean
        return multiply_scaler(intensity, factor)
        
    @staticmethod
    def ev2gmean(ev: float) -> float:
        """Calculates the geometric mean of an image
        having an relative exposure value of `ev`,
        where 0 EV means that the geometric mean of an image is 0.18

        Args:
            ev: an relative exposure value
        
        Returns:
            The geometric mean
        
        Examples:
        >>> ExposureCompensation.ev2gmean(0)
        0.18
        >>> ExposureCompensation.ev2gmean(2)
        0.72
        >>> ExposureCompensation.ev2gmean(-1)
        0.09
        """
        return 0.18 * 2**ev
    
    @staticmethod
    def gmean2ev(gmean: float) -> float:
        """Calculates an relative exposure value of an image
        whose geometric mean of luminance is `gmean`,
        where 0 EV means that the geometric mean is 0.18

        Args:
            ev: an relative exposure value
        
        Returns:
            The geometric mean
        
        Examples:
        >>> ExposureCompensation.gmean2ev(0.18)
        0.0
        >>> ExposureCompensation.ev2gmean(0.72)
        2.0
        >>> ExposureCompensation.ev2gmean(0.09)
        -1.0
        """
        return np.log2(gmean / 0.18)
    

class NormalizeRange(ColorProcessing, LuminanceProcessing):
    """Scales an image so that
    its color or luminance range is in [0, 1] range.
    This processing is different from the min-max normalization
    because this does not include the shift operation.
    Attributes:
        mode: "luminance" or "color". Default is "luminance"
    Examples:
    >>> f = NormalizeRange("color")
    >>> image = f(np.random.randn(100, 100, 3))
    >>> np.amax(image)
    1.
    >>> np.amin(image)
    0.
    """
    def __init__(self, mode: str = "luminance"):
        super().__init__()
        
        if mode.lower() not in ("luminance", "color"):
            raise ValueError
        
        self.mode = mode

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Normalizes an image luminance or color into [0, 1] range.
        Args:
            image: image with a size of (H, W, C)
                or luminance with a size of (H, W)
        Returns:
            normalized image
        """
        if self.mode is "luminance":
            lum = hdrpy.get_lumianance(image)
            factor = 1. / np.amax(lum)
        else:
            factor = 1. / np.amax(image)
        
        return multiply_scaler(image, factor)


if __name__ == "__main__":
    import doctest
    doctest.testmod()