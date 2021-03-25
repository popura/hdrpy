from typing import Union, Optional

import numpy as np

from hdrpy.stats import gmean
from hdrpy.image import get_luminance
from hdrpy.tmo import ColorProcessing, LuminanceProcessing


def multiply_scalar(
    intensity: np.ndarray,
    factor: float,
    nan_sub: Optional[float] = 0,
    inf_sub: Union[Optional[float], str] = "Inf",
    minus_sub: Optional[float] = 0) -> np.ndarray:
    """Compensates exposure of given image.
    Args:
        intensity: input array
        factor: scale ratio
        nan_sub: value for substituting NaN in `intensity`
        inf_sub: value for substituting Inf in `intensity`
        minus_sub: value for substituting negative values in `intensity`
    Returns:
        multiplied image
    Raise:
        ValueError
    >>> image = np.ones((5, 5))
    >>> multiply_scalar(image, factor=2)
    array([[ 2.,  2.,  2.,  2.,  2.],
           [ 2.,  2.,  2.,  2.,  2.],
           [ 2.,  2.,  2.,  2.,  2.],
           [ 2.,  2.,  2.,  2.,  2.],
           [ 2.,  2.,  2.,  2.,  2.]])
    >>> multiply_scalar(image, factor=0.5)
    array([[ 0.5,  0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5,  0.5],
           [ 0.5,  0.5,  0.5,  0.5,  0.5]])
    >>> image = np.random.randn(5, 5)
    >>> image = multiply_scalar(image, factor=2, minus_sub=0)
    >>> np.all(image >= 0)
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
    >>> import hdrpy
    >>> hdr = hdrpy.io.read("./data/CandleGlass.exr")
    >>> luminance = hdrpy.image.get_luminance(hdr)
    >>> f = ExposureCompensation()
    >>> new_luminance = f(luminance)
    >>> np.isclose(gmean(new_luminance, axis=None), 0.18, atol=1e-3)
    True
    >>> f = ExposureCompensation(ev=1)
    >>> new_luminance = f(luminance)
    >>> np.isclose(gmean(new_luminance, axis=None), 0.36, atol=1e-3)
    True
    >>> f = ExposureCompensation(ev=0)
    >>> ldr = f(hdr)
    >>> hdrpy.io.write("./data/CandleGlass.jpg", ldr)
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
        g = gmean(image, eps=self.eps)
        factor = self.ev2gmean(self.ev) / g
        return multiply_scalar(image, factor)
        
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
        return 0.18 * (2**ev)
    
    @staticmethod
    def gmean2ev(g: float) -> float:
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
        >>> ExposureCompensation.gmean2ev(0.72)
        2.0
        >>> ExposureCompensation.gmean2ev(0.09)
        -1.0
        """
        return np.log2(g / 0.18)
    

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
    >>> np.all(image <= 1)
    True
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
        if self.mode == "luminance":
            lum = hdrpy.get_lumianance(image)
            factor = 1. / np.amax(lum)
        else:
            factor = 1. / np.amax(image)
        
        return multiply_scalar(image, factor)


if __name__ == "__main__":
    import doctest
    doctest.testmod()