from typing import Union, Optional

import numpy as np

from hdrpy.image import get_luminance
from hdrpy.tmo import ColorProcessing, LuminanceProcessing, Compose, ReplaceLuminance
from hdrpy.tmo.linear import ExposureCompensation


def reinhard_curve(
    lum: np.ndarray,
    lum_ave: np.ndarray,
    lum_white: float) -> np.ndarray:
    """Tone curve for Reinhard's TMO.
    This function does not clip pixel values greater than 1.0
    but pixel values less than 0.0 will be clipped
    Args:
        lum: luminance
        lum_ave: locally averaged luminance
        lum_white: white point.
    Returns:
        tone-mapped intensity
    >>> lum = np.linspace(0, 10, 5)
    >>> reinhard_curve(lum=lum, lum_ave=lum, lum_white=float("Inf"))
    array([ 0.        ,  0.71428571,  0.83333333,  0.88235294,  0.90909091])
    """
    lum = np.clip(lum, 0, np.finfo(np.float32).max)
    lum_ave = np.clip(lum_ave, 0, np.finfo(np.float32).max)
    lum_disp = (lum / (1 + lum_ave)) * (1 + (lum / (lum_white ** 2)))
    return lum_disp


class ReinhardCurve(LuminanceProcessing):
    """Non-linear tone curve for Reinhard's TMO.
    Attributes:
        mode: `global` or `local`
        whitepoint: whitepoint
    Examples:
    >>> import hdrpy
    >>> image = hdrpy.io.read("./data/CandleGlass.exr")
    >>> luminance = hdrpy.image.get_luminance(image)
    >>> f = ReinhardCurve("global")
    >>> new_lumminance = f(luminance)
    """
    def __init__(
        self,
        mode: str = "global",
        whitepoint: Union[float, str] = "Inf") -> None:
        if mode not in ("global", "local"):
            raise ValueError()

        if mode == "local":
            raise NotImplementedError

        self.mode = mode

        if isinstance(whitepoint, str) and whitepoint.lower() == "Inf".lower():
            whitepoint = float("Inf")
        
        self.whitepoint = whitepoint

    def __call__(self, luminance: np.ndarray) -> np.ndarray:
        """Calculates local averages of luminance
        and maps luminance by Reinhard's curve using
        the calculated local averages.
        Args:
            luminance: luminance with a size of (H, W)
        Returns:
            tone-mapped intensity
        """
        if self.mode == "global":
            average_luminance = luminance
        return reinhard_curve(luminance, average_luminance, self.whitepoint)


class ReinhardTMO(ColorProcessing):
    """Reinhard's TMO.
    Attributes:
        tmo: an instance of ReplaceLuminance that performs
        the exposure compensation, non-linear mapping, and replacing luminance.
    Examples:
    >>> import hdrpy
    >>> hdr = hdrpy.io.read("./data/CandleGlass.exr")
    >>> f = ReinhardTMO()
    >>> ldr = f(hdr)
    >>> hdrpy.io.write("./data/CandleGlass.jpg", ldr)
    """
    def __init__(
        self,
        ev: float = 0,
        mode: str = "global",
        whitepoint: Union[float, str] = "Inf") -> None:
        super().__init__()
        self.tmo = ReplaceLuminance(
            Compose([ExposureCompensation(ev=ev),
                     ReinhardCurve(mode=mode, whitepoint=whitepoint)]))
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Applying Reinhard's TMO to an image.
        This function does not clip pixel values greater than 1.0
        but pixel values less than 0.0 will be clipped
        Args:
            image: ndarray with a size of (H, W, C)
        Returns:
            tone-mapped image
        """
        return self.tmo(image)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
