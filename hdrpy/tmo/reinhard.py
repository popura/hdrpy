from typing import Union, Optional

import numpy as np

from hdrpy.tmo import ColorProcessing, LuminanceProcessing


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
    >>> lum = np.linspace(0, 10, 1000)
    >>> reinhard_curve(lum=lum, lum_ave=lum, lum_white=float("Inf"))
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
    >>> f = ReinhardCurve("global")
    >>> new_lumminance = f(lumminance)
    """
    def __init__(self, mode: str = "global", whitepoint: Union[float, str] = "Inf") -> None:
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


def reinhard_tmo(
    hdrimage: np.ndarray,
    ev: float = 0,
    lum_white: Optional[float] = None) -> np.ndarray:
    """Applying Reinhard's TMO to an image.
    This function does not clip pixel values greater than 1.0
    but pixel values less than 0.0 will be clipped
    Args:
        image: ndarray with a size of (H, W, C)
    Returns:
        tone-mapped image
    >>> reinhard_tmo(10 * np.random.rand(100, 100, 3))
    """
    if lum_white is None:
        lum_white = float("Inf")

    colourspace = RGB_COLOURSPACES["sRGB"]
    hdrimage = np.clip(hdrimage, 0, np.finfo(np.float32).max)
    lum = RGB_luminance(hdrimage, colourspace.primaries, colourspace.whitepoint)
    scaled_lum = multiply_scalar(lum, ev=ev)
    mapped_lum = reinhard_curve(scaled_lum, scaled_lum, lum_white)
    ldrimage = replace_color(hdrimage, mapped_lum, lum)
    return ldrimage


if __name__ == "__main__":
    import doctest
    doctest.testmod()
