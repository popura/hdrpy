from typing import Union, Optional

import numpy as np

from hdrpy.image import get_luminance
from hdrpy.stats import gmean
from hdrpy.tmo import ColorProcessing, LuminanceProcessing, Compose, ReplaceLuminance
from hdrpy.tmo.linear import ExposureCompensation, multiply_scalar
from hdrpy.tmo.operator import replace_luminance


def kinoshita_curve(luminance: np.ndarray, eps: float = 1e-6):
    """Maps non-linear display luminance of LDR images to normalized linear luminance.
    Args:
        luminance: input luminance
        eps: small value for avoiding anomaly
    Returns:
        normalized linear luminance
    Examples:
    >>> kinoshita_curve(np.linspace(0, 1, 5))
    array([  0.00000000e+00,   3.33333333e-01,   1.00000000e+00,
             3.00000000e+00,   9.99999000e+05])
    """
    lum = luminance.copy()
    lum[lum == 1] = 1 - eps
    return lum / (1 - lum)


class KinoshitaCurve(LuminanceProcessing):
    """Non-linear inverse tone curve for Kinoshita's iTMO.
    This is the inverse of Reinhard's global tone curve.
    Attributes:
        eps: small value for avoiding anomaly
    Examples:
    >>> f = KinoshitaCurve()
    >>> f(np.linspace(0, 1, 5))
    array([  0.00000000e+00,   3.33333333e-01,   1.00000000e+00,
             3.00000000e+00,   9.99999000e+05])
    """
    def __init__(
        self,
        eps: float = 1e-6) -> None:
        self.eps= eps
    
    def __call__(self, luminance: np.ndarray) -> np.ndarray:
        return kinoshita_curve(luminance, self.eps)


class KinoshitaITMO(ColorProcessing):
    """Kinoshita iTMO.
    Attributes:
        alpha:
        hdr_gmean:
        eps:
        curve:
    Examples:
    >>> import hdrpy
    >>> hdr = hdrpy.io.read("./data/CandleGlass.exr")
    >>> f = hdrpy.tmo.ReinhardTMO()
    >>> ldr = f(hdr)
    >>> f = KinoshitaITMO()
    >>> hdr_ = f(ldr)
    >>> np.sum(np.abs(hdr - hdr_))
    """
    def __init__(
        self,
        alpha: Optional[float] = 0.18,
        hdr_gmean: Optional[float] = None,
        eps: float = 1e-6) -> None:
        self.alpha = alpha
        self.hdr_gmean = hdr_gmean 
        self.eps = eps
        self.curve = KinoshitaCurve(eps)

    def __call__(self, image: np.ndarray) -> np.ndarray:
        ld = get_luminance(image)
        ls = self.curve(ld)
        alpha, hdr_gmean = self.get_params(ld, self.alpha, self.hdr_gmean)
        lw = multiply_scalar(ls, hdr_gmean / alpha)
        return replace_luminance(image, lw)
    
    def get_params(
        self,
        luminance: np.ndarray,
        alpha: Optional[float],
        hdr_gmean: Optional[float]) -> np.ndarray:
        """
        Args:
            luminance:
            alpha:
            hdr_gmean:
        Returns:
            (alpha, hdr_gmean)
        """
        
        if alpha is None and hdr_gmean is None:
            return 1, 1
        
        if alpha is None:
            return self.estimate_alpha(luminance, hdr_gmean), hdr_gmean 
            
        if hdr_gmean is None:
            return alpha, self.estimate_hdr_gmean(luminance, alpha)
        
        return alpha, hdr_gmean 

    def estimate_alpha(
        self,
        luminance: np.ndarray,
        hdr_gmean: float) -> float:
        """estimates alpha from luminance
        and the geometric mean of corresponding HDR luminance
        Args:
            luminance:
            hdr_gmean:
        Returns:
            estimated alpha
        """
        black_pixel_number = np.sum(luminance == 0)
        if black_pixel_number == 0:
            return hdr_gmean

        pixel_number = luminance.size
        ldr_gmean = hdr_gmean(luminance)
        alpha = (pixel_number / (pixel_number-black_pixel_number)) * np.log(ldr_gmean)
        alpha -= (black_pixel_number / (pixel_number-black_pixel_number)) * np.log(hdr_gmean)
        return exp(alpha)
        
    def estimate_hdr_gmean(
        self,
        luminance: np.ndarray,
        alpha: float) -> float:
        """estimates the geometric mean of HDR luminance
        from luminance and alpha
        Args:
            luminance:
            alpha:
        Returns:
            estimated HDR geometric mean
        """
        black_pixel_number = np.sum(luminance == 0)
        normalized_gmean = gmean(luminance)

        if black_pixel_number == 0:
            return normalized_gmean

        pixel_number = luminance.size
        hdr_gmean = (pixel_number / black_pixel_number) * np.log(normalized_gmean)
        hdr_gmean -= ((pixel_number-black_pixel_number) / black_pixel_number) * np.log(alpha)
        return np.exp(hdr_gmean)
        
if __name__ == "__main__":
    import doctest
    doctest.testmod()
