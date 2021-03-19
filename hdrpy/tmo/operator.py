import copy
import numpy as np
from scipy.stats import gmean
import sys
from colour import oetf, RGB_COLOURSPACES, RGB_luminance
import cv2


def multiply_scalar(
    intensity: np.ndarray,
    factor: Optional[float] = None,
    ev: float = 0) -> np.ndarray:
    """Compensates exposure of given image.
    Args:
        image: ndarray with a size of (H, W, C)
    Returns:
        multiplied image
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
    intensity = copy.deepcopy(intensity)
    intensity[np.isnan(intensity)] = eps
    intensity[intensity <= 0] = eps
    if factor is None:
        factor = 0.18 * 2**ev / gmean(intensity, axis=None)

    if np.isnan(gmean(intensity, axis=None)):
        print("NaN: {0}".format(np.isnan(intensity).any()))
    if np.isinf(gmean(intensity, axis=None)):
        print("INF: {0}".format(np.isinf(intensity).any()))
    #print("factor: {0}".format(factor))
    new_intensity = intensity * factor
    np.clip(new_intensity, 0, np.finfo(np.float32).max, out=new_intensity)
    return new_intensity


def normalize_luminance(hdrimage: np.ndarray) -> np.ndarray:
    """Normalizes an image luminance into [0, 1] range.
    Args:
        image: ndarray with a size of (H, W, C)
    Returns:
        normalized image
    """
    colourspace = RGB_COLOURSPACES["sRGB"]
    hdrimage = np.clip(hdrimage, 0, np.finfo(np.float32).max)
    lum = RGB_luminance(hdrimage, colourspace.primaries, colourspace.whitepoint)
    max_lum = lum.max()
    ldrimage = replace_color(hdrimage, lum / max_lum, lum)
    return ldrimage


def normalize_color(hdrimage: np.ndarray) -> np.ndarray:
    """Normalizes an image into [0, 1] range.
    Args:
        image: ndarray with a size of (H, W, C)
    Returns:
        normalized image
    >>> image = normalize_color(np.random.rand(100, 100, 3))
    >>> np.amax(image)
    1.
    >>> np.amin(image)
    0.
    """
    eps = sys.float_info.epsilon
    hdrimage = np.clip(hdrimage, 0, np.finfo(np.float32).max)
    hdrimage[np.isnan(hdrimage)] = eps
    max_intensity = hdrimage.max()
    min_intensity = hdrimage.min()
    ldrimage = (hdrimage - min_intensity) / (max_intensity - min_intensity)
    return ldrimage


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


def eilertsen_tmo(
    hdrimage: np.ndarray,
    ev: float = 0,
    exponent: float = 0.9,
    sigma: float = 0.6):
    """Applying Eilertsen's TMO to an image.
    This function does not clip pixel values greater than 1.0
    but pixel values less than 0.0 will be clipped
    Args:
        image: ndarray with a size of (H, W, C)
    Returns:
        tone-mapped image
    >>> eilertsen_tmo(10 * np.random.rand(100, 100, 3))
    """
    colourspace = RGB_COLOURSPACES["sRGB"]
    clipped_image = np.clip(hdrimage, 0, np.finfo(np.float32).max)
    lum = RGB_luminance(clipped_image, colourspace.primaries, colourspace.whitepoint)
    scaled_lum = multiply_scalar(lum, ev=ev)
    scaled_image = replace_color(clipped_image, scaled_lum, lum)
    ldrimage = eilertsen_curve(scaled_image, exponent, sigma)
    return ldrimage


def eilertsen_curve(
    intensity: np.ndarray,
    exponent: float,
    sigma: float) -> np.ndarray:
    """Tone curve for Eilertsen's TMO.
    This function does not clip pixel values greater than 1.0
    or less than 0.0.
    Args:
        intensity: luminance
        exponent:
        sigma:
    Returns:
        tone-mapped intensity
    >>> intensity = np.linspace(0, 10, 1000)
    >>> eilertsen_curve(intensity=intensity, exponent=0.9, sigma=0.6)
    """
    powered_intensity = intensity ** exponent
    intensity_disp = (1 + sigma) * (powered_intensity / (powered_intensity + sigma))
    return intensity_disp


def replace_color(
    rgb: np.ndarray,
    lum_new: np.ndarray,
    lum_org: np.ndarray) -> np.ndarray:
    """Replaces luminance of given image with given luminance.
    Args:
        rgb: RGB image with a size of (H, W, 3)
        lum_new: New luminance that the resulting image will have
        lum_old: Luminance of image `rgb`
    Returns:
        New RGB image
    """
    rgb_new = copy.deepcopy(rgb)
    clipped_lum = np.clip(lum_new, 0, np.finfo(np.float32).max)
    ratio = clipped_lum / lum_org
    ratio[lum_org == 0] = 0
    rgb_new[:, :, 0] *= ratio
    rgb_new[:, :, 1] *= ratio
    rgb_new[:, :, 2] *= ratio
    return rgb_new


if __name__ == "__main__":
    import doctest
    doctest.testmod()