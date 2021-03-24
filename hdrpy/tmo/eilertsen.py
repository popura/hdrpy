from typing import Union, Optional

import numpy as np

from hdrpy.tmo import ColorProcessing, LuminanceProcessing


def eilertsen_curve(
    intensity: np.ndarray,
    exponent: float = 0.9,
    sigma: float = 0.6) -> np.ndarray:
    """Tone curve for Eilertsen's TMO.
    This function does not clip pixel values greater than 1.0
    or less than 0.0.
    Args:
        intensity: intensities of RGB value
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


class EilertsenCurve(ColorProcessing):
    """Non-linear tone curve for Reinhard's TMO.
    Attributes:
        exponent:
        sigma:
    Examples:
    >>> f = EilertsenCurve()
    >>> new_image = f(image)
    """
    def __init__(
        self,
        exponent: float = 0.9,
        sigma: float = 0.6) -> None:

        self.exponent = exponent
        self.sigma = sigma

    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Maps image by applying Eilertsen's curve
        to RGB component, individually.
        Args:
            image : image with a size of (H, W, C)
        Returns:
            tone-mapped image
        """
        return eilertsen_curve(image, exponent, sigma)


def EilertsenTMO(ColorProcessing):
    """Eilertsen's TMO.
    Attributes:
        tmo: an instance of Compose that performs
        the exposure compensation, and non-linear mapping.
    Examples:
    >>> hdr = 10 * np.random.rand(100, 100, 3)
    >>> f = EilertsenTMO()
    >>> ldr = f(hdr)
    """
    def __init__(
        self,
        ev: int = 0,
        exponent: float = 0.9,
        sigma: float = 0.6) -> None:
        super().__init__()
        self.tmo = Compose(
            [ReplaceLuminance(ExposureCompensation(ev=ev)),
             EilertsenCurve(exponent=exponent, sigma=sigma)])
    
    def __call__(self, image: np.ndarray) -> np.ndarray:
        """Applying Eilertsen's TMO to an image.
        This function does not clip pixel values greater than 1.0
        but pixel values less than 0.0 will be clipped
        Args:
            image: ndarray with a size of (H, W, C)
        Returns:
            tone-mapped image
        >>> eilertsen_tmo(10 * np.random.rand(100, 100, 3))
        """
        return self.tmo(image)


if __name__ == "__main__":
    import doctest
    doctest.testmod()
