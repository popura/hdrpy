from typing import Union, Optional

import numpy as np
from colour import oetf, RGB_COLOURSPACES, RGB_luminance


def get_luminance(image: np.ndarray) -> np.ndarray:
    colourspace = RGB_COLOURSPACES["sRGB"]
    lum = RGB_luminance(image, colourspace.primaries, colourspace.whitepoint)
    return lum