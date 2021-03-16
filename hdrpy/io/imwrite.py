import re
import struct
import OpenEXR
import Imath
import numpy as np
from pathlib import Path
import cv2
from colour import RGB_COLOURSPACES
from hdrpy.io.pfm_format import PfmWriter


def write(
    path: Uniton[Path, str],
    image: np.ndarray,
    nan_sub: Optional[float] = None,
    inf_sub: Optional[float] = None) -> None:
    """Writes an HDR image file.
    Args:
        path: path to a file
        image: ndarray with a size of (H, W, C)
        nan_sub: value used for substituting NaN
        inf_sub: value used for substituting Inf
    Returns:
        None
    >>> image = write("../data/test_img.pfm", np.random.rand(100, 100, 3))
    """
    if isinstance(path, str):
        path = Path(path)
    
    if nan_sub is not None:
        image[np.isnan(image)] = nan_sub
    if inf_sub is not None:
        image[np.isinf(image)] = inf_sub

    writer = WriterFactory.create(path)
    image = writer(path, image)

    return image


class WriterFactory(object):
    @staticmethod
    def create(path: Union[Path, str]):
        """Returns a function for writing an image file to `path`.
        Args:
            Path:
        Returns:
            writer: 
        >>> WriterFactory.create()
        """
        if isinstance(path, str):
            path = Path(path)
        
        ext = path.suffix.lower()
        if ext == ".hdr":
            raise NotImplementedError()
        elif ext == ".exr":
            raise NotImplementedError()
        elif ext == ".pfm":
            writer = PFMFormat.write
        else:
            raise NotImplementedError()
        return writer



if __name__ == "__main__":
    import doctest
    doctest.testmod()