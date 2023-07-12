from pathlib import Path
from typing import Union, Optional

import numpy as np
from PIL import Image

from hdrpy.format import RadianceHDRFormat, PFMFormat
try:
    from hdrpy.format import OpenEXRFormat
except ImportError:
    pass


def write(
    path: Union[Path, str],
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
    >>> image = write("./data/test_img.pfm", np.random.rand(100, 100, 3))
    >>> image = write("./data/test_img.jpg", np.random.rand(100, 100, 3))
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
            path: path to a file to be wrote
        Returns:
            writer: python function for writing image at `path`
        Raises:
            NotImplementedError
        >>> type(WriterFactory.create("test.pfm"))
        <class 'function'>
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
            def pil_writer(path: Union[Path, str], image: np.ndarray) -> None:
                image = np.clip(image, 0, 1)
                image *= np.iinfo("uint8").max
                Image.fromarray(image.astype(np.uint8)).save(str(path))
            writer = pil_writer
        return writer



if __name__ == "__main__":
    import doctest
    doctest.testmod()