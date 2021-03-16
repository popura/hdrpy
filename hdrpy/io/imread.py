from pathlib import Path
from typing import Union, Optional

import numpy as np

from hdrpy.format import RadianceHDRReader, PFMFormat


HDR_IMG_EXTENSIONS = ('.hdr', '.exr', '.pfm')


def has_file_allowed_extension(
    filename: Union[Path, str],
    extensions: tuple[str, ...]) -> bool:
    """Checks if a file is an allowed extension.
    Args:
        filename: path to a file
        extensions: extensions to consider (lowercase)
    Returns:
        bool: True if the filename ends with one of given extensions
    >>> has_file_allowed_extension("image.hdr", (".hdr", ".exr", ".pfm"))
    True
    >>> has_file_allowed_extension("image.hrd", (".hdr", ".exr", ".pfm"))
    False
    """
    return filename.lower().endswith(extensions)


def read(path: Union[Path, str],
         nan_sub: Optional[float] = None,
         inf_sub: Optional[float] = None) -> np.ndarray:
    """Reads an HDR image file.
    Args:
        path: path to a file
        nan_sub: value used for substituting NaN
        inf_sub: value used for substituting Inf
    Returns:
        image: readed image with a size of (H, W, C)
    >>> image = read("../data/memorial_o876.hdr")
    >>> image.shape
    """
    if isinstance(path, str):
        path = Path(path)
    
    reader = ReaderFactory.create(path)
    image = reader(path)

    if nan_sub is not None:
        image[np.isnan(image)] = nan_sub
    if inf_sub is not None:
        image[np.isinf(image)] = inf_sub

    return image


class ReaderFactory(object):
    @staticmethod
    def create(path: Union[Path, str]):
        """Returns a function for reading an image file at `path`.
        Args:
        Returns:
        >>> ReaderFactory.create()
        """
        if isinstance(path, str):
            path = Path(path)
        
        ext = path.suffix.lower()
        if ext == ".hdr":
            reader = RadianceHDRReader.read
        elif ext == ".exr":
            raise NotImplementedError()
        elif ext == ".pfm":
            reader = PFMFormat.read
        else:
            raise NotImplementedError()
        return reader


if __name__ == "__main__":
    import doctest
    doctest.testmod()