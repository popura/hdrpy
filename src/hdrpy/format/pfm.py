import struct
from pathlib import Path
from typing import Union

import numpy as np

from hdrpy.format import Format

class PFMFormat(Format):
    """Handles HDR images written in the portable floatmap image (PFM) format,
    e.g., reading and writing
    """
    @staticmethod
    def read(path: Union[Path, str]) -> np.ndarray:
        """Reads an HDR image with PFM format.
        Args:
            path: path to a file
        Return:
            image: readed image with a size of (H, W, C)
        >>> image = PFMFormat.read("./data/Flowers.pfm")
        >>> image.shape
        (853, 1280, 3)
        """
        image = None
        with open(path, 'rb') as f:
            f.readline()
            w, h = f.readline().decode('ascii').strip().split(' ')
            w = int(w)
            h = int(h)
            f.readline()

            siz = h * w * 3
            image = np.array(struct.unpack('f' * siz, f.read(4 * siz)))
            image = image.reshape((h, w, 3))

        if image is None:
            raise Exception('Failed to load file "{0}"'.format(path))

        return image

    @staticmethod
    def write(
        path: Union[Path, str],
        image: np.ndarray) -> None:
        """Writes an image to path as PFM format.
        Args:
            path: path to a file
            image: ndarray with a size of (H, W, C)
        >>> PFMFormat.write("./data/test_img.pfm", np.random.rand(100, 100, 3))
        >>> image = PFMFormat.read("./data/test_img.pfm")
        >>> image.shape
        (100, 100, 3)
        """
        h, w, dim = image.shape
        with open(path, 'wb') as f:
            f.write(bytearray('PF\n', 'ascii'))
            f.write(bytearray('{0:d} {1:d}\n'.format(w, h), 'ascii'))
            f.write(bytearray('-1.0\n', 'ascii'))

            siz = h * w * 3
            tmp = image.reshape(siz)
            f.write(struct.pack('f'*siz, *tmp))


if __name__ == "__main__":
    import doctest
    doctest.testmod()