from pathlib import Path
from typing import Union

import numpy as np

class Format(object):
    """Base class for handling an HDR image format,
    e.g., reading and writing
    """
    def __init__(self):
        pass

    @staticmethod
    def read(path: Union[Path, str]) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def write(path: Union[Path, str], image: np.ndarray) -> None:
        raise NotImplementedError()