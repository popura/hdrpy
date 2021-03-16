from pathlib import Path
from typing import Union

import numpy as np

class Format(object):
    def __init__(self):
        pass

    @staticmethod
    def read(path: Union[Path, str]) -> np.ndarray:
        raise NotImplementedError()

    @staticmethod
    def write(path: Union[Path, str], image: np.ndarray) -> None:
        raise NotImplementedError()