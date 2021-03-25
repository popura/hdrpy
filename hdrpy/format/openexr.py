from pathlib import Path
from typing import Union

import numpy as np
import OpenEXR
import Imath

import cv2
from colour import oetf, RGB_COLOURSPACES, RGB_luminance

from hdrpy.format import Format


class _OpenEXRReader():
    def __init__(self, ):
        self.pt = Imath.PixelType(Imath.PixelType.FLOAT)
        return

    def imread(self, path):
        exrfile = OpenEXR.InputFile(path)
        channel_type = set(exrfile.header()["channels"].keys())
        if channel_type >= {"R", "G", "B"}:
            image = self.read_rgb_channels(exrfile)
        elif channel_type >= {"Y", "BY", "RY"}:
            image = self.read_ycbcr_channels(exrfile)
        elif channel_type >= {"Y"}:
            image = self.read_y_channel(exrfile)
        else:
            raise TypeError("This channel type is not supported.")

        exrfile.close()
        return image

    def read_rgb_channels(self, exrfile):
        dw = exrfile.header()["dataWindow"]
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        (redstr, greenstr, bluestr) = exrfile.channels("RGB", self.pt)
        red = np.fromstring(redstr, dtype=np.float32)
        green = np.fromstring(greenstr, dtype=np.float32)
        blue = np.fromstring(bluestr, dtype=np.float32)

        image = np.zeros((size[1], size[0], 3), dtype=np.float)
        image[:, :, 0] = red.reshape(size[1], size[0])
        image[:, :, 1] = green.reshape(size[1], size[0])
        image[:, :, 2] = blue.reshape(size[1], size[0])
        return image

    def read_ycbcr_channels(self, exrfile):
        channel = exrfile.header()["channels"]
        dw = exrfile.header()["dataWindow"]
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)

        image = np.zeros((size[1], size[0], 3), dtype=np.float)
        for i, ch in enumerate(["Y", "BY", "RY"]):
            chstr = exrfile.channel(ch, self.pt)
            charray = np.fromstring(chstr, dtype=np.float32)
            x_subsample = channel[ch].xSampling
            y_subsample = channel[ch].ySampling
            charray.shape = (size[1]//y_subsample, size[0]//x_subsample)
            image[:, :, i] = cv2.resize(charray, None,
                                        fx=x_subsample, fy=y_subsample,
                                        interpolation=cv2.INTER_LINEAR)
        image = self.ycbcr_to_rgb(image)

        return image

    def read_y_channel(self, exrfile):
        dw = exrfile.header()["dataWindow"]
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        chstr = exrfile.channel("Y", self.pt)
        charray = np.fromstring(chstr, dtype=np.float32)
        image = np.zeros((size[1], size[0], 3), dtype=np.float)
        image[:, :, 0] = charray.reshape(size[1], size[0])
        image[:, :, 1] = charray.reshape(size[1], size[0])
        image[:, :, 2] = charray.reshape(size[1], size[0])

        return image


    def rgb_to_ycbcr(self, rgb):
        pass

    def ycbcr_to_rgb(self, ycbcr):
        colourspace = RGB_COLOURSPACES["ITU-R BT.709"]
        y_weight = colourspace.matrix_RGB_to_XYZ[1, :]
        rgb = np.zeros(ycbcr.shape, dtype=float)
        rgb[:, :, 0] = ycbcr[:, :, 2] * ycbcr[:, :, 0] + ycbcr[:, :, 0]
        rgb[:, :, 2] = ycbcr[:, :, 1] * ycbcr[:, :, 0] + ycbcr[:, :, 0]
        rgb[:, :, 1] = (ycbcr[:, :, 0]
                        - y_weight[0] * rgb[:, :, 0]
                        - y_weight[2] * rgb[:, :, 2])
        rgb[:, :, 1] /= y_weight[1]

        return rgb


_reader = _OpenEXRReader()


class OpenEXRFormat(Format):
    """Handles HDR images written in the OpenEXR format,
    e.g., reading and writing
    """

    @staticmethod
    def read(path: Union[Path, str]) -> np.ndarray:
        """Reads an HDR image with OpenEXR format.
        Args:
            path: path to a file
        Return:
            image: readed image with a size of (H, W, C)
        >>> image = OpenEXRFormat.read("./data/CandleGlass.exr")
        >>> image.shape
        (853, 1280, 3)
        """
        return _reader.imread(str(path))
    
    @staticmethod
    def write(
        path: Union[Path, str],
        image: np.ndarray) -> None:
        """Writes an image to path as OpenEXR format.
        Args:
            path: path to a file
            image: ndarray with a size of (H, W, C)
        >>> OpenEXRFormat.write("./data/test_img.exr", np.random.rand(100, 100, 3))
        >>> image = OpenEXRFormat.read("./data/test_img.exr")
        >>> image.shape
        (100, 100, 3)
        """
        raise NotImplementedError()


if __name__ == "__main__":
    import doctest
    doctest.testmod()