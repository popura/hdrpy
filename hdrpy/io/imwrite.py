import re
import struct
#import OpenEXR
import Imath
import numpy as np
from pathlib import Path
import cv2
from colour import RGB_COLOURSPACES
from hdrpy.io.pfm_format import PfmWriter


def imwrite(path, img, nan_sub=None, inf_sub=None):
    path = Path(path)
    ext = path.suffix
    writer = ImWriter.get(ext)

    if nan_sub is not None:
        img[np.isnan(img)] = nan_sub
    if inf_sub is not None:
        img[np.isinf(img)] = inf_sub

    try:
        writer.imwrite(str(path), img)
    except TypeError as e:
        print("at {0}".format(path.name))
        print(e)


class ImWriter():
    @staticmethod
    def get(ext):
        try:
            writer = ImWriter.generate_writer(ext)
        except TypeError as e:
            print(e)

        return writer

    @staticmethod
    def generate_writer(ext):
        if ext == ".hdr":
            raise TypeError("ImWriter cannot handle " + ext + " files")
        elif ext == ".exr":
            raise TypeError("ImWriter cannot handle " + ext + " files")
        elif ext == ".pfm":
            writer = PfmWriter()
        else:
            raise TypeError(ext + " is not an HDR image")

        return writer
