import OpenEXR
import Imath
import numpy as np

class ExrReader():
    def __init__(self, ):
        return

    def imread(self, path):
        pt = Imath.PixelType(Imath.PixelType.FLOAT)
        exrfile = OpenEXR.InputFile(path)
        dw = exrfile.header()["dataWindow"]
        size = (dw.max.x - dw.min.x + 1, dw.max.y - dw.min.y + 1)
        (redstr, greenstr, bluestr) = exrfile.channels("RGB", pt)
        red = np.fromstring(redstr, dtype = np.float32)
        print(red.shape)
        green = np.fromstring(greenstr, dtype = np.float32)
        blue = np.fromstring(bluestr, dtype = np.float32)
        image = np.zeros((size[1], size[0], 3), dtype = np.float)
        image[:, :, 0] = red.reshape(size[1], size[0])
        image[:, :, 1] = green.reshape(size[1], size[0])
        image[:, :, 2] = blue.reshape(size[1], size[0])
        exrfile.close()

        return image

