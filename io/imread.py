import re
import struct
import OpenEXR
import Imath
import numpy as np
from pathlib import Path

def imread(path):
    path = Path(path)
    ext = path.suffix
    reader = ImReader.get(ext)
    image = reader.imread(str(path))

    return image

class ImReader():
    @staticmethod
    def get(ext):
        try:
            reader = ImReader.generate_reader(ext)
        except TypeError as e:
            print(e)
        
        return reader

    @staticmethod
    def generate_reader(ext):
        if ext == ".hdr":
            reader = HdrReader()
        elif ext == ".exr":
            reader = ExrReader()
        elif ext == ".pfm":
            reader = PfmReader()
        else:
            raise TypeError(ext + "is not an HDR image")

        return reader

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
        green = np.fromstring(greenstr, dtype = np.float32)
        blue = np.fromstring(bluestr, dtype = np.float32)

        image = np.zeros((size[1], size[0], 3), dtype = np.float)
        image[:, :, 0] = red.reshape(size[1], size[0])
        image[:, :, 1] = green.reshape(size[1], size[0])
        image[:, :, 2] = blue.reshape(size[1], size[0])
        exrfile.close()

        return image

class HdrReader():
    def __init__(self, ):
        self.HDR_NONE = 0x00
        self.HDR_RLE_RGBE_32 = 0x01
        return
    
    def imread(self, path):
        with open(path, "rb") as im_file:
            bufsize = 4096
            filetype = self.HDR_NONE
            valid = False
            exposure = 1.0

            # Read header section
            while True:
                buf = im_file.readline(bufsize).decode('ascii')
                if buf[0] == '#' and (buf == '#?RADIANCE\n' or buf == '#?RGBE\n'):
                    valid = True
                else:
                    p = re.compile('FORMAT=(.*)')
                    m = p.match(buf)
                    if m is not None and m.group(1) == '32-bit_rle_rgbe':
                        filetype = self.HDR_RLE_RGBE_32
                        continue

                    p = re.compile('EXPOSURE=(.*)')
                    m = p.match(buf)
                    if m is not None:
                        exposure = float(m.group(1))
                        continue

                if buf[0] == '\n':
                    # Header section ends
                    break

            if not valid:
                raise Exception('HDR header is invalid!!')

            # Read body section
            width = 0
            height = 0
            buf = im_file.readline(bufsize).decode()
            p = re.compile('([\-\+]Y) ([0-9]+) ([\-\+]X) ([0-9]+)')
            m = p.match(buf)
            if m is not None and m.group(1) == '-Y' and m.group(3) == '+X':
                width = int(m.group(4))
                height = int(m.group(2))
            else:
                raise Exception('HDR image size is invalid!!')

            # Check byte array is truly RLE or not
            byte_start = im_file.tell()
            now = ord(im_file.read(1))
            now2 = ord(im_file.read(1))
            if now != 0x02 or now2 != 0x02:
                filetype = HDR_NONE
            im_file.seek(byte_start)

            if filetype == self.HDR_RLE_RGBE_32:
                # Run length encoded HDR
                tmpdata = np.zeros((width * height * 4), dtype=np.uint8)
                nowy = 0
                while True:
                    now = -1
                    now2 = -1
                    try:
                        now = ord(im_file.read(1))
                        now2 = ord(im_file.read(1))
                    except:
                        break

                    if now != 0x02 or now2 != 0x02:
                        break

                    A = ord(im_file.read(1))
                    B = ord(im_file.read(1))
                    width = (A << 8) | B

                    nowx = 0
                    nowv = 0
                    while True:
                        if nowx >= width:
                            nowv += 1
                            nowx = 0
                            if nowv == 4:
                                break

                        info = ord(im_file.read(1))
                        if info <= 128:
                            data = im_file.read(info)
                            for i in range(info):
                                tmpdata[(nowy * width + nowx) * 4 + nowv] = data[i]
                                nowx += 1
                        else:
                            num = info - 128
                            data = ord(im_file.read(1))
                            for i in range(num):
                                tmpdata[(nowy * width + nowx) * 4 + nowv] = data
                                nowx += 1

                    nowy += 1

                tmpdata = tmpdata.reshape((height, width, 4))
            else:
                # Non-encoded HDR format
                totsize = width * height * 4
                tmpdata = struct.unpack('B' * totsize, im_file.read(totsize))
                tmpdata = np.asarray(tmpdata, np.uint8).reshape((height, width, 4))

            expo = np.power(2.0, tmpdata[:,:,3] - 128.0) / 256.0
            img = np.multiply(tmpdata[:,:,0:3], expo[:,:,np.newaxis])

        if img is None:
            raise Exception('Failed to load file "{0}"'.format(filename))

        return img
        
