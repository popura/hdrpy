import re
import struct
import numpy as np
import OpenEXR
import Imath
import cv2
from pathlib import Path
from colour import RGB_COLOURSPACES


class RadianceHDRReader():
    def __init__(self, ):
        self.HDR_NONE = 0x00
        self.HDR_RLE_RGBE_32 = 0x01
        return

    def imread(self, path):
        with open(path, "rb") as im_file:
            # Read header section
            header = RadianceHDRHeader()
            header.read_header(im_file)

            # Read body section
            body = RadianceHDRBody()
            body.read_body()

        if img is None:
            raise Exception('Failed to load file "{0}"'.format(path))

        return img


class RadianceHDRHeader():
    def __init__(self, variables=None):
        # dictionary of meta data
        # e.g. {"FORMAT": "32-bit_rle_rgbe", "EXPOSURE": 1}
        if variables is None:
            self.variables = {}
        else:
            self.variables = variables

    # im_file: A file object of .pic or .hdr.
    def read_header(self, im_file):
        bufsize = 4096
        im_file.seek(0)
        buf = im_file.readline(bufsize).decode("ascii")
        if buf == "#?RADIANCE\n" or buf == "#?RGBE\n":
            pass
        else:
            raise Exception("HDR header is invalid!!")

        pattern = re.compile("([A-Z_]+)=(.*)")
        while True:
            buf = im_file.readline(bufsize).decode('ascii')
            if buf[0] == "\n":
                # Header section ends
                break
            m = pattern.match(buf)
            if m is None:
                continue
            else:
                self.variables[m.group(1)] = m.group(2)

        if not self.is_valid():
            raise Exception("HDR header is invalid!!")

        self.header_end = im_file.tell()
        return

    def is_valid(self):
        format = self.variables["FORMAT"]
        rgbe = "32-bit_rle_rgbe"
        xyze = "32-bit_rle_xyze"
        if format == rgbe or format == xyze:
            return True
        else:
            return False


class RadianceHDRBody():
    def __init__(self, height=0, width=0, pixel_values=None):
        self.height = height
        self.width = width

        if pixel_values is None:
            self.pixel_values = []
        else:
            self.pixel_values = pixel_values

        self.is_fliplr = False
        self.is_flipud = False
        self.is_rotated = False
        return

    # im_file: A file object of .pic or .hdr.
    # header: A RadianceHDRHeader object
    def read_body(self, im_file, header):
        im_file.seek(header.header_end)

        self.read_resolution_string(im_file)
        self.record_start = im_file.tell()

        is_rle = self.is_run_length_encoded()
        im_file.seek(self.record_start)

        if is_rle:
            # Run length encoded HDR
            tmpdata = np.zeros((self.width * self.height * 4), dtype=np.uint8)
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
            tmpdata = tmpdata.reshape((self.height, self.width, 4))
        else:
            # Non-encoded HDR format
            totsize = self.width * self.height * 4
            tmpdata = struct.unpack('B' * totsize, im_file.read(totsize))
            tmpdata = np.asarray(tmpdata, np.uint8).reshape((self.height, self.width, 4))

        expo = np.power(2.0, tmpdata[:, :, 3] - 128.0) / 256.0
        img = np.multiply(tmpdata[:, :, 0:3], expo[:, :, np.newaxis])

    def is_run_length_encoded(self, im_file):
        im_file.seek(self.record_start)
        now = ord(im_file.read(1))
        now2 = ord(im_file.read(1))
        if now == 0x02 and now2 == 0x02:
            return True
        else:
            return False

    def read_resolution_string(self, im_file):
        bufsize = 4096
        buf = im_file.readline(bufsize).decode()
        height_pattern = re.compile("([\-\+]Y) ([0-9]+)")
        height_match = height_pattern.search(buf)

        width_pattern = re.compile("([\-\+]X) ([0-9]+)")
        width_match = width_pattern.search(buf)

        if height_match is not None and width_match is not None:
            self.height = int(height_match.group(2))
            if height_match == "+Y":
                self.is_flipud = True
            self.width = int(width_match.group(2))
            if width_match == "-X":
                self.is_fliplr = True
            if height_match.start() > width_match.start():
                self.is_rotated = True
        else:
            raise Exception('HDR image size is invalid!!')
