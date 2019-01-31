import struct
import numpy as np


class PfmReader():
    def __init__(self, ):
        return

    def imread(self, path):
        img = None
        with open(path, 'rb') as f:
            f.readline()
            w, h = f.readline().decode('ascii').strip().split(' ')
            w = int(w)
            h = int(h)
            f.readline()

            siz = h * w * 3
            img = np.array(struct.unpack('f' * siz, f.read(4 * siz)))
            img = img.reshape((h, w, 3))

        if img is None:
            raise Exception('Failed to load file "{0}"'.format(path))

        return img


class PfmWriter():
    def __init__(self, ):
        return

    def imwrite(self, filename, img):
        h, w, dim = img.shape
        with open(filename, 'wb') as f:
            f.write(bytearray('PF\n', 'ascii'))
            f.write(bytearray('{0:d} {1:d}\n'.format(w, h), 'ascii'))
            f.write(bytearray('-1.0\n', 'ascii'))

            siz = h * w * 3
            tmp = img.reshape(siz)
            f.write(struct.pack('f'*siz, *tmp))
