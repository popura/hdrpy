
class RadianceHDRFormat(object):
    pass


class RadianceHDRReader():
    def __init__(self, ):
        self.HDR_NONE = 0x00
        self.HDR_RLE_RGBE_32 = 0x01
        return

    @classmethod
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
            raise Exception('Failed to load file "{0}"'.format(path))

        return img

    def read_header(self, im_file):
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


    def save(filename, img):
        """
        Save .hdr format
        """
        with open(filename, 'wb') as f:
            # Write header
            ret = 0x0a
            strwrite(f, '#?RADIANCE%c' % ret)
            strwrite(f, '# Generated by hdrpy%c' % ret)
            strwrite(f, 'FORMAT=32-bit_rle_rgbe%c' % ret)
            strwrite(f, 'EXPOSURE=1.0000000000000%c%c' % (ret, ret))

            # Write size
            [height, width, dim] = img.shape
            if dim != 3:
                raise Exception('HDR image must have 3 channels')

            strwrite(f, '-Y %d +X %d%c' % (height, width, ret))

            line = np.zeros((width, 4))
            for i in range(height):
                f.write(struct.pack('BBBB', 0x02, 0x02, (width >> 8) & 0xff, width & 0xff))

                d = np.max(img[i], axis=1)
                zero_ids = np.where(d < hydra.core.EPS)
                m, ie = np.frexp(d)
                d = m * 256.0 / d
                d[zero_ids] = 0.0

                line[:, :3] = img[i] * np.tile(d[:, np.newaxis], [1, 3])
                line[:, 3] = ie + 128
                line = np.clip(line, 0.0, 255.0).astype(np.uint8)

                buf = []
                for ch in range(4):
                    cursor = 0
                    while cursor < width:
                        cursor_move = min(127, width - cursor)
                        buf.append(cursor_move)
                        for j in range(cursor, cursor + cursor_move):
                            buf.append(line[j, ch])
                        cursor += cursor_move

            f.write(struct.pack('B' * len(buf), *buf))


if __name__ == "__main__":
    import doctest
    doctest.testmod()