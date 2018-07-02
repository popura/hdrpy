class ImReader():
    def __init__(self, ext):
        try:
            reader = self.generate_reader(ext)
        except TypeError as e:
            print(e)
        
        return reader

    def generate_reader(self, ext):
        if ext == ".hdr":
            reader = HdrReader
        elif ext == ".exr":
            reader = ExrReader
        elif ext == ".pfm":
            reader = PfmReader
        else:
            raise TypeError(ext + "is not an HDR image")

        return reader


