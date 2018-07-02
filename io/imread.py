from pathlib import Path

def imread(path):
    path = Path(path)
    ext = path.suffix
    reader = ImReader(ext)
    image = reader.imread(str(path))

    return image
