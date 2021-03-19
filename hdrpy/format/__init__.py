from hdrpy.format.format import Format
from hdrpy.format.pfm import PFMFormat
from hdrpy.format.radiance_hdr import RadianceHDRFormat

try:
    from hdrpy.format.openexr import OpenEXRFormat
except ImportError:
    pass