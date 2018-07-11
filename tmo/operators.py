import numpy as np
from scipy.stats import gmean
import sys
from colour import oetf, RGB_COLOURSPACES, RGB_luminance


def multiply_scalar(intensity, factor=None, ev=0):
    eps = sys.float_info.epsilon
    intensity[np.isnan(intensity)] = eps
    intensity[intensity <= 0] = eps
    if factor is None:
        factor = 0.18 * 2**ev / gmean(intensity, axis=None)

    print("NaN: {0}".format(np.isnan(intensity).any()))
    print("INF: {0}".format(np.isinf(intensity).any()))
    print("factor: {0}".format(factor))
    return intensity * factor


def reinhard_tmo(hdrimage, ev=0, lum_white=float("inf")):
    colourspace = RGB_COLOURSPACES["sRGB"]
    lum = RGB_luminance(hdrimage, colourspace.primaries, colourspace.whitepoint)
    np.clip(lum, 0, np.finfo(np.float32).max, out=lum)
    scaled_lum = multiply_scalar(lum, ev=ev)
    mapped_lum = reinhard_curve(scaled_lum, scaled_lum, lum_white)
    ldrimage = replace_color(hdrimage, mapped_lum, lum)
    np.clip(ldrimage, 0, 1, out=ldrimage)
    return ldrimage


def reinhard_curve(lum, lum_ave, lum_white):
    lum_disp = (lum / (1 + lum_ave)) * (1 + (lum / (lum_white ** 2)))
    return lum_disp


def eilertsen_tmo(hdrimage, ev=0, exponent=0.9, sigma=0.6):
    colourspace = RGB_COLOURSPACES["sRGB"]
    lum = RGB_luminance(hdrimage, colourspace.primaries, colourspace.whitepoint)
    np.clip(lum, 0, np.finfo(np.float32).max, out=lum)
    scaled_lum = multiply_scalar(lum, ev=ev)
    scaled_image = replace_color(hdrimage, scaled_lum, lum)
    ldrimage = eilertsen_curve(scaled_image, exponent, sigma)
    np.clip(ldrimage, 0, 1, out=ldrimage)
    return ldrimage


def eilertsen_curve(intensity, exponent, sigma):
    powered_intensity = intensity ** exponent
    intensity_disp = (1 + sigma) * (powered_intensity / (powered_intensity + sigma))
    return intensity_disp


def replace_color(rgb, lum_new, lum_org):
    rgb_new = rgb
    ratio = lum_new / lum_org
    ratio[lum_org == 0] = 0
    rgb_new[:, :, 0] *= ratio
    rgb_new[:, :, 1] *= ratio
    rgb_new[:, :, 2] *= ratio
    return rgb_new


if __name__ == "__main__":
    sys.path.append("./io/")
    import imread
    from pathlib import Path

    hdrpath = Path("./") / "data" / "memorial_o876.hdr"
    hdrimage = imread.imread(str(hdrpath))
    ldrimage = reinhard_tmo(hdrimage, ev=0)
    ldrimage = oetf(ldrimage, function="sRGB")

    import matplotlib.pyplot as plt
    plt.imshow(ldrimage)
    plt.show()

    ldrimage = eilertsen_tmo(hdrimage, 0.9, 1)
    ldrimage = oetf(ldrimage, function="sRGB")
    plt.imshow(ldrimage)
    plt.show()
