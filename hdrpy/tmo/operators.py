import copy
import numpy as np
from scipy.stats import gmean
import sys
from colour import oetf, RGB_COLOURSPACES, RGB_luminance
import cv2


def multiply_scalar(intensity, factor=None, ev=0):
    eps = sys.float_info.epsilon
    intensity = copy.deepcopy(intensity)
    intensity[np.isnan(intensity)] = eps
    intensity[intensity <= 0] = eps
    if factor is None:
        factor = 0.18 * 2**ev / gmean(intensity, axis=None)

    if np.isnan(gmean(intensity, axis=None)):
        print("NaN: {0}".format(np.isnan(intensity).any()))
    if np.isinf(gmean(intensity, axis=None)):
        print("INF: {0}".format(np.isinf(intensity).any()))
    #print("factor: {0}".format(factor))
    new_intensity = intensity * factor
    np.clip(new_intensity, 0, np.finfo(np.float32).max, out=new_intensity)
    return new_intensity


def normalize_luminance(hdrimage):
    colourspace = RGB_COLOURSPACES["sRGB"]
    hdrimage = np.clip(hdrimage, 0, np.finfo(np.float32).max)
    lum = RGB_luminance(hdrimage, colourspace.primaries, colourspace.whitepoint)
    max_lum = lum.max()
    ldrimage = replace_color(hdrimage, lum / max_lum, lum)
    return ldrimage


def normalize_color(hdrimage):
    eps = sys.float_info.epsilon
    hdrimage = np.clip(hdrimage, 0, np.finfo(np.float32).max)
    hdrimage[np.isnan(hdrimage)] = eps
    max_intensity = hdrimage.max()
    min_intensity = hdrimage.min()
    ldrimage = (hdrimage - min_intensity) / (max_intensity - min_intensity)
    return ldrimage


def reinhard_tmo(hdrimage, ev=0, lum_white=float("inf")):
    colourspace = RGB_COLOURSPACES["sRGB"]
    hdrimage = np.clip(hdrimage, 0, np.finfo(np.float32).max)
    lum = RGB_luminance(hdrimage, colourspace.primaries, colourspace.whitepoint)
    scaled_lum = multiply_scalar(lum, ev=ev)
    mapped_lum = reinhard_curve(scaled_lum, scaled_lum, lum_white)
    ldrimage = replace_color(hdrimage, mapped_lum, lum)
    #np.clip(ldrimage, 0, 1, out=ldrimage)
    return ldrimage


def reinhard_curve(lum, lum_ave, lum_white):
    lum = np.clip(lum, 0, np.finfo(np.float32).max)
    lum_ave = np.clip(lum_ave, 0, np.finfo(np.float32).max)
    lum_disp = (lum / (1 + lum_ave)) * (1 + (lum / (lum_white ** 2)))
    return lum_disp


def eilertsen_tmo(hdrimage, ev=0, exponent=0.9, sigma=0.6):
    colourspace = RGB_COLOURSPACES["sRGB"]
    clipped_image = np.clip(hdrimage, 0, np.finfo(np.float32).max)
    lum = RGB_luminance(clipped_image, colourspace.primaries, colourspace.whitepoint)
    scaled_lum = multiply_scalar(lum, ev=ev)
    scaled_image = replace_color(clipped_image, scaled_lum, lum)
    ldrimage = eilertsen_curve(scaled_image, exponent, sigma)
    #np.clip(ldrimage, 0, 1, out=ldrimage)
    return ldrimage


def eilertsen_curve(intensity, exponent, sigma):
    powered_intensity = intensity ** exponent
    intensity_disp = (1 + sigma) * (powered_intensity / (powered_intensity + sigma))
    return intensity_disp


def replace_color(rgb, lum_new, lum_org):
    rgb_new = copy.deepcopy(rgb)
    clipped_lum = np.clip(lum_new, 0, np.finfo(np.float32).max)
    ratio = clipped_lum / lum_org
    ratio[lum_org == 0] = 0
    rgb_new[:, :, 0] *= ratio
    rgb_new[:, :, 1] *= ratio
    rgb_new[:, :, 2] *= ratio
    return rgb_new


if __name__ == "__main__":
    import hdrpy.io.imread as imread
    from pathlib import Path

    hdrpath = Path("./") / "data" / "memorial_o876.hdr"
    hdrimage = imread.imread(str(hdrpath))
    cp_hdrimage = copy.deepcopy(hdrimage)
    ldrimage = reinhard_tmo(hdrimage, ev=0)
    print((hdrimage==cp_hdrimage).all())
    ldrimage = oetf(ldrimage, function="sRGB")

    import matplotlib.pyplot as plt
    plt.imshow(ldrimage)
    plt.show()

    ldrimage = eilertsen_tmo(hdrimage, 0.9, 1)
    print((hdrimage==cp_hdrimage).all())
    ldrimage = oetf(ldrimage, function="sRGB")
    plt.imshow(ldrimage)
    plt.show()

    ldrimage = normalize_luminance(hdrimage)
    ldrimage = oetf(ldrimage, function="sRGB")
    plt.imshow(ldrimage)
    plt.show()
    colourspace = RGB_COLOURSPACES["sRGB"]
    lum = RGB_luminance(hdrimage, colourspace.primaries, colourspace.whitepoint)
    print(lum.max())
    print(lum.min())


    image_list = [reinhard_tmo(hdrimage, ev=0.25*j, lum_white=1)
                  for j in range(-1, 2)]
    print(len(image_list))
    for j in range(len(image_list)):
        print(image_list[j].shape)
        image_list[j] = np.clip((image_list[j][:, :, ::-1] ** (1/2.2))*255, 0, 255)
    merge_mertens = cv2.createMergeMertens(1., 1., 1.)
    ldrimage = merge_mertens.process(image_list)
    plt.imshow(np.clip(ldrimage[:, :, ::-1], 0, 1))
    plt.show()
