import numpy as np

def reinhard_curve(lum, lum_ave, lum_white):
    lum_disp = (lum / (1 + lum_ave)) * (1 + (lum / (lum_white ** 2)))
    return lum_disp

def eilertsen_curve(intensity, exponent, sigma):
    powered_intensity = intensity ** exponent
    intensity_disp = (1 + sigma) * (powered_intensity / (powered_intensity + sigma))
    return intensity_disp
