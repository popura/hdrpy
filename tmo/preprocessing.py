from scipy.stats import gmean
import sys


def multiply_scalar(intensity, factor=None, ev=0):
    eps = sys.float_info.epsilon
    replaced_intensity = intensity
    replaced_intensity[intensity == 0] = eps
    if factor is None:
        factor = 0.18 * 2**ev / gmean(intensity, axis=None)

    return intensity * factor
