import numpy as np
from scipy.stats import gmean
import sys

def multiply_scalar(intensity, factor=None, ev=0):
    eps = sys.float_info.epsilon
    if factor == None:
        factor = 0.18 * 2**ev
               / gmean(intensity, axis=None, zero_sub=eps)
    
    return intensity * factor
