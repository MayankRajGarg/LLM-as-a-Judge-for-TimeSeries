import numpy as np
from scipy.stats import zscore

def percent_change(prev, curr):
    return ((curr - prev) / prev) * 100

def compute_zscore(series, index):
    zs = zscore(series)
    return float(zs[index])

def slope(x, y):
    coef = np.polyfit(x, y, 1)
    return float(coef[0])
