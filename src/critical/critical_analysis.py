import numpy as np


def correlation_with_vix(critical_series, vix_series):

    mask = ~np.isnan(critical_series)
    return np.corrcoef(critical_series[mask], vix_series[mask])[0,1]