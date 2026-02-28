import numpy as np
from src.qwan_complexity import susceptibility, structure_factor


def rolling_susceptibility(returns, window=252):

    returns = np.asarray(returns)
    chi_series = np.full(len(returns), np.nan)

    for i in range(window, len(returns)):
        window_data = returns[i-window:i]
        chi_series[i] = susceptibility(window_data)

    return chi_series

def rolling_structure_slope(returns, window=252):

    slope_series = np.full(len(returns), np.nan)

    for i in range(window, len(returns)):

        window_data = returns[i-window:i]
        Sk = structure_factor(window_data)

        k = np.arange(1, len(Sk))
        logk = np.log(k)
        logSk = np.log(Sk[1:] + 1e-8)

        coef = np.polyfit(logk, logSk, 1)
        slope_series[i] = coef[0]

    return slope_series