import numpy as np
from src.qwan_complexity import susceptibility

def cross_asset_criticality(returns_matrix, window=252):
    """
    returns_matrix: shape (T, N_assets)
    """

    T, N = returns_matrix.shape
    chi_matrix = np.full((T, N), np.nan)

    for asset in range(N):
        for t in range(window, T):
            chi_matrix[t, asset] = susceptibility(
                returns_matrix[t-window:t, asset]
            )

    return chi_matrix


def portfolio_overlay(weights, chi_matrix, percentile=80):

    T, N = chi_matrix.shape
    adjusted_weights = weights.copy()

    for t in range(T):
        threshold = np.nanpercentile(chi_matrix[t], percentile)
        mask = chi_matrix[t] > threshold
        adjusted_weights[t, mask] *= 0.5

    return adjusted_weights