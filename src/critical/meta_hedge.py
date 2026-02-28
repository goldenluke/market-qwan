import numpy as np


def optimize_hedge_exposure(returns, chi, exposures=[0.3,0.4,0.5,0.6]):

    best_sharpe = -np.inf
    best_exposure = exposures[0]

    for e in exposures:

        hedge = np.where(chi > np.nanpercentile(chi, 80), e, 1.0)
        adjusted = returns * hedge

        sharpe = np.mean(adjusted) / np.std(adjusted)

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_exposure = e

    return best_exposure