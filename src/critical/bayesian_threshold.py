import numpy as np
from skopt import gp_minimize
from src.critical.critical_overlay import apply_critical_overlay

def optimize_threshold_bayes(returns, chi):

    def objective(percentile):
        adjusted, _ = apply_critical_overlay(
            returns,
            chi,
            mode="threshold",
            threshold=np.nanpercentile(chi, percentile[0])
        )
        sharpe = np.mean(adjusted) / np.std(adjusted)
        return -sharpe

    result = gp_minimize(objective, [(60, 95)], n_calls=20)

    return result.x[0]