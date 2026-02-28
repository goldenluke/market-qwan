import numpy as np


def future_drawdown_conditioned(returns, critical_series, threshold, horizon=21):

    future_dd = []
    high_critical_dd = []

    for i in range(len(returns) - horizon):

        future_window = returns[i:i+horizon]
        dd = np.min(np.cumsum(future_window))

        future_dd.append(dd)

        if critical_series[i] > threshold:
            high_critical_dd.append(dd)

    return {
        "mean_future_dd": np.mean(future_dd),
        "mean_dd_when_critical": np.mean(high_critical_dd)
    }