import numpy as np

def adaptive_threshold(chi, vol_series, base_percentile=80):

    thresholds = np.full(len(chi), np.nan)

    for t in range(252, len(chi)):

        vol = vol_series[t]
        adj_percentile = base_percentile + (vol * 100)

        adj_percentile = min(95, max(60, adj_percentile))

        thresholds[t] = np.nanpercentile(
            chi[t-252:t], adj_percentile
        )

    return thresholds