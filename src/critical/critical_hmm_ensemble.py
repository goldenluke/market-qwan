import numpy as np

def ensemble_signal(hmm_regime, critical_series,
                    critical_threshold):

    signal = []

    for regime, crit in zip(hmm_regime, critical_series):

        if regime == "high_vol" and crit > critical_threshold:
            signal.append(0.3)  # forte hedge

        elif regime == "high_vol":
            signal.append(0.6)

        elif crit > critical_threshold:
            signal.append(0.5)

        else:
            signal.append(1.0)

    return np.array(signal)