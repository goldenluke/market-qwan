# src/information.py

import numpy as np
from sklearn.metrics import mutual_info_score

class InformationTheory:

    @staticmethod
    def entropy(w):
        w = np.clip(w, 1e-12, None)
        w = w / np.sum(w)
        return -np.sum(w * np.log(w))

    @staticmethod
    def mutual_information(portfolio_returns, regimes):
        bins = np.histogram_bin_edges(portfolio_returns, bins=5)
        port_disc = np.digitize(portfolio_returns, bins)
        regimes = regimes[-len(port_disc):]
        return mutual_info_score(port_disc, regimes)