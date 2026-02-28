# src/hmm_regime.py

import numpy as np
from hmmlearn.hmm import GaussianHMM


class HMMRegimeDetector:

    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.model = GaussianHMM(
            n_components=n_regimes,
            covariance_type="full",
            n_iter=200
        )

    def fit(self, returns):
        self.model.fit(returns.values)
        hidden_states = self.model.predict(returns.values)
        return hidden_states

    def get_transition_matrix(self):
        return self.model.transmat_

    def posterior_probabilities(self, returns):
        posteriors = self.model.predict_proba(returns.values)
        return posteriors[-1]  # probabilidade do último período