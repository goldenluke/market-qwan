import numpy as np
from hmmlearn.hmm import GaussianHMM


class RobustHMM:

    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.model = None

    def _robust_transform(self, X):
        # Winsorize extremos
        lower = np.percentile(X, 1, axis=0)
        upper = np.percentile(X, 99, axis=0)
        return np.clip(X, lower, upper)

    def fit(self, returns):

        X = returns.values
        X = self._robust_transform(X)

        self.model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=200
        )

        self.model.fit(X)

    def predict(self, returns):
        X = self._robust_transform(returns.values)
        return self.model.predict(X)

    def predict_proba(self, returns):
        X = self._robust_transform(returns.values)
        return self.model.predict_proba(X)

    def transition_matrix(self):
        return self.model.transmat_