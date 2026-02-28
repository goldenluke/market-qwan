import numpy as np
from pomegranate import HiddenMarkovModel, StudentTDistribution


class StudentTHMM:

    def __init__(self, n_regimes=3):
        self.n_regimes = n_regimes
        self.model = None

    def fit(self, returns):

        self.model = HiddenMarkovModel.from_samples(
            StudentTDistribution,
            n_components=self.n_regimes,
            X=returns.values
        )

    def predict(self, returns):
        return np.array(self.model.predict(returns.values))

    def predict_proba(self, returns):
        return np.array(self.model.predict_proba(returns.values))

    def transition_matrix(self):
        return self.model.dense_transition_matrix()