# src/walkforward_regime.py

import numpy as np
from .qwan_regime_model import RegimeAwareQWAN


class WalkForwardRegime:

    def __init__(self,
                 train_window=750,
                 test_window=125,
                 n_regimes=3,
                 alpha=0.5,
                 gamma=0.5):

        self.train_window = train_window
        self.test_window = test_window
        self.n_regimes = n_regimes
        self.alpha = alpha
        self.gamma = gamma

    def run(self, returns):

        portfolio_returns = []

        for start in range(
            0,
            len(returns) - self.train_window - self.test_window,
            self.test_window
        ):

            train = returns.iloc[start:start+self.train_window]
            test = returns.iloc[
                start+self.train_window:
                start+self.train_window+self.test_window
            ]

            model = RegimeAwareQWAN(
                train,
                n_regimes=self.n_regimes,
                alpha=self.alpha,
                gamma=self.gamma
            )

            weights = model.get_probabilistic_weights()

            test_returns = test.values @ weights

            portfolio_returns.extend(test_returns)

        return np.array(portfolio_returns)