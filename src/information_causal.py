import numpy as np
from sklearn.feature_selection import mutual_info_regression


class PredictiveInformation:

    @staticmethod
    def predictive_mi(portfolio_returns):

        future_returns = np.roll(portfolio_returns, -1)

        X = portfolio_returns[:-1].reshape(-1, 1)
        y = future_returns[:-1]

        return mutual_info_regression(X, y)[0]