import numpy as np
import pandas as pd
import scipy.stats as stats
import ruptures as rpt

from statsmodels.tsa.stattools import adfuller, kpss
from statsmodels.stats.diagnostic import acorr_ljungbox
from statsmodels.tsa.seasonal import STL
from statsmodels.tsa.api import VAR
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.regime_switching.markov_regression import MarkovRegression
from arch import arch_model
from scipy.fft import fft
from scipy.signal import periodogram
from sklearn.linear_model import LinearRegression


class AdvancedTimeSeriesAnalyzer:

    def __init__(self, returns: pd.Series):
        self.returns = returns.dropna()

    # ==========================================================
    # 1️⃣ Stationarity Tests
    # ==========================================================

    def stationarity_tests(self):

        adf = adfuller(self.returns)
        kpss_test = kpss(self.returns, regression='c')

        return {
            "ADF_stat": adf[0],
            "ADF_p": adf[1],
            "KPSS_stat": kpss_test[0],
            "KPSS_p": kpss_test[1]
        }

    # ==========================================================
    # 2️⃣ STL Decomposition
    # ==========================================================

    def stl_decomposition(self):

        stl = STL(self.returns, period=21)
        result = stl.fit()

        return result.trend, result.seasonal, result.resid

    # ==========================================================
    # 3️⃣ Hurst Exponent
    # ==========================================================

    def hurst_exponent(self):

        lags = range(2, 20)
        tau = [
            np.std(self.returns.diff(lag).dropna())
            for lag in lags
        ]

        reg = np.polyfit(np.log(lags), np.log(tau), 1)

        return reg[0] * 2

    # ==========================================================
    # 4️⃣ GARCH Volatility
    # ==========================================================

    def garch_volatility(self):

        model = arch_model(self.returns * 100, vol='Garch', p=1, q=1)
        res = model.fit(disp='off')

        return res.conditional_volatility / 100

    # ==========================================================
    # 5️⃣ Markov Switching
    # ==========================================================

    def markov_switching(self):

        model = MarkovRegression(
            self.returns,
            k_regimes=2,
            trend='c',
            switching_variance=True
        )

        res = model.fit()

        return res.smoothed_marginal_probabilities

    # ==========================================================
    # 6️⃣ Spectral Density
    # ==========================================================

    def spectral_analysis(self):

        freq, power = periodogram(self.returns)

        return freq, power

    # ==========================================================
    # 7️⃣ Change Point Detection
    # ==========================================================

    def change_points(self):

        algo = rpt.Pelt(model="rbf").fit(self.returns.values)
        result = algo.predict(pen=10)

        return result

    # ==========================================================
    # 8️⃣ EVT Tail Risk
    # ==========================================================

    def extreme_value_analysis(self, threshold_quantile=0.95):

        threshold = self.returns.quantile(threshold_quantile)
        excess = self.returns[self.returns > threshold] - threshold

        shape, loc, scale = stats.genpareto.fit(excess)

        return {
            "shape": shape,
            "scale": scale
        }

    # ==========================================================
    # 9️⃣ VAR Multivariado
    # ==========================================================

    @staticmethod
    def var_model(data: pd.DataFrame):

        model = VAR(data)
        res = model.fit(maxlags=5)

        return res.summary()

    # ==========================================================
    # 🔟 Entropia Rolling
    # ==========================================================

    def rolling_entropy(self, window=60):

        def entropy(x):
            hist = np.histogram(x, bins=10)[0]
            p = hist / np.sum(hist)
            p = p[p > 0]
            return -np.sum(p * np.log(p))

        return self.returns.rolling(window).apply(entropy)