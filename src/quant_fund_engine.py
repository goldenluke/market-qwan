import numpy as np
import pandas as pd

from hmmlearn.hmm import GaussianHMM
from arch import arch_model
from sklearn.linear_model import Ridge
from sklearn.ensemble import RandomForestRegressor
from sklearn.covariance import LedoitWolf
from scipy.optimize import minimize
from scipy.stats import norm


class QuantFundEngine:

    def __init__(
        self,
        returns: pd.DataFrame,
        window=504,
        target_vol=0.15,
        cvar_limit=0.05
    ):

        self.returns = returns.dropna()
        self.window = window
        self.target_vol = target_vol
        self.cvar_limit = cvar_limit

        self.weights_history = []
        self.equity_curve = None

    # ==========================================================
    # 1️⃣ REGIME PROBABILÍSTICO
    # ==========================================================

    def fit_hmm(self, data):

        hmm = GaussianHMM(
            n_components=3,
            covariance_type="diag",
            n_iter=200,
            random_state=42
        )

        hmm.fit(data.values)
        probs = hmm.predict_proba(data.values)

        return probs[-1]

    # ==========================================================
    # 2️⃣ VOL CONDICIONAL GARCH
    # ==========================================================

    def estimate_vol(self, series):

        model = arch_model(series * 100, p=1, q=1)
        res = model.fit(disp="off")

        vol = res.conditional_volatility.iloc[-1] / 100
        return vol

    # ==========================================================
    # 3️⃣ ENSEMBLE FORECAST
    # ==========================================================

    def forecast_returns(self, data):

        X = data.shift(1).dropna()
        y = data.loc[X.index]

        ridge = Ridge()
        rf = RandomForestRegressor(n_estimators=50)

        ridge.fit(X, y)
        rf.fit(X, y)

        pred_ridge = ridge.predict(data.iloc[-1:].values)
        pred_rf = rf.predict(data.iloc[-1:].values)

        forecast = 0.5 * pred_ridge + 0.5 * pred_rf

        return forecast.flatten()

    # ==========================================================
    # 4️⃣ CVaR HARD CONSTRAINT
    # ==========================================================

    def cvar_constraint(self, weights, returns):

        portfolio_returns = returns @ weights
        alpha = 0.05

        var = np.percentile(portfolio_returns, 100 * alpha)
        cvar = portfolio_returns[portfolio_returns <= var].mean()

        return -cvar - self.cvar_limit

    # ==========================================================
    # 5️⃣ OPTIMIZAÇÃO INSTITUCIONAL
    # ==========================================================

    def optimize(self, mu, cov, returns_window):

        n = len(mu)

        def objective(w):
            port_return = w @ mu
            port_vol = np.sqrt(w @ cov @ w)
            return -(port_return / (port_vol + 1e-8))

        constraints = (
            {'type': 'eq', 'fun': lambda w: np.sum(w) - 1},
            {'type': 'ineq', 'fun': lambda w: self.cvar_constraint(w, returns_window)}
        )

        bounds = [(0, 1) for _ in range(n)]

        result = minimize(
            objective,
            np.ones(n) / n,
            bounds=bounds,
            constraints=constraints
        )

        return result.x

    # ==========================================================
    # 6️⃣ WALK-FORWARD REAL
    # ==========================================================

    def run(self):

        equity = [1.0]

        for t in range(self.window, len(self.returns) - 1):

            window_data = self.returns.iloc[t - self.window:t]

            # Regime
            regime_probs = self.fit_hmm(window_data)

            # Forecast
            mu = self.forecast_returns(window_data)

            # Cov robusta
            lw = LedoitWolf()
            cov = lw.fit(window_data).covariance_

            # Vol scaling
            vol = np.sqrt(np.diag(cov))
            mu = mu / (vol + 1e-8)

            # Optimize
            weights = self.optimize(mu, cov, window_data.values)

            # Vol targeting
            port_vol = np.sqrt(weights @ cov @ weights)

            if port_vol > 0:
                scale = self.target_vol / port_vol
                weights *= scale

            self.weights_history.append(weights)

            # Retorno fora da amostra
            ret = self.returns.iloc[t + 1].values @ weights
            equity.append(equity[-1] * (1 + ret))

        self.equity_curve = pd.Series(
            equity,
            index=self.returns.index[self.window - 1:]
        )

        return self.equity_curve