import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.covariance import LedoitWolf
from hmmlearn.hmm import GaussianHMM


# ==========================================================
# FUNÇÕES AUXILIARES ROBUSTAS
# ==========================================================

def robust_covariance(returns):

    lw = LedoitWolf().fit(returns)
    cov = lw.covariance_

    # Eigenvalue clipping
    eigvals, eigvecs = np.linalg.eigh(cov)
    eigvals = np.maximum(eigvals, 1e-5)
    cov_clipped = eigvecs @ np.diag(eigvals) @ eigvecs.T

    return cov_clipped


def risk_contribution_entropy(weights, cov):

    portfolio_var = weights @ cov @ weights
    marginal = cov @ weights
    RC = weights * marginal / portfolio_var

    RC = np.clip(RC, 1e-8, None)
    RC = RC / RC.sum()

    return -np.sum(RC * np.log(RC))


def cvar(returns, alpha=0.05):

    var = np.percentile(returns, alpha * 100)
    tail = returns[returns <= var]

    if len(tail) == 0:
        return 0

    return np.mean(tail)


# ==========================================================
# MODELO QWAN REGIME AWARE ROBUSTO
# ==========================================================

class RegimeAwareQWAN:

    def __init__(
        self,
        returns: pd.DataFrame,
        n_regimes: int = 3,
        alpha: float = 0.6,
        gamma: float = 0.5,
        target_vol: float = 0.15,
        lambda_turnover: float = 1.0,
        cvar_limit: float = 0.05
    ):

        self.returns = returns.dropna()
        self.n_regimes = n_regimes
        self.alpha = alpha
        self.gamma = gamma
        self.target_vol = target_vol
        self.lambda_turnover = lambda_turnover
        self.cvar_limit = cvar_limit

        self.hmm = None
        self.posterior_probs = None
        self.hidden_states = None
        self.regime_weights = {}
        self.transition_matrix = None

        self._fit_hmm()
        self._optimize_all_regimes()

    # ======================================================
    # HMM FIT
    # ======================================================

    def _fit_hmm(self):

        model = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="full",
            n_iter=200
        )

        model.fit(self.returns.values)

        self.hmm = model
        self.posterior_probs = model.predict_proba(self.returns.values)
        self.hidden_states = model.predict(self.returns.values)
        self.transition_matrix = model.transmat_

    # ======================================================
    # FUNCIONAL Φ ROBUSTO
    # ======================================================

    def phi(self, weights, regime):

        regime_returns = self.returns[
            self.hidden_states == regime
        ]

        if len(regime_returns) < 30:
            return -1e6

        cov = robust_covariance(regime_returns.values)

        portfolio_returns = regime_returns.values @ weights

        # Ganho ajustado por risco
        G = np.mean(portfolio_returns) * 252

        # Entropia estrutural via contribuição ao risco
        H = risk_contribution_entropy(weights, cov)

        # Momentum com lookback móvel
        lookback = min(60, len(portfolio_returns))
        mu = np.mean(portfolio_returns[-lookback:]) * 252

        # Penalização por correlação média
        corr_penalty = np.mean(np.abs(cov))

        # CVaR
        cvar_val = cvar(portfolio_returns)
        cvar_penalty = 0
        if cvar_val < -self.cvar_limit:
            cvar_penalty = 10 * abs(cvar_val)

        return (
            G
            - self.alpha * H
            - self.gamma * mu
            - 0.1 * corr_penalty
            - cvar_penalty
        )

    # ======================================================
    # OTIMIZAÇÃO POR REGIME
    # ======================================================

    def optimize_for_regime(self, regime, w_prev=None):

        n = self.returns.shape[1]
        bounds = [(0, 1)] * n

        constraints = ({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1
        })

        w0 = np.ones(n) / n

        def objective(w):

            turnover_penalty = 0
            if w_prev is not None:
                turnover = np.sum(np.abs(w - w_prev))
                turnover_penalty = self.lambda_turnover * turnover

            return -self.phi(w, regime) + turnover_penalty

        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )

        return result.x

    # ======================================================
    # OTIMIZAR TODOS OS REGIMES
    # ======================================================

    def _optimize_all_regimes(self):

        w_prev = None

        for k in range(self.n_regimes):
            w_opt = self.optimize_for_regime(k, w_prev)
            self.regime_weights[k] = w_opt
            w_prev = w_opt

    # ======================================================
    # PESO ATUAL PONDERADO POR POSTERIOR
    # ======================================================

    def get_current_weights(self):

        posterior_last = self.posterior_probs[-1]

        weights = np.zeros(self.returns.shape[1])

        for k in range(self.n_regimes):
            weights += posterior_last[k] * self.regime_weights[k]

        # Vol targeting
        cov_full = robust_covariance(self.returns.values)
        vol = np.sqrt(weights @ cov_full @ weights) * np.sqrt(252)

        if vol > 0:
            weights *= self.target_vol / vol

        weights = np.clip(weights, 0, None)
        weights = weights / weights.sum()

        return weights

    # ======================================================
    # REGIME ATUAL (SOFT)
    # ======================================================

    def get_current_regime(self):

        posterior_last = self.posterior_probs[-1]
        return np.argmax(posterior_last)


# ==========================================================
# PESOS BAYESIANOS MULTI-REGIME
# ==========================================================

    def get_probabilistic_weights(self):

        # Caso não tenha posterior, usar pesos atuais
        if not hasattr(self, "posterior_probs"):
            return self.get_current_weights()

        if not hasattr(self, "regime_weights"):
            return self.get_current_weights()

        posterior_last = self.posterior_probs[-1]

        weights = np.zeros_like(self.regime_weights[0])

        for k in range(self.n_regimes):
            weights += posterior_last[k] * self.regime_weights[k]

        # Normalização defensiva
        weights = np.clip(weights, 0, None)
        if weights.sum() == 0:
            weights = np.ones_like(weights) / len(weights)
        else:
            weights /= weights.sum()

        return weights