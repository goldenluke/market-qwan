import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.covariance import LedoitWolf


class RegimeAwareQWAN:

    def __init__(
        self,
        returns: pd.DataFrame,
        n_regimes=3,
        alpha=0.6,
        gamma=0.5,
        target_vol=0.15,
        lookback_momentum=60
    ):

        self.returns = returns.dropna()
        self.n_regimes = n_regimes
        self.alpha = alpha
        self.gamma = gamma
        self.target_vol = target_vol
        self.lookback_momentum = lookback_momentum

        self._fit_hmm()
        self._compute_posteriors()
        self._compute_regime_weights()  # 🔥 IMPORTANTE

    # ==========================================================
    # HMM
    # ==========================================================

    def _fit_hmm(self):

        self.hmm = GaussianHMM(
            n_components=self.n_regimes,
            covariance_type="diag",
            n_iter=300,
            random_state=42
        )

        self.hmm.fit(self.returns.values)

        self.hidden_states = self.hmm.predict(self.returns.values)
        self.posterior_probs = self.hmm.predict_proba(self.returns.values)

        self.transition_matrix = self.hmm.transmat_
        self.start_prob = self.hmm.startprob_

    def _compute_posteriors(self):
        self.posterior_probs = self.hmm.predict_proba(self.returns.values)

    # ==========================================================
    # COMPUTAR PESOS POR REGIME
    # ==========================================================

    def _compute_regime_weights(self):

        self.regime_weights = {}

        for k in range(self.n_regimes):
            self.regime_weights[k] = self.optimize_for_regime(k)

    # ==========================================================
    # REGIME ATUAL
    # ==========================================================

    def get_current_regime(self):
        return np.argmax(self.posterior_probs[-1])

    def get_current_regime_weights(self):
        regime = self.get_current_regime()
        return self.regime_weights[regime]

    # ==========================================================
    # ENTROPIA
    # ==========================================================

    def entropy(self, weights):
        weights = np.clip(weights, 1e-8, None)
        weights = weights / np.sum(weights)
        return -np.sum(weights * np.log(weights))

    # ==========================================================
    # COVARIÂNCIA ROBUSTA
    # ==========================================================

    def robust_covariance(self, data):

        lw = LedoitWolf()
        cov = lw.fit(data).covariance_

        eigvals, eigvecs = np.linalg.eigh(cov)
        eigvals = np.clip(eigvals, 1e-6, None)

        cov = eigvecs @ np.diag(eigvals) @ eigvecs.T

        return cov

    # ==========================================================
    # MÉDIA BAYESIANA
    # ==========================================================

    def bayesian_mean(self, data):

        mu_sample = data.mean().values
        tau = 0.1
        mu_bayes = mu_sample / (1 + tau)

        return mu_bayes

    # ==========================================================
    # MOMENTUM
    # ==========================================================

    def momentum(self, data):

        lookback = min(self.lookback_momentum, len(data))
        return data.iloc[-lookback:].mean().values

    # ==========================================================
    # Φ
    # ==========================================================

    def phi(self, weights, regime):

        mask = self.hidden_states == regime

        if np.sum(mask) < 20:
            return -1e6

        regime_data = self.returns.iloc[mask]

        mu = self.bayesian_mean(regime_data)
        cov = self.robust_covariance(regime_data)

        port_return = weights @ mu
        port_var = weights @ cov @ weights

        H = self.entropy(weights)

        return port_return - self.alpha * H - 0.5 * port_var

    # ==========================================================
    # PESOS PROBABILÍSTICOS
    # ==========================================================

    def get_probabilistic_weights(self):

        posterior = self.posterior_probs[-1]

        weights_matrix = np.array(
            [self.regime_weights[k] for k in range(self.n_regimes)]
        )

        final_weights = np.sum(
            posterior[:, None] * weights_matrix,
            axis=0
        )

        final_weights = np.clip(final_weights, 0, None)

        if final_weights.sum() == 0:
            final_weights = np.ones_like(final_weights)

        final_weights /= final_weights.sum()

        return final_weights

    # ==========================================================
    # OTIMIZAÇÃO POR REGIME
    # ==========================================================

    def optimize_for_regime(self, regime):

        mask = self.hidden_states == regime

        if np.sum(mask) < 20:
            return np.ones(self.returns.shape[1]) / self.returns.shape[1]

        regime_data = self.returns.iloc[mask]

        mu = self.bayesian_mean(regime_data)
        cov = self.robust_covariance(regime_data)
        mom = self.momentum(regime_data)

        score = mu + self.gamma * mom

        inv_cov = np.linalg.pinv(cov)

        raw_weights = inv_cov @ score
        raw_weights = np.clip(raw_weights, 0, None)

        if raw_weights.sum() == 0:
            raw_weights = np.ones_like(raw_weights)

        weights = raw_weights / raw_weights.sum()

        # Target Vol Scaling
        port_vol = np.sqrt(weights @ cov @ weights)

        if port_vol > 0:
            scale = self.target_vol / port_vol
            weights = weights * scale

        return weights