import numpy as np
import pandas as pd
from scipy.stats import t
from hmmlearn.base import _BaseHMM


class StudentTHMM(_BaseHMM):
    """
    Hidden Markov Model com emissões Student-t.
    Mais robusto a outliers do que GaussianHMM.
    """

    def __init__(
        self,
        n_components=3,
        df=5,
        n_iter=100,
        tol=1e-3,
        random_state=42
    ):

        super().__init__(
            n_components=n_components,
            covariance_type="full",
            n_iter=n_iter,
            tol=tol,
            random_state=random_state
        )

        self.df = df  # graus de liberdade Student-t

    # ==========================================================
    # Inicialização
    # ==========================================================

    def _init(self, X, lengths=None):

        super()._init(X, lengths)

        n_features = X.shape[1]

        self.means_ = np.random.randn(self.n_components, n_features)
        self.covars_ = np.array([
            np.eye(n_features) for _ in range(self.n_components)
        ])

    # ==========================================================
    # Log Likelihood Student-t
    # ==========================================================

    def _compute_log_likelihood(self, X):

        n_samples, n_features = X.shape
        log_likelihood = np.zeros((n_samples, self.n_components))

        for k in range(self.n_components):

            mean = self.means_[k]
            cov = self.covars_[k]

            inv_cov = np.linalg.pinv(cov)
            det_cov = np.linalg.det(cov) + 1e-8

            diff = X - mean
            mahal = np.sum(diff @ inv_cov * diff, axis=1)

            term1 = np.log(np.exp(np.log(np.math.gamma((self.df + n_features)/2))
                                 - np.log(np.math.gamma(self.df/2))))
            term2 = -0.5 * np.log(det_cov)
            term3 = -((self.df + n_features)/2) * np.log(
                1 + mahal / self.df
            )

            log_likelihood[:, k] = term1 + term2 + term3

        return log_likelihood

    # ==========================================================
    # Atualização parâmetros (EM simplificado)
    # ==========================================================

    def _do_mstep(self, stats):

        super()._do_mstep(stats)

        # Regularização covariância
        for k in range(self.n_components):
            self.covars_[k] += 1e-6 * np.eye(self.covars_[k].shape[0])

    # ==========================================================
    # API amigável
    # ==========================================================

    def fit_model(self, returns):

        if isinstance(returns, pd.DataFrame):
            X = returns.values
        else:
            X = returns

        self.fit(X)

        return self

    def predict_states(self, returns):

        X = returns.values if isinstance(returns, pd.DataFrame) else returns
        return self.predict(X)

    def predict_proba_states(self, returns):

        X = returns.values if isinstance(returns, pd.DataFrame) else returns
        return self.predict_proba(X)

    def get_transition_matrix(self):

        return self.transmat_