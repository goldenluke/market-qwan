import numpy as np
from sklearn.metrics import mutual_info_score


class QWANCore:

    def __init__(self, returns, regimes=None):
        self.returns = returns
        self.regimes = regimes

    # --------------------------------------------------
    # 2. ENTROPIA ESTRUTURADA
    # --------------------------------------------------
    def structured_entropy(self, weights):

        portfolio_returns = self.returns.values @ weights
        p = np.abs(portfolio_returns)
        p = p / (np.sum(p) + 1e-12)

        entropy = -np.sum(p * np.log(p + 1e-12))
        return entropy

    # --------------------------------------------------
    # 3. COERÊNCIA ESTRUTURAL (I(X;M))
    # --------------------------------------------------
    def structural_coherence(self, weights):

        if self.regimes is None:
            return 0

        portfolio_returns = self.returns.values @ weights

        disc = np.digitize(
            portfolio_returns,
            bins=np.histogram(portfolio_returns, bins=10)[1]
        )

        min_len = min(len(disc), len(self.regimes))
        return mutual_info_score(
            disc[:min_len],
            self.regimes[:min_len]
        )
    
import numpy as np
from sklearn.metrics import mutual_info_score


class QWANCore:

    def __init__(self, returns, regimes=None):
        self.returns = returns
        self.regimes = regimes

    # --------------------------------------------------
    # 2. ENTROPIA ESTRUTURADA
    # --------------------------------------------------
    def structured_entropy(self, weights):

        portfolio_returns = self.returns.values @ weights
        p = np.abs(portfolio_returns)
        p = p / (np.sum(p) + 1e-12)

        entropy = -np.sum(p * np.log(p + 1e-12))
        return entropy

    # --------------------------------------------------
    # 3. COERÊNCIA ESTRUTURAL (I(X;M))
    # --------------------------------------------------
    def structural_coherence(self, weights):

        if self.regimes is None:
            return 0

        portfolio_returns = self.returns.values @ weights

        disc = np.digitize(
            portfolio_returns,
            bins=np.histogram(portfolio_returns, bins=10)[1]
        )

        min_len = min(len(disc), len(self.regimes))
        return mutual_info_score(
            disc[:min_len],
            self.regimes[:min_len]
        )