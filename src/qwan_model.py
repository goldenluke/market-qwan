# src/qwan_model.py

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .risk import RiskMetrics
from .information import InformationTheory
from .regimes import RegimeDetector


class MarketQWAN:

    def __init__(self, returns, alpha=0.5, beta=0.5):
        """
        returns: DataFrame de retornos logarítmicos
        alpha: peso da entropia (liberdade estrutural)
        beta: peso da informação mútua (coerência com regime)
        """

        self.returns = returns.copy()
        self.n_assets = returns.shape[1]
        self.alpha = alpha
        self.beta = beta

        # Detectar regimes
        self.regimes = RegimeDetector().detect(self.returns)

        # Garantir alinhamento inicial
        self._align_data()

    # ==========================================================
    # ALINHAMENTO TEMPORAL
    # ==========================================================

    def _align_data(self):
        """
        Garante que retornos e regimes tenham mesmo índice.
        Remove períodos inválidos.
        """

        common_index = self.returns.index.intersection(self.regimes.index)

        self.returns = self.returns.loc[common_index]
        self.regimes = self.regimes.loc[common_index]

    # ==========================================================
    # ENERGIA DE RISCO (CVaR)
    # ==========================================================

    def risk_energy(self, w):
        portfolio_returns = self.returns.values @ w
        return abs(RiskMetrics.cvar(portfolio_returns))

    # ==========================================================
    # FUNCIONAL Φ_market
    # ==========================================================

    def phi(self, w):

        # Normalizar pesos
        w = np.clip(w, 1e-12, None)
        w = w / np.sum(w)

        portfolio_returns = self.returns.values @ w

        # --- Energia (risco extremo)
        G = self.risk_energy(w)

        # --- Entropia estrutural
        H = InformationTheory.entropy(w)

        # --- Informação mútua (regime × retorno)
        I = InformationTheory.mutual_information(
            portfolio_returns,
            self.regimes.values
        )

        # Φ = Energia - α Entropia + β Informação
        phi_value = G - self.alpha * H + self.beta * I

        return phi_value

    # ==========================================================
    # OTIMIZAÇÃO
    # ==========================================================

    def optimize(self):

        def objective(w):
            return self.phi(w)

        # Restrição: soma dos pesos = 1
        constraints = ({
            'type': 'eq',
            'fun': lambda w: np.sum(w) - 1
        })

        # Sem short (long-only)
        bounds = [(0, 1)] * self.n_assets

        # Inicialização uniforme
        w0 = np.ones(self.n_assets) / self.n_assets

        result = minimize(
            objective,
            w0,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 200}
        )

        if not result.success:
            print("⚠ Otimização não convergiu:", result.message)

        w_opt = np.clip(result.x, 0, None)
        w_opt /= np.sum(w_opt)

        return w_opt