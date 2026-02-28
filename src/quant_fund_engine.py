import numpy as np
import pandas as pd

from src.qwan_regime_model import RegimeAwareQWAN
from src.hyperparameter_search import nested_alpha_gamma_search
from src.ensemble_engine import EnsembleEngine
from src.risk_model import risk_parity_weights
from src.risk_model import robust_covariance
from src.qwan_meta import MetaAllocator
from src.metrics import compute_metrics


class QuantFundEngine:

    def __init__(
        self,
        returns,
        lookback=252,
        rebalance_frequency=21,
        target_vol=0.15,
        stop_drawdown=-0.25
    ):

        self.returns = returns
        self.lookback = lookback
        self.rebalance_frequency = rebalance_frequency
        self.target_vol = target_vol
        self.stop_drawdown = stop_drawdown

        self.equity_curve = None
        self.benchmark_equity = None
        self.drawdown = None
        self.metrics = None

        self.run_walk_forward()

    # ==========================================================
    # WALK FORWARD ENGINE
    # ==========================================================

    def run_walk_forward(self):

        returns = self.returns
        n = len(returns)

        portfolio_returns = []
        benchmark_returns = []

        equity = 1.0
        equity_series = []

        benchmark_equity = 1.0
        benchmark_series = []

        peak = 1.0

        for start in range(self.lookback, n, self.rebalance_frequency):

            train_data = returns.iloc[start - self.lookback:start]
            test_data = returns.iloc[start:start + self.rebalance_frequency]

            if len(test_data) == 0:
                break

            # ==================================================
            # 1️⃣ Nested Hyperparameter Search
            # ==================================================

            best_alpha, best_gamma = nested_alpha_gamma_search(train_data)

            qwan_model = RegimeAwareQWAN(
                returns=train_data,
                alpha=best_alpha,
                gamma=best_gamma,
                target_vol=self.target_vol
            )

            # ==================================================
            # 2️⃣ Risk Parity Model
            # ==================================================

            cov = robust_covariance(train_data.values)
            rp_weights = risk_parity_weights(cov)

            class RPWrapper:
                def get_current_weights(self):
                    return rp_weights

            rp_model = RPWrapper()

            # ==================================================
            # 3️⃣ Ensemble
            # ==================================================

            ensemble = EnsembleEngine([qwan_model, rp_model])
            weights = ensemble.get_weights()

            # ==================================================
            # 4️⃣ Forward Test
            # ==================================================

            for t in range(len(test_data)):

                r = test_data.iloc[t].values @ weights
                b = test_data.iloc[t].mean()

                # Target volatility scaling
                realized_vol = np.std(train_data.values @ weights) * np.sqrt(252)
                scaling = self.target_vol / (realized_vol + 1e-8)
                r *= scaling

                # Stop estrutural
                equity *= (1 + r)

                peak = max(peak, equity)
                dd = (equity / peak) - 1

                if dd < self.stop_drawdown:
                    r = 0
                    equity = peak

                portfolio_returns.append(r)
                equity_series.append(equity)

                benchmark_equity *= (1 + b)
                benchmark_series.append(benchmark_equity)
                benchmark_returns.append(b)

        self.portfolio_returns = pd.Series(portfolio_returns)
        self.benchmark_returns = pd.Series(benchmark_returns)

        self.equity_curve = pd.Series(equity_series)
        self.benchmark_equity = pd.Series(benchmark_series)

        self.drawdown = self.equity_curve / self.equity_curve.cummax() - 1

        self.metrics = compute_metrics(
            self.portfolio_returns,
            self.benchmark_returns
        )

    # ==========================================================
    # ACCESSORS
    # ==========================================================

    def get_equity_curve(self):
        return self.equity_curve

    def get_benchmark_curve(self):
        return self.benchmark_equity

    def get_drawdown(self):
        return self.drawdown

    def get_metrics(self):
        return self.metrics